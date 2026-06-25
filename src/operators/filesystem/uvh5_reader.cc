#include <cassert>
#include <cstring>
#include <complex>

#include <hdf5.h>

#include <stelline/types.hh>
#include <stelline/operators/filesystem/base.hh>
#include <stelline/utils/tensor.hh>
#include <fmt/format.h>

#include <matx.h>
#include <cuda/std/complex>

using namespace holoscan;

namespace stelline::operators::filesystem {

// UVH5 HDF5 group / dataset paths (pyuvdata / UVH5 specification).
static constexpr const char* UVH5_HEADER_GROUP = "Header";
static constexpr const char* UVH5_DATA_GROUP    = "Data";
static constexpr const char* UVH5_VISDATA_DS    = "Data/visdata";
static constexpr const char* UVH5_TIMES_DS      = "Data/times";

static hsize_t read_scalar_u64(hid_t file_id, const char* path) {
    hid_t dset = H5Dopen2(file_id, path, H5P_DEFAULT);
    hsize_t val = 0;
    if (dset >= 0) {
        H5Dread(dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
        H5Dclose(dset);
    } else {
        // Fall back to attribute on the Header group.
        hid_t grp = H5Gopen2(file_id, UVH5_HEADER_GROUP, H5P_DEFAULT);
        if (grp >= 0) {
            // Strip the group prefix when reading as attribute.
            const char* attr_name = std::strrchr(path, '/');
            attr_name = attr_name ? attr_name + 1 : path;
            hid_t attr = H5Aopen(grp, attr_name, H5P_DEFAULT);
            if (attr >= 0) {
                H5Aread(attr, H5T_NATIVE_UINT64, &val);
                H5Aclose(attr);
            }
            H5Gclose(grp);
        }
    }
    return val;
}

struct Uvh5ReaderOp::Impl {
    // Configuration.

    std::string filePath;
    hsize_t chunkSize; // number of blt entries per compute()

    // HDF5 state.
    UVH5_file_t uvh5;

    // Current read position (blt index).

    hsize_t readOffset;

    // CPU pinned bounce buffers.

    void* visBounceBuffer;
    size_t visBounceBytes;
    double* timesBounceBuffer; // one double per blt entry in the chunk

    cudaStream_t stream;

    // GPU output tensor: [chunkSize, nFreqs, nPols] complex64.

    std::shared_ptr<holoscan::Tensor> outputTensor;

    // Metrics.

    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;
    std::atomic<int64_t> bytesSinceLastMeasurement{0};
    std::atomic<double> currentBandwidthMBps{0.0};
    int64_t totalBytesRead{0};
    uint64_t chunkCounter{0};
};

void Uvh5ReaderOp::initialize() {
    pimpl = new Impl();
    Operator::initialize();
}

Uvh5ReaderOp::~Uvh5ReaderOp() {
    delete pimpl;
}

void Uvh5ReaderOp::setup(OperatorSpec& spec) {
    spec.output<std::shared_ptr<holoscan::Tensor>>("out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(filePath_, "file_path");
    spec.param(chunkSize_, "chunk_size");
}

void Uvh5ReaderOp::start() {
    pimpl->filePath  = filePath_.get();
    pimpl->chunkSize = static_cast<hsize_t>(chunkSize_.get());

    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

    pimpl->uvh5 = UVH5access_file(
		pimpl->filePath.c_str(),
		H5P_DEFAULT
	);
    if (pimpl->uvh5.file_id < 0) {
        HOLOSCAN_LOG_ERROR("UVH5 Reader: cannot open '{}'.", pimpl->filePath);
        throw std::runtime_error(fmt::format("UVH5 Reader: cannot open '{}'.", pimpl->filePath));
    }

    HOLOSCAN_LOG_INFO("UVH5 Reader: '{}' — Nblts={}, Nspws={}, Nfreqs={}, Npols={}, chunk_size={}.",
                      pimpl->filePath,
                      pimpl->uvh5.header.Nblts, pimpl->uvh5.header.Nspws, pimpl->uvh5.header.Nfreqs, pimpl->uvh5.header.Npols,
                      pimpl->chunkSize);
    
    UVH5change_access_chunking(&pimpl->uvh5, pimpl->chunkSize);

    // Allocate pinned bounce buffer: chunkSize × Nspws × Nfreqs × Npols × 8 bytes (complex64).
    pimpl->visBounceBytes = H5DSsize(&pimpl->uvh5.DS_data_visdata);
    if (cudaMallocHost(&pimpl->visBounceBuffer, pimpl->visBounceBytes) != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("UVH5 Reader: failed to allocate pinned bounce buffer.");
        UVH5close(&pimpl->uvh5);
        throw std::runtime_error("UVH5 Reader: bounce buffer allocation failed.");
    }

    // Timestamps bounce buffer.
    pimpl->timesBounceBuffer = nullptr;
    if (pimpl->timesDatasetId >= 0) {
        if (cudaMallocHost(reinterpret_cast<void**>(&pimpl->timesBounceBuffer),
                           pimpl->chunkSize * sizeof(double)) != cudaSuccess) {
            HOLOSCAN_LOG_WARN("UVH5 Reader: failed to allocate times bounce buffer.");
        }
    }

    // GPU output tensor: [chunkSize, nFreqs, nPols] complex64.
    // We flatten the Nspws dimension (always 1 for standard UVH5) into Nfreqs.
    auto t = matx::make_tensor<cuda::std::complex<float>>(
        {static_cast<matx::index_t>(pimpl->uvh5.DS_data_visdata.dimchunks[0]),
         static_cast<matx::index_t>(pimpl->uvh5.DS_data_visdata.dimchunks[1]),
         static_cast<matx::index_t>(pimpl->uvh5.DS_data_visdata.dimchunks[2])},
        matx::MATX_DEVICE_MEMORY);
    pimpl->outputTensor = std::make_shared<holoscan::Tensor>(t.ToDlPack());

    cudaStreamCreateWithFlags(&pimpl->stream, cudaStreamNonBlocking);

    pimpl->readOffset = 0;
    pimpl->totalBytesRead = 0;
    pimpl->chunkCounter = 0;
    pimpl->bytesSinceLastMeasurement = 0;
    pimpl->lastMeasurementTime = std::chrono::steady_clock::now();
    pimpl->currentBandwidthMBps = 0.0;
}

void Uvh5ReaderOp::stop() {
    cudaStreamDestroy(pimpl->stream);
    UVH5close(&pimpl->uvh5);
}

void Uvh5ReaderOp::compute(InputContext&, OutputContext& output, ExecutionContext&) {
    pimpl->readOffset = pimpl->uvh5.DS_data_visdata.hyperslab_start[0];
    herr_t status UVH5read(&pimpl->uvh5);
    if (status < 0) {
        HOLOSCAN_LOG_ERROR("UVH5 Reader: H5Dread (visdata) failed at blt offset {}.", pimpl->readOffset);
        throw std::runtime_error("UVH5 Reader: H5Dread failed.");
    }
    else if (status == 1) {
        HOLOSCAN_LOG_INFO("UVH5 Reader: looping back to start of '{}'.", pimpl->filePath);
    }

    // Copy CPU → GPU.
    cudaMemcpyAsync(pimpl->outputTensor->data(),
                    pimpl->uvh5.visdata,
                    pimpl->visBounceBytes,
                    cudaMemcpyHostToDevice,
                    pimpl->stream);
    cudaStreamSynchronize(pimpl->stream);

    pimpl->totalBytesRead += static_cast<int64_t>(pimpl->visBounceBytes);
    pimpl->bytesSinceLastMeasurement += static_cast<int64_t>(pimpl->visBounceBytes);
    pimpl->chunkCounter++;

    // Propagate the JD timestamp as metadata so downstream operators can use it.
    const auto& meta = metadata();
    meta->set("timestamp_jd", pimpl->uvh5.header.time_array[0]);
    meta->set("timestamp", pimpl->chunkCounter);

    output.emit(pimpl->outputTensor, "out");
}

void Uvh5ReaderOp::tick() {
    if (!pimpl || !metrics()) {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    auto elapsedSeconds = std::chrono::duration<double>(now - pimpl->lastMeasurementTime).count();

    if (elapsedSeconds > 0.0) {
        int64_t bytes = pimpl->bytesSinceLastMeasurement.exchange(0);
        pimpl->currentBandwidthMBps = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsedSeconds;
        pimpl->lastMeasurementTime = now;
    }

    metrics()->record("current_bandwidth_mb_s", fmt::format("{:.2f}", pimpl->currentBandwidthMBps.load()));
    metrics()->record("total_data_read_mb", fmt::format("{:.0f}", static_cast<double>(pimpl->totalBytesRead) / (1024.0 * 1024.0)));
    metrics()->record("chunks_read", fmt::format("{}", pimpl->chunkCounter));
}

std::string Uvh5ReaderOp::formatMetrics(const MetricsProvider::MetricsMap& metrics) {
    return fmt::format("  Current Bandwidth: {} MB/s\n"
                       "  Total Data Read: {} MB\n"
                       "  Chunks Read: {}",
                       metrics.at("current_bandwidth_mb_s").value,
                       metrics.at("total_data_read_mb").value,
                       metrics.at("chunks_read").value);
}

}  // namespace stelline::operators::filesystem
