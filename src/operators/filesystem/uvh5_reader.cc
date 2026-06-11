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

    hid_t fileId;
    hid_t visDatasetId;
    hid_t visDataspaceId;
    hid_t timesDatasetId;

    // File dimensions: visdata is [Nblts, Nspws, Nfreqs, Npols] complex64.

    hsize_t nBlts;
    hsize_t nSpws;
    hsize_t nFreqs;
    hsize_t nPols;

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
    spec.param(chunkSize_, "chunk_size", "chunk_size", static_cast<uint64_t>(1));
}

void Uvh5ReaderOp::start() {
    pimpl->filePath  = filePath_.get();
    pimpl->chunkSize = static_cast<hsize_t>(chunkSize_.get());

    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

    pimpl->fileId = H5Fopen(pimpl->filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (pimpl->fileId < 0) {
        HOLOSCAN_LOG_ERROR("UVH5 Reader: cannot open '{}'.", pimpl->filePath);
        throw std::runtime_error(fmt::format("UVH5 Reader: cannot open '{}'.", pimpl->filePath));
    }

    // Read dimensions from the Header group.
    // UVH5 spec stores Nblts, Nfreqs, Npols, Nspws as scalar datasets under /Header/.
    auto read_header_uint = [&](const char* name) -> hsize_t {
        std::string path = fmt::format("{}/{}", UVH5_HEADER_GROUP, name);
        hid_t dset = H5Dopen2(pimpl->fileId, path.c_str(), H5P_DEFAULT);
        if (dset < 0) {
            // Also try as an attribute on the Header group.
            hid_t grp = H5Gopen2(pimpl->fileId, UVH5_HEADER_GROUP, H5P_DEFAULT);
            if (grp >= 0) {
                hid_t attr = H5Aopen(grp, name, H5P_DEFAULT);
                if (attr >= 0) {
                    hsize_t val = 0;
                    H5Aread(attr, H5T_NATIVE_UINT64, &val);
                    H5Aclose(attr);
                    H5Gclose(grp);
                    return val;
                }
                H5Gclose(grp);
            }
            HOLOSCAN_LOG_WARN("UVH5 Reader: header field '{}' not found, defaulting to 1.", name);
            return 1;
        }
        hsize_t val = 0;
        H5Dread(dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
        H5Dclose(dset);
        return val;
    };

    pimpl->nBlts  = read_header_uint("Nblts");
    pimpl->nFreqs = read_header_uint("Nfreqs");
    pimpl->nPols  = read_header_uint("Npols");
    pimpl->nSpws  = read_header_uint("Nspws");
    if (pimpl->nSpws == 0) pimpl->nSpws = 1;

    HOLOSCAN_LOG_INFO("UVH5 Reader: '{}' — Nblts={}, Nspws={}, Nfreqs={}, Npols={}, chunk_size={}.",
                      pimpl->filePath,
                      pimpl->nBlts, pimpl->nSpws, pimpl->nFreqs, pimpl->nPols,
                      pimpl->chunkSize);

    // Open visdata dataset: [Nblts, Nspws, Nfreqs, Npols] complex64.
    pimpl->visDatasetId = H5Dopen2(pimpl->fileId, UVH5_VISDATA_DS, H5P_DEFAULT);
    if (pimpl->visDatasetId < 0) {
        HOLOSCAN_LOG_ERROR("UVH5 Reader: dataset '{}' not found.", UVH5_VISDATA_DS);
        H5Fclose(pimpl->fileId);
        throw std::runtime_error("UVH5 Reader: visdata dataset not found.");
    }
    pimpl->visDataspaceId = H5Dget_space(pimpl->visDatasetId);

    // Confirm actual on-disk shape (Nspws may be 1 for non-flex-spw files).
    {
        hsize_t actual_dims[4] = {};
        int ndims = H5Sget_simple_extent_ndims(pimpl->visDataspaceId);
        if (ndims == 4) {
            H5Sget_simple_extent_dims(pimpl->visDataspaceId, actual_dims, nullptr);
            pimpl->nBlts  = actual_dims[0];
            pimpl->nSpws  = actual_dims[1];
            pimpl->nFreqs = actual_dims[2];
            pimpl->nPols  = actual_dims[3];
        }
    }

    // Open times dataset: [Nblts] float64.
    pimpl->timesDatasetId = H5Dopen2(pimpl->fileId, UVH5_TIMES_DS, H5P_DEFAULT);
    if (pimpl->timesDatasetId < 0) {
        HOLOSCAN_LOG_WARN("UVH5 Reader: '{}' not found — timestamps will not be set.", UVH5_TIMES_DS);
    }

    // Allocate pinned bounce buffer: chunkSize × Nspws × Nfreqs × Npols × 8 bytes (complex64).
    pimpl->visBounceBytes = pimpl->chunkSize * pimpl->nSpws * pimpl->nFreqs * pimpl->nPols
                            * sizeof(std::complex<float>);
    if (cudaMallocHost(&pimpl->visBounceBuffer, pimpl->visBounceBytes) != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("UVH5 Reader: failed to allocate pinned bounce buffer.");
        H5Sclose(pimpl->visDataspaceId);
        H5Dclose(pimpl->visDatasetId);
        H5Fclose(pimpl->fileId);
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
        {static_cast<matx::index_t>(pimpl->chunkSize),
         static_cast<matx::index_t>(pimpl->nSpws * pimpl->nFreqs),
         static_cast<matx::index_t>(pimpl->nPols)},
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

    if (pimpl->visBounceBuffer) {
        cudaFreeHost(pimpl->visBounceBuffer);
        pimpl->visBounceBuffer = nullptr;
    }
    if (pimpl->timesBounceBuffer) {
        cudaFreeHost(reinterpret_cast<void*>(pimpl->timesBounceBuffer));
        pimpl->timesBounceBuffer = nullptr;
    }

    if (pimpl->timesDatasetId >= 0) H5Dclose(pimpl->timesDatasetId);
    if (pimpl->visDataspaceId >= 0) H5Sclose(pimpl->visDataspaceId);
    if (pimpl->visDatasetId  >= 0) H5Dclose(pimpl->visDatasetId);
    if (pimpl->fileId        >= 0) H5Fclose(pimpl->fileId);
}

void Uvh5ReaderOp::compute(InputContext&, OutputContext& output, ExecutionContext&) {
    // Wrap around when fewer blt entries remain than a full chunk.
    if (pimpl->readOffset + pimpl->chunkSize > pimpl->nBlts) {
        pimpl->readOffset = 0;
        HOLOSCAN_LOG_INFO("UVH5 Reader: looping back to start of '{}'.", pimpl->filePath);
    }

    // Select hyperslab in visdata: [readOffset, 0, 0, 0] × [chunkSize, Nspws, Nfreqs, Npols].
    {
        const hsize_t offset[4] = {pimpl->readOffset, 0, 0, 0};
        const hsize_t count[4]  = {pimpl->chunkSize, pimpl->nSpws, pimpl->nFreqs, pimpl->nPols};

        herr_t status = H5Sselect_hyperslab(pimpl->visDataspaceId,
                                             H5S_SELECT_SET,
                                             offset, nullptr, count, nullptr);
        if (status < 0) {
            HOLOSCAN_LOG_ERROR("UVH5 Reader: H5Sselect_hyperslab failed.");
            throw std::runtime_error("UVH5 Reader: hyperslab selection failed.");
        }

        // HDF5 complex64 compound type matching std::complex<float> layout.
        hid_t complex_type = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<float>));
        H5Tinsert(complex_type, "r", 0,                  H5T_NATIVE_FLOAT);
        H5Tinsert(complex_type, "i", sizeof(float),      H5T_NATIVE_FLOAT);

        hid_t memspace = H5Screate_simple(4, count, nullptr);
        status = H5Dread(pimpl->visDatasetId, complex_type,
                         memspace, pimpl->visDataspaceId,
                         H5P_DEFAULT, pimpl->visBounceBuffer);
        H5Sclose(memspace);
        H5Tclose(complex_type);

        if (status < 0) {
            HOLOSCAN_LOG_ERROR("UVH5 Reader: H5Dread (visdata) failed at blt offset {}.", pimpl->readOffset);
            throw std::runtime_error("UVH5 Reader: H5Dread failed.");
        }
    }

    // Read corresponding timestamps if available.
    double firstTimestamp = 0.0;
    if (pimpl->timesDatasetId >= 0 && pimpl->timesBounceBuffer) {
        hid_t ts_space = H5Dget_space(pimpl->timesDatasetId);
        const hsize_t t_offset[1] = {pimpl->readOffset};
        const hsize_t t_count[1]  = {pimpl->chunkSize};
        H5Sselect_hyperslab(ts_space, H5S_SELECT_SET, t_offset, nullptr, t_count, nullptr);
        hid_t t_mem = H5Screate_simple(1, t_count, nullptr);
        H5Dread(pimpl->timesDatasetId, H5T_NATIVE_DOUBLE, t_mem, ts_space, H5P_DEFAULT, pimpl->timesBounceBuffer);
        H5Sclose(t_mem);
        H5Sclose(ts_space);
        firstTimestamp = pimpl->timesBounceBuffer[0];
    }

    // Copy CPU → GPU.
    cudaMemcpyAsync(pimpl->outputTensor->data(),
                    pimpl->visBounceBuffer,
                    pimpl->visBounceBytes,
                    cudaMemcpyHostToDevice,
                    pimpl->stream);
    cudaStreamSynchronize(pimpl->stream);

    pimpl->readOffset += pimpl->chunkSize;
    pimpl->totalBytesRead += static_cast<int64_t>(pimpl->visBounceBytes);
    pimpl->bytesSinceLastMeasurement += static_cast<int64_t>(pimpl->visBounceBytes);
    pimpl->chunkCounter++;

    // Propagate the JD timestamp as metadata so downstream operators can use it.
    const auto& meta = metadata();
    meta->set("timestamp_jd", firstTimestamp);

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
