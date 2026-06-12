#include <cassert>
#include <cstring>

#include <hdf5.h>

#include <stelline/types.hh>
#include <stelline/operators/filesystem/base.hh>
#include <stelline/utils/tensor.hh>
#include <fmt/format.h>

#include <matx.h>
#include <cuda/std/complex>

using namespace holoscan;

namespace stelline::operators::filesystem {

// HDF5 dataset path used by filterbankc99 / the standard filterbank HDF5 spec.
static constexpr const char* FBH5_DATA_DATASET = "data";

struct Fbh5ReaderOp::Impl {
    // Configuration.

    std::string filePath;
    hsize_t chunkSize;

    // HDF5 state.

    hid_t fileId;
    hid_t datasetId;
    hid_t dataspaceId;

    // Data dimensions read from the file.
    // Dataset layout: [ntimes, nifs, nchans]  (standard filterbank HDF5)

    hsize_t dims[3];
    int ndims;

    // Current read position (first-dimension index).

    hsize_t readOffset;

    // CPU pinned bounce buffer and CUDA stream for H→D transfer.

    void* bounceBuffer;
    size_t bounceBufferBytes;
    cudaStream_t stream;

    // GPU output tensor (reused across compute() calls).

    std::shared_ptr<holoscan::Tensor> outputTensor;

    // Metrics.

    std::chrono::time_point<std::chrono::steady_clock> lastMeasurementTime;
    std::atomic<int64_t> bytesSinceLastMeasurement{0};
    std::atomic<double> currentBandwidthMBps{0.0};
    int64_t totalBytesRead{0};
    uint64_t chunkCounter{0};
};

void Fbh5ReaderOp::initialize() {
    pimpl = new Impl();
    Operator::initialize();
}

Fbh5ReaderOp::~Fbh5ReaderOp() {
    delete pimpl;
}

void Fbh5ReaderOp::setup(OperatorSpec& spec) {
    spec.output<std::shared_ptr<holoscan::Tensor>>("out")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(filePath_, "file_path");
    spec.param(chunkSize_, "chunk_size");
}

void Fbh5ReaderOp::start() {
    pimpl->filePath  = filePath_.get();
    pimpl->chunkSize = static_cast<hsize_t>(chunkSize_.get());

    // Suppress HDF5 automatic error printing — we handle errors manually.
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

    // Open the file read-only with the default (POSIX) VFD — no GDS required.
    pimpl->fileId = H5Fopen(pimpl->filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (pimpl->fileId < 0) {
        HOLOSCAN_LOG_ERROR("FBH5 Reader: cannot open '{}'.", pimpl->filePath);
        throw std::runtime_error(fmt::format("FBH5 Reader: cannot open '{}'.", pimpl->filePath));
    }

    pimpl->datasetId = H5Dopen2(pimpl->fileId, FBH5_DATA_DATASET, H5P_DEFAULT);
    if (pimpl->datasetId < 0) {
        HOLOSCAN_LOG_ERROR("FBH5 Reader: dataset '{}' not found in '{}'.", FBH5_DATA_DATASET, pimpl->filePath);
        H5Fclose(pimpl->fileId);
        throw std::runtime_error("FBH5 Reader: dataset not found.");
    }

    pimpl->dataspaceId = H5Dget_space(pimpl->datasetId);
    pimpl->ndims = H5Sget_simple_extent_ndims(pimpl->dataspaceId);

    if (pimpl->ndims < 1 || pimpl->ndims > 3) {
        HOLOSCAN_LOG_ERROR("FBH5 Reader: unexpected dataset rank {} (expected 1–3).", pimpl->ndims);
        H5Sclose(pimpl->dataspaceId);
        H5Dclose(pimpl->datasetId);
        H5Fclose(pimpl->fileId);
        throw std::runtime_error("FBH5 Reader: unexpected dataset rank.");
    }

    // Fill dims[]; pad missing trailing dims with 1.
    std::fill(pimpl->dims, pimpl->dims + 3, static_cast<hsize_t>(1));
    H5Sget_simple_extent_dims(pimpl->dataspaceId, pimpl->dims, nullptr);

    HOLOSCAN_LOG_INFO("FBH5 Reader: opened '{}' — dims=[{},{},{}], chunk_size={}.",
                      pimpl->filePath,
                      pimpl->dims[0], pimpl->dims[1], pimpl->dims[2],
                      pimpl->chunkSize);

    // Bounce buffer: one chunk's worth of float32 samples.
    pimpl->bounceBufferBytes = pimpl->chunkSize * pimpl->dims[1] * pimpl->dims[2] * sizeof(float);
    if (cudaMallocHost(&pimpl->bounceBuffer, pimpl->bounceBufferBytes) != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("FBH5 Reader: failed to allocate pinned bounce buffer.");
        H5Sclose(pimpl->dataspaceId);
        H5Dclose(pimpl->datasetId);
        H5Fclose(pimpl->fileId);
        throw std::runtime_error("FBH5 Reader: bounce buffer allocation failed.");
    }

    // GPU output tensor: [chunk_size, nifs, nchans].
    auto t = matx::make_tensor<float>(
        {static_cast<matx::index_t>(pimpl->chunkSize),
         static_cast<matx::index_t>(pimpl->dims[1]),
         static_cast<matx::index_t>(pimpl->dims[2])},
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

void Fbh5ReaderOp::stop() {
    cudaStreamDestroy(pimpl->stream);

    if (pimpl->bounceBuffer) {
        cudaFreeHost(pimpl->bounceBuffer);
        pimpl->bounceBuffer = nullptr;
    }

    if (pimpl->dataspaceId >= 0) {
        H5Sclose(pimpl->dataspaceId);
    }
    if (pimpl->datasetId >= 0) {
        H5Dclose(pimpl->datasetId);
    }
    if (pimpl->fileId >= 0) {
        H5Fclose(pimpl->fileId);
    }
}

void Fbh5ReaderOp::compute(InputContext&, OutputContext& output, ExecutionContext&) {
    const hsize_t totalTimes = pimpl->dims[0];

    // Wrap around when fewer samples remain than a full chunk.
    if (pimpl->readOffset + pimpl->chunkSize > totalTimes) {
        pimpl->readOffset = 0;
        HOLOSCAN_LOG_INFO("FBH5 Reader: looping back to start of '{}'.", pimpl->filePath);
    }

    // Select hyperslab: [readOffset, 0, 0] with count [chunkSize, nifs, nchans].
    const hsize_t offset[3] = {pimpl->readOffset, 0, 0};
    const hsize_t count[3]  = {pimpl->chunkSize, pimpl->dims[1], pimpl->dims[2]};

    herr_t status = H5Sselect_hyperslab(pimpl->dataspaceId,
                                         H5S_SELECT_SET,
                                         offset, nullptr, count, nullptr);
    if (status < 0) {
        HOLOSCAN_LOG_ERROR("FBH5 Reader: H5Sselect_hyperslab failed.");
        throw std::runtime_error("FBH5 Reader: hyperslab selection failed.");
    }

    hid_t memspace = H5Screate_simple(pimpl->ndims, count, nullptr);
    status = H5Dread(pimpl->datasetId, H5T_NATIVE_FLOAT,
                     memspace, pimpl->dataspaceId,
                     H5P_DEFAULT, pimpl->bounceBuffer);
    H5Sclose(memspace);

    if (status < 0) {
        HOLOSCAN_LOG_ERROR("FBH5 Reader: H5Dread failed at offset {}.", pimpl->readOffset);
        throw std::runtime_error("FBH5 Reader: H5Dread failed.");
    }

    // Copy CPU → GPU asynchronously then synchronise before emit.
    cudaMemcpyAsync(pimpl->outputTensor->data(),
                    pimpl->bounceBuffer,
                    pimpl->bounceBufferBytes,
                    cudaMemcpyHostToDevice,
                    pimpl->stream);
    cudaStreamSynchronize(pimpl->stream);

    pimpl->readOffset += pimpl->chunkSize;
    pimpl->totalBytesRead += static_cast<int64_t>(pimpl->bounceBufferBytes);
    pimpl->bytesSinceLastMeasurement += static_cast<int64_t>(pimpl->bounceBufferBytes);
    pimpl->chunkCounter++;

    metadata()->set("timestamp", pimpl->chunkCounter);
    output.emit(pimpl->outputTensor, "out");
}

void Fbh5ReaderOp::tick() {
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

std::string Fbh5ReaderOp::formatMetrics(const MetricsProvider::MetricsMap& metrics) {
    return fmt::format("  Current Bandwidth: {} MB/s\n"
                       "  Total Data Read: {} MB\n"
                       "  Chunks Read: {}",
                       metrics.at("current_bandwidth_mb_s").value,
                       metrics.at("total_data_read_mb").value,
                       metrics.at("chunks_read").value);
}

}  // namespace stelline::operators::filesystem
