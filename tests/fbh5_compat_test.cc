#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <hdf5.h>

extern "C" {
#include "filterbankc99.h"
}

static constexpr size_t kTimesPerChunk = 3;
static constexpr size_t kChansPerChunk = 4;
static constexpr size_t kNumChunks = 4;

static void fill_chunk(float* data, size_t chunk_index) {
    for (size_t t = 0; t < kTimesPerChunk; ++t) {
        for (size_t c = 0; c < kChansPerChunk; ++c) {
            const float sample = (chunk_index == 2)
                ? static_cast<float>(t * 1000.0 + (kChansPerChunk - c))
                : static_cast<float>(t * 1000.0 + c + 1.0);
            data[t * kChansPerChunk + c] = sample;
        }
    }
}

static int verify_generated_file(const char* path) {
    filterbank_h5_file_t fbh5 = filterbank_h5_access_file_explicit(
        const_cast<char*>(path),
        H5P_DEFAULT
    );
    if (fbh5.file_id == H5I_INVALID_HID) {
        std::fprintf(stderr, "failed to open generated file for validation\n");
        return 1;
    }

    char* attr_class = H5DSread_all(fbh5.file_id, "CLASS");
    char* attr_version = H5DSread_all(fbh5.file_id, "VERSION");
    const bool class_ok = attr_class && std::strcmp(attr_class, "FILTERBANK") == 0;
    const bool version_ok = attr_version && std::strcmp(attr_version, "1.0") == 0;
    if (!class_ok || !version_ok) {
        std::fprintf(stderr, "unexpected root attrs: CLASS='%s' VERSION='%s'\n",
                     attr_class ? attr_class : "(null)",
                     attr_version ? attr_version : "(null)");
        std::free(attr_class);
        std::free(attr_version);
        filterbank_h5_close(&fbh5);
        return 1;
    }
    std::free(attr_class);
    std::free(attr_version);

    if (fbh5.ds_data.rank != 3 ||
        fbh5.ds_data.dims[0] != kTimesPerChunk * kNumChunks ||
        fbh5.ds_data.dims[1] != 1 ||
        fbh5.ds_data.dims[2] != kChansPerChunk) {
        std::fprintf(stderr, "unexpected dataset shape: [%ld, %ld, %ld]\n",
                     static_cast<long>(fbh5.ds_data.dims[0]),
                     static_cast<long>(fbh5.ds_data.dims[1]),
                     static_cast<long>(fbh5.ds_data.dims[2]));
        filterbank_h5_close(&fbh5);
        return 1;
    }

    filterbank_h5_change_access_chunking(&fbh5, kTimesPerChunk, 0, 0);
    filterbank_h5_alloc(&fbh5);

    int failed = 0;
    const int chunks = 1;
    int read_count = 0;
    herr_t status = 0;
    while ((status = filterbank_h5_read(&fbh5)) >= 0 && read_count <= 10) {
        size_t s = 0;
        for (size_t t = 0; !failed && t < kTimesPerChunk; ++t) {
            for (size_t i = 0; !failed && i < 1; ++i) {
                for (size_t c = 0; !failed && c < kChansPerChunk; ++c) {
                    const size_t chan = ((read_count % chunks) * kChansPerChunk) + c;
                    float sample_exp = static_cast<float>(t * 1000.0 + chan + 1.0);
                    if (read_count * chunks == 2) {
                        sample_exp = static_cast<float>(t * 1000.0 + (fbh5.ds_data.dims[2] - c));
                    }
                    const float sample = static_cast<float*>(fbh5.data)[s];
                    if (sample != sample_exp) {
                        std::fprintf(stderr,
                                     "mismatch at chunk=%d t=%zu i=%zu c=%zu: %f != %f\n",
                                     read_count, t, i, c, sample, sample_exp);
                        failed = 1;
                    }
                    ++s;
                }
            }
        }

        ++read_count;
        if (status == 1) {
            break;
        }
    }

    filterbank_h5_free(&fbh5);
    filterbank_h5_close(&fbh5);

    if (status != 1) {
        std::fprintf(stderr, "final read status not positive: %d\n", status);
        failed = 1;
    }
    if (read_count != static_cast<int>(kNumChunks)) {
        std::fprintf(stderr, "expected %zu chunks, got %d\n", kNumChunks, read_count);
        failed = 1;
    }

    return failed;
}

int main(int argc, char* argv[]) {
    const std::string out_path = argc > 1 ? argv[1] : "./fbutils_h5.00.fbh5";

    filterbank_h5_file_t fbh5 = {0};
    fbh5.header = {0};
    filterbank_header_t* hdr = &fbh5.header;

    hdr->machine_id = 20;
    hdr->telescope_id = 6;
    hdr->data_type = 1;
    hdr->barycentric = 1;
    hdr->pulsarcentric = 1;
    hdr->src_raj = 20.0 + 39.0 / 60.0 + 7.4 / 3600.0;
    hdr->src_dej = 42.0 + 24.0 / 60.0 + 24.5 / 3600.0;
    hdr->az_start = 12.3456;
    hdr->za_start = 65.4321;
    hdr->fch1 = 4626.464842353016138;
    hdr->foff = -0.000002793967724;
    hdr->nchans = static_cast<int>(kChansPerChunk);
    hdr->nbeams = 1;
    hdr->ibeam = 1;
    hdr->nbits = 32;
    hdr->tstart = 57856.810798611114;
    hdr->tsamp = 1.825361100800;
    hdr->nifs = 1;
    std::strncpy(hdr->source_name,
                 "1234567891123456789212345678931234567894123456789512345678961234567897123456789",
                 sizeof(hdr->source_name) - 1);
    std::strncpy(hdr->rawdatafile, "", sizeof(hdr->rawdatafile) - 1);

    fbh5.nchans_per_write = kChansPerChunk;
    fbh5.ntimes_per_write = kTimesPerChunk;

    fbh5.file_id = H5I_INVALID_HID;
    if (filterbank_h5_open_explicit(out_path.c_str(), &fbh5, H5Tcopy(H5T_IEEE_F32LE), H5P_DEFAULT) != 0) {
        std::fprintf(stderr, "failed to create FBH5 fixture: %s\n", out_path.c_str());
        return 1;
    }

    filterbank_h5_alloc(&fbh5);
    if (!fbh5.data) {
        std::fprintf(stderr, "failed to allocate data buffer\n");
        filterbank_h5_free(&fbh5);
        filterbank_h5_close(&fbh5);
        return 1;
    }
    std::memset(fbh5.mask, 0, H5DSsize(&fbh5.ds_mask));

    for (size_t chunk = 0; chunk < kNumChunks; ++chunk) {
        fill_chunk(static_cast<float*>(fbh5.data), chunk);
        if (filterbank_h5_write(&fbh5) != 0) {
            std::fprintf(stderr, "failed to write chunk %zu\n", chunk);
            filterbank_h5_free(&fbh5);
            filterbank_h5_close(&fbh5);
            return 1;
        }
    }

    filterbank_h5_free(&fbh5);
    filterbank_h5_close(&fbh5);

    return verify_generated_file(out_path.c_str());
}
