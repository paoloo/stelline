deps = [
    dependency('hdf5_vfd_gds', required: false),
    dependency('h5dsc99', required: false),
]

format_deps = [
    dependency('filterbankc99', required: false),
    dependency('uvh5c99', required: false),
]

basic_deps_found = true
foreach x_dep : deps
    basic_deps_found = basic_deps_found and x_dep.found()
endforeach

fbh5_dep = dependency('filterbankc99', required: false)
fbh5_dep_found = fbh5_dep.found()
uvh5_dep = dependency('uvh5c99', version: '>= 0.9.0', required: false)
uvh5_dep_found = uvh5_dep.found()

if basic_deps_found and (fbh5_dep_found or uvh5_dep_found)
    cfg_lst.set('STELLINE_LOADER_HDF5', true)
    dep_lst += deps

    if fbh5_dep_found
        cfg_lst.set('STELLINE_LOADER_FBH5', true)
        dep_lst += [fbh5_dep]
    endif
    if uvh5_dep_found
        cfg_lst.set('STELLINE_LOADER_UVH5', true)
        dep_lst += [uvh5_dep]
    endif
endif

# Reader path: plain HDF5 only — no GDS VFD required.
# Enables FBH5/UVH5 readers on any CUDA system (including non-NVMe/non-GPUDirect).
#
# Use h5dsc99 as the HDF5 availability proxy instead of dependency('hdf5').
# Meson's hdf5.py looks up the C compiler (coredata.compilers['c']) which
# raises KeyError on projects that only declare C++ and CUDA languages.
# h5dsc99 is already detected above, requires HDF5, and avoids that code path.
h5dsc99_probe = dependency('h5dsc99', required: false)  # cached — no re-probe
fbh5_read_found = h5dsc99_probe.found() and fbh5_dep_found
uvh5_read_found = h5dsc99_probe.found() and uvh5_dep_found

if (fbh5_read_found or uvh5_read_found) and not basic_deps_found
    # GDS path not available: add h5dsc99 (brings in HDF5) and format deps.
    dep_lst += [h5dsc99_probe]
    if fbh5_read_found
        dep_lst += [fbh5_dep]
    endif
    if uvh5_read_found
        dep_lst += [uvh5_dep]
    endif
endif

if fbh5_read_found
    cfg_lst.set('STELLINE_LOADER_FBH5_READ', true)
endif
if uvh5_read_found
    cfg_lst.set('STELLINE_LOADER_UVH5_READ', true)
endif

ldr_lst += {'FBH5': basic_deps_found and fbh5_dep_found}
ldr_lst += {'UVH5': basic_deps_found and uvh5_dep_found}
ldr_lst += {'FBH5 Reader': fbh5_read_found}
ldr_lst += {'UVH5 Reader': uvh5_read_found}
