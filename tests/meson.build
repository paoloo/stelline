filterbankc99_dep = dependency('filterbankc99', required: false)

if filterbankc99_dep.found()
    fbh5_compat_test = executable(
        'fbh5_compat_test',
        'fbh5_compat_test.cc',
        include_directories: inc_lst,
        dependencies: dep_lst + [filterbankc99_dep],
    )

    test(
        'fbh5_compat_test',
        fbh5_compat_test,
        args: [join_paths(meson.current_build_dir(), 'fbutils_h5.00.fbh5')],
    )
endif
