if [ "${USER:-}" = "thobson" ]; then
    docker_tag=thobson2/${docker_tag##*/}
    docker_tag=${docker_tag%%:*}:latest

    docker_name=thobson2--${docker_name##*--}
fi

ospray_cmake_config+=(
    -DCMAKE_BUILD_TYPE:STRING=DebWithRelInfo
)
vtk_cmake_config+=(
    -DCMAKE_BUILD_TYPE:STRING=Debug
    -D_vtk_module_log=module
)
src_cmake_config+=(
    -DCMAKE_BUILD_TYPE:STRING=Debug
)

go-vtk-debug() {
    go-vtk-debug-"$@"
}

go-vtk-debug-modules() {
    mkdir -p "${root:?}/tmp" \
    || die "Could not create tmp dir: ${root:?}/tmp"

    rm -rf "${vtk_build:?}" \
    || die "Could not clean vtk build dir: ${vtk_build:?}"

    vtk_config=(
        -D_vtk_module_log=module
    )

    1>"${root:?}/tmp/stdout.txt" \
    2>"${root:?}/tmp/stderr.txt" \
    go-vtk-cmake-configure \
    || die "Could not configure cmake build"

    0<"${root:?}/tmp/stdout.txt" \
    1>"${root:?}/tmp/modules.txt" \
    sed -ne 's/^-- VTK module debug module: VTK::\([a-zA-Z0-9]\{1,\}\) declared by .*$/\1/p'

    0<"${root:?}/tmp/modules.txt" \
    sed -e 's/.*/-DVTK_MODULE_ENABLE_&:STRING=DONT_WANT/'
}
