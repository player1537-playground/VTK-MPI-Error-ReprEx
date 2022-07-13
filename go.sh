#!/usr/bin/env bash

die() { printf $'Error: %s\n' "$*" >&2; exit 1; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
project=${root##*/}
user=${USER:-nouser}
hostname=${HOSTNAME:-nohostname}


#---

mpich_version=3.4.2


#---

docker_source=${root:?}/tools/docker
docker_tag=(
    ${project,,}:latest
)
docker_name=${project:?}
docker_root=${root:?}

go-docker() {
    "${FUNCNAME[0]:?}-$@"
}

go-docker-build() (
    exec docker build \
        "${docker_tag[@]/#/--tag=}" \
        "${docker_source:?}"
)

go-docker-start() (
    exec docker run \
        --rm \
        --detach \
        --init \
        --privileged \
        --name "${docker_name:?}" \
        --mount "type=bind,src=/etc/passwd,dst=/etc/passwd,ro" \
        --mount "type=bind,src=/etc/group,dst=/etc/group,ro" \
        --mount "type=bind,src=${HOME:?},dst=${HOME:?},ro" \
        --mount "type=bind,src=${docker_root:?},dst=${docker_root:?}" \
        "${docker_tag:?}" \
        sleep infinity
)

go-docker-stop() (
    exec docker stop \
        --time 0 \
        "${docker_name:?}"
)

go-docker-exec() (
    exec docker exec \
        --interactive \
        --detach-keys="ctrl-q,ctrl-q" \
        --tty \
        --user "$(id -u):$(id -g)" \
        --workdir "${PWD:?}" \
        --env USER \
        --env HOSTNAME \
        "${docker_name:?}" \
        "$@"
)

go-docker-go() {
    go-docker-exec "${root:?}/go.sh" \
        "$@"
}

go-docker-run() {
    go-docker-go docker--run \
        "$@"
}

go-docker--run() {
    "$@"
}


#---

spack_source=${root:?}/tools/spack
spack_cache=${root:?}/tmp/spack
spack_env=${root:?}/senv
spack_specs=(
    mpich@4.0.2
)

spack_git_source=${spack_source:?}
spack_git_repo=https://github.com/spack/spack.git
spack_git_ref=

go--spack() (
    SPACK_DISABLE_LOCAL_CONFIG=1 \
    SPACK_USER_CACHE_PATH=${spack_cache:?} \
    exec "${spack_source:?}/bin/spack" \
        "$@"
)

go-spack() {
    go docker run "${FUNCNAME[0]:?}-$@"
}

go-spack-git() {
    "${FUNCNAME[0]:?}-$@"
}

go-spack-git-clone() (
    exec git \
        clone \
        "${spack_git_repo:?}" \
        "${spack_git_source:?}" \
        ${spack_git_ref:+--branch "${spack_git_ref:?}"}
)

go-spack-git-checkout() (
    exec git \
        -C "${spack_git_source:?}" \
        checkout \
        "${spack_git_ref:?}"
)

go-spack-env() {
    "${FUNCNAME[0]:?}-$@"
}

go-spack-env-create() {
    go--spack \
        env create \
        -d "${spack_env##*/}" \
        "${spack_env%%/*}"
}

go-spack-install() {
    mkdir -p "${spack_cache:?}" \
    || die "Failed to create spack cache directory: ${spack_cache:?}"

    go--spack \
        --env "${spack_env:?}" \
        install \
        --verbose \
        "${spack_specs[@]}"
}

go-spack-spec() {
    go--spack \
        spec \
        "${spack_specs[@]}"
}

go-spack-run() {
    local tmp
    tmp=$(shopt -po xtrace)
    set +x
    eval $(go--spack --env "${spack_env:?}" load --sh mpich)
    set +vx; eval "${tmp:?}"

    "$@"
}

go-spack-exec() {
    go-spack-run exec "$@"
}

go-spack-go() {
    go-spack-run go "$@"
}


#---

ospray_source=${root:?}/external/ospray

ospray_git_source=${ospray_source:?}
ospray_git_repo=https://github.com/ospray/ospray.git
ospray_git_ref=v2.9.0

ospray_cmake_source=${ospray_source:?}/scripts/superbuild
ospray_cmake_build=${root:?}/build/ospray
ospray_cmake_stage=${root:?}/stage/ospray
ospray_cmake_config=(
    -DCMAKE_POLICY_DEFAULT_CMP0074:STRING=NEW
    -DINSTALL_IN_SEPARATE_DIRECTORIES:BOOL=OFF
)

ospray_run_bindir=${ospray_cmake_stage:?}/bin
ospray_run_libdir=${ospray_cmake_stage:?}/lib:${ospray_cmake_stage:?}/lib64
ospray_run_incdir=${ospray_cmake_stage:?}/include


go-ospray() {
    go spack run "${FUNCNAME[0]:?}-$@"
}

go-ospray-clean() (
    exec rm -rf \
        "${ospray_cmake_build:?}" \
        "${ospray_cmake_stage:?}"
)

go-ospray-git() {
    "${FUNCNAME[0]:?}-$@"
}

go-ospray-git-clone() (
    exec git clone \
        "${ospray_git_repo:?}" \
        "${ospray_git_source:?}" \
        ${ospray_git_ref:+--branch "${ospray_git_ref:?}"}
)

go-ospray-git-checkout() (
    exec git \
        -C "${ospray_git_source:?}" \
        checkout \
        ${ospray_git_ref:+"${ospray_git_ref:?}"}
)

go-ospray-cmake() {
    "${FUNCNAME[0]:?}-$@"
}

go-ospray-cmake-configure() (
    exec cmake \
        -H"${ospray_cmake_source:?}" \
        -B"${ospray_cmake_build:?}" \
        -DCMAKE_INSTALL_PREFIX:PATH="${ospray_cmake_stage:?}" \
        "${ospray_cmake_config[@]}"
)

# env: re, par
go-ospray-cmake--build() (
    exec cmake \
        --build "${ospray_cmake_build:?}" \
        --verbose \
        ${re+--clean-first} \
        ${par+--parallel}
)

go-ospray-cmake-build() {
    local re par
    go-ospray-cmake--build
}

go-ospray-cmake-parbuild() {
    local re par=
    go-ospray-cmake--build
}

go-ospray-cmake-rebuild() {
    local re= par
    go-ospray-cmake--build
}

go-ospray-cmake-reparbuild() {
    local re= par=
    go-ospray-cmake--build
}

go-ospray-cmake-install() (
    exec cmake \
        --install "${ospray_cmake_build:?}" \
        --verbose
)

go-ospray-run() {
    PATH=${ospray_run_bindir:?}${PATH:+:${PATH:?}} \
    CPATH=${ospray_run_incdir:?}${CPATH:+:${CPATH:?}} \
    LD_LIBRARY_PATH=${ospray_run_libdir:?}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH:?}} \
    "$@"
}

go-ospray-go() {
    go-ospray-run go-"$@"
}

go-ospray-exec() {
    go-ospray-run exec "$@"
}

go-ospray-vtk() {
    go-ospray-run go-vtk "$@"
}


#---

vtk_source=${root:?}/external/vtk

vtk_git_source=${vtk_source:?}
vtk_git_repo=https://github.com/Kitware/VTK.git
vtk_git_ref=v9.1.0

vtk_cmake_source=${vtk_source:?}
vtk_cmake_build=${root:?}/build/vtk
vtk_cmake_stage=${root:?}/stage/vtk
vtk_cmake_config=(
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DVTK_ENABLE_LOGGING:BOOL=OFF
    -DVTK_ENABLE_WRAPPING:BOOL=OFF
    -DVTK_LEGACY_REMOVE:BOOL=ON
    -DVTK_OPENGL_USE_GLES:BOOL=OFF
    -DVTK_USE_SDL2:BOOL=OFF
    -DVTK_USE_RENDERING:BOOL=FALSE
    -DVTK_USEINFOVIS:BOOL=FALSE
    -DVTK_NO_PLATFORM_SOCKETS:BOOL=ON
    -DVTK_BUILD_TESTING=OFF
    -DVTK_BUILD_DOCUMENTATION=OFF
    -DVTK_REPORT_OPENGL_ERRORS=OFF
    -DVTK_ALL_NEW_OBJECT_FACTORY=ON
    -DVTK_USE_MPI=ON
    -DVTK_OPENGL_HAS_EGL=ON

    -DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmDataModel:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmFilters:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ChartsCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonArchive:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonColor:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonComputationalGeometry:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonDataModel:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonExecutionModel:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonMath:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonMisc:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonPython:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonSystem:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_CommonTransforms:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_DomainsChemistry:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_DomainsChemistryOpenGL2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_DomainsMicroscopy:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_DomainsParallelChemistry:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersAMR:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersExtraction:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersFlowPaths:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersGeneral:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersGeneric:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersGeometry:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersHybrid:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersHyperTree:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersImaging:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersModeling:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersOpenTURNS:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersParallel:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_FiltersParallelDIY2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersParallelFlowPaths:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersParallelGeometry:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_FiltersParallelImaging:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersParallelMPI:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_FiltersParallelStatistics:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersParallelVerdict:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersPoints:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersProgrammable:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersPython:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersReebGraph:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersSMP:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersSelection:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersSources:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersStatistics:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersTexture:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersTopology:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_FiltersVerdict:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_GeovisCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_GeovisGDAL:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_GUISupportMFC:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_GUISupportQt:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_GUISupportQtQuick:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_GUISupportQtSQL:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingColor:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingFourier:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingGeneral:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingHybrid:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingMath:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingMorphological:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingOpenGL2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingSources:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingStatistics:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ImagingStencil:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_InfovisBoost:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_InfovisBoostGraphAlgorithms:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_InfovisCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_InfovisLayout:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_InteractionImage:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_InteractionStyle:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_InteractionWidgets:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOADIOS2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOAMR:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOAsynchronous:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOCGNSReader:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOCONVERGECFD:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOChemistry:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOCityGML:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOEnSight:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOExodus:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOExport:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOExportGL2PS:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOExportPDF:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOFFMPEG:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOFides:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOGDAL:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOGeoJSON:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOGeometry:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOH5Rage:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOH5part:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOHDF:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOIOSS:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOImage:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOImport:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOInfovis:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOLAS:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOLSDyna:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOLegacy:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOMINC:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOMPIImage:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOMPIParallel:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOMotionFX:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOMovie:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOMySQL:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IONetCDF:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOODBC:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOOMF:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOOggTheora:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOOpenVDB:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOPDAL:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOPIO:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOPLY:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOParallel:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOParallelExodus:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOParallelLSDyna:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOParallelNetCDF:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOParallelXML:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOParallelXdmf3:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOPostgreSQL:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOSQL:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOSegY:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOTRUCHAS:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOTecplotTable:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOVPIC:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOVeraOut:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOVideo:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOXML:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOXMLParser:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOXdmf2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_IOXdmf3:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ParallelCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ParallelDIY:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ParallelMPI:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_ParallelMPI4Py:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingAnnotation:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingContext2D:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingContextOpenGL2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingExternal:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingFFMPEGOpenGL2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingFreeType:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingFreeTypeFontConfig:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingGL2PSOpenGL2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingImage:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingLICOpenGL2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingLOD:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingLabel:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingMatplotlib:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_RenderingOpenVR:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingParallel:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingParallelLIC:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_PythonContext2D:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingQt:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingRayTracing:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_RenderingSceneGraph:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_RenderingTk:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingUI:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_RenderingVR:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingVolume:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_RenderingVolumeAMR:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingVolumeOpenGL2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_RenderingVtkJS:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_TestingCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_TestingGenericBridge:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_TestingIOSQL:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_TestingRendering:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_cgns:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_cli11:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_diy2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_doubleconversion:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_eigen:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_exodusII:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_expat:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_exprtk:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_fides:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_fmt:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_freetype:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_gl2ps:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_glew:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_h5part:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_hdf5:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ioss:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_jpeg:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_jsoncpp:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_kissfft:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_libharu:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_libproj:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_libxml2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_loguru:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_lz4:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_lzma:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_mpi4py:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_netcdf:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ogg:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_pegtl:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_png:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_pugixml:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_sqlite:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_theora:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_tiff:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_utf8:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_verdict:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_vpic:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_vtkm:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_xdmf2:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_xdmf3:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_zfp:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_zlib:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_UtilitiesBenchmarks:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_DICOMParser:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_Java:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_kwiml:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_vtksys:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_mpi:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_metaio:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_opengl:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_Python:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_PythonInterpreter:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_octree:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ViewsContext2D:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ViewsCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ViewsInfovis:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_ViewsQt:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_WebCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_WebPython:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_WebGLExporter:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_WrappingPythonCore:STRING=DONT_WANT
    -DVTK_MODULE_ENABLE_VTK_WrappingTools:STRING=DONT_WANT

    -DVTK_DEFAULT_RENDER_WINDOW_HEADLESS:BOOL=ON
)

vtk_run_bindir=${vtk_cmake_stage:?}/bin
vtk_run_libdir=${vtk_cmake_stage:?}/lib
vtk_run_incdir=${vtk_cmake_stage:?}/include


go-vtk() {
    go ospray run "${FUNCNAME[0]:?}-$@"
}

go-vtk-clean() (
    exec rm -rf \
        "${vtk_cmake_build:?}" \
        "${vtk_cmake_stage:?}"
)

go-vtk-git() {
    "${FUNCNAME[0]:?}-$@"
}

go-vtk-git-clone() (
    exec git clone \
        "${vtk_git_repo:?}" \
        "${vtk_git_source:?}" \
        ${vtk_git_ref:+--branch "${vtk_git_ref:?}"}
)

go-vtk-git-checkout() (
    exec git \
        -C "${vtk_git_source:?}" \
        checkout \
        ${vtk_git_ref:+"${vtk_git_ref:?}"}
)

go-vtk-cmake() {
    "${FUNCNAME[0]:?}-$@"
}

go-vtk-cmake-configure() (
    exec cmake \
        -H"${vtk_cmake_source:?}" \
        -B"${vtk_cmake_build:?}" \
        -DCMAKE_INSTALL_PREFIX:PATH="${vtk_cmake_stage:?}" \
        "${vtk_cmake_config[@]}"
)

# env: re, par
go-vtk-cmake--build() (
    exec cmake \
        --build "${vtk_cmake_build:?}" \
        --verbose \
        ${re+--clean-first} \
        ${par+--parallel}
)

go-vtk-cmake-build() {
    local re par
    go-vtk-cmake--build
}

go-vtk-cmake-parbuild() {
    local re par=
    go-vtk-cmake--build
}

go-vtk-cmake-rebuild() {
    local re= par
    go-vtk-cmake--build
}

go-vtk-cmake-reparbuild() {
    local re= par=
    go-vtk-cmake--build
}

go-vtk-cmake-install() (
    exec cmake \
        --install "${vtk_cmake_build:?}" \
        --verbose
)

go-vtk-run() {
    PATH=${vtk_run_bindir:?}${PATH:+:${PATH:?}} \
    CPATH=${vtk_run_incdir:?}${CPATH:+:${CPATH:?}} \
    LD_LIBRARY_PATH=${vtk_run_libdir:?}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH:?}} \
    vtk_ROOT=${vtk_cmake_stage:?} \
    "$@"
}

go-vtk-go() {
    go-vtk-run go-"$@"
}

go-vtk-exec() {
    go-vtk-run exec "$@"
}

go-vtk-src() {
    go-vtk-run go-src "$@"
}


#---

src_source=${root:?}/src

src_cmake_source=${src_source:?}
src_cmake_build=${root:?}/build/src
src_cmake_stage=${root:?}/stage/src
src_cmake_config=(
)

src_run_bindir=${src_cmake_stage:?}/bin
src_run_libdir=${src_cmake_stage:?}/lib
src_run_incdir=${src_cmake_stage:?}/include


go-src() {
    go vtk run "${FUNCNAME[0]:?}-$@"
}

go-src-clean() (
    exec rm -rf \
        "${src_cmake_build:?}" \
        "${src_cmake_stage:?}"
)

go-src-cmake() {
    "${FUNCNAME[0]:?}-$@"
}

go-src-cmake-configure() (
    exec cmake \
        -H"${src_cmake_source:?}" \
        -B"${src_cmake_build:?}" \
        -DCMAKE_INSTALL_PREFIX:PATH="${src_cmake_stage:?}" \
        "${src_cmake_config[@]}"
)

# env: re, par
go-src-cmake--build() (
    exec cmake \
        --build "${src_cmake_build:?}" \
        --verbose \
        ${re+--clean-first} \
        ${par+--parallel}
)

go-src-cmake-build() {
    local re par
    go-src-cmake--build
}

go-src-cmake-parbuild() {
    local re par=
    go-src-cmake--build
}

go-src-cmake-rebuild() {
    local re= par
    go-src-cmake--build
}

go-src-cmake-reparbuild() {
    local re= par=
    go-src-cmake--build
}

go-src-cmake-install() (
    exec cmake \
        --install "${src_cmake_build:?}" \
        --verbose
)

go-src-run() {
    PATH=${src_run_bindir:?}${PATH:+:${PATH:?}} \
    CPATH=${src_run_incdir:?}${CPATH:+:${CPATH:?}} \
    LD_LIBRARY_PATH=${src_run_libdir:?}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH:?}} \
    "$@"
}

go-src-go() {
    go-src-run go-"$@"
}

go-src-exec() {
    go-src-run exec "$@"
}


#---

go--demo-exec() {
    1>"${root:?}/tmp/output.txt" \
    2>&1 \
    go "$@"
}

go-demo() {
    go() {
        local _go_tmp
        printf -v _go_tmp $' %q' go "$@"
        printf -v _go_tmp $'$%s' "${_go_tmp:?}"

        1>&2 printf $'%s' "${_go_tmp:?}"

        if
            /usr/bin/time \
            -f $'\r[real %E]'" ${_go_tmp:?}" \
            -- \
                "${root:?}/go.sh" \
                -demo-exec \
                    "$@"
        then
            return 0
        else
            1>&2 printf $'Command failed. Output:\n'
            1>&2 printf $'====\n'
            1>&2 cat "${root:?}/tmp/output.txt"
            1>&2 printf $'====\n'
            return 1
        fi
    }

    go src cmake configure \
    || die "Configure failed"

    go src cmake build \
    || die "Build failed"

    go src cmake install \
    || die "Install failed"

    go() {
        "${FUNCNAME[0]:?}-$@"
    }

    go src exec mpirun -np 1 gdb -ex=r --args vtkPDistributedDataFilterExample
}


#---

go() {
    "${FUNCNAME[0]:?}-$@"
}

test -f "${root:?}/env.sh" && source "${_:?}"
test -f "${root:?}/${hostname:+@${hostname:?}.env.sh}" && source "${_:?}"
test -f "${root:?}/${user:+${user:?}@.env.sh}" && source "${_:?}"
test -f "${root:?}/${user:+${hostname:+${user:?}@${hostname:?}.env.sh}}" && source "${_:?}"

go "$@"
