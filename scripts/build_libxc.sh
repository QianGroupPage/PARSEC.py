#!/bin/bash
# Build libxc (MPL-2.0, https://libxc.gitlab.io) as the optional exchange-correlation library of
# PARSEC-Python: a shared library that parsec_python.V_xc.libxc loads with ctypes. The source is
# downloaded from libxc's GitLab (nothing of libxc is kept in this repository).
#
#   bash scripts/build_libxc.sh [--version 7.1.2] [--prefix DIR] [--cuda] [--cuda-arch 80]
#                               [--maxorder 1] [--jobs N] [--source TARBALL] [--sha256 HASH]
#                               [--work DIR] [--probe]
#
#   --cuda        the CUDA build: every functional is compiled for the device as well; one library
#                 then evaluates host arrays (functionals initialized with XC_FLAGS_ON_HOST) and
#                 CuPy device arrays (XC_FLAGS_ON_DEVICE), which needs libxc 7.1 or later
#                 (xc_func_init_flags). Without --cuda: the host build.
#   --cuda-arch   CMAKE_CUDA_ARCHITECTURES (default 80: the A100 of Perlmutter; 90 for H100).
#   --fmad        on|off: nvcc's fused multiply-adds (default off, as the device kernels of this code:
#                 the device then rounds as the host build, and host and device agree to round-off).
#   --maxorder    highest derivative compiled (libxc's MAXORDER; default 1 = energy and potential,
#                 all an SCF needs; libxc's own default 2 adds the kernels of linear response and
#                 doubles the compile time of the CUDA build).
#   --prefix      install folder (default: $PWD/libxc-<version>-<cpu|cuda-smXX>); the library is
#                 <prefix>/lib/libxc.so (lib64 on some systems).
#   --source      a libxc-<version>.tar.gz already on disk (no download).
#   --sha256      the expected sha256 of the source archive; known for 7.1.2 (GitLab's archive),
#                 required for another version (or `--sha256 any` to skip the check).
#   --probe       print the tools this script would use and stop.
#
# After the build: export PARSEC_LIBXC_LIBRARY=<prefix>/lib/libxc.so (or put <prefix>/lib on
# LD_LIBRARY_PATH); `python -m parsec_python.V_xc.libxc` prints what the code finds.
#
# Perlmutter (NERSC): run on a compute node (the CUDA build compiles ~600 functionals for the device;
# 77 s with 16 jobs at MAXORDER 1 on an A100 node, the host build 25 s) after
# `module load cudatoolkit cmake` (PrgEnv-gnu), installed into a project directory such as
# <prefix>/libxc/<version>-cuda-sm80 and <prefix>/libxc/<version>-cpu.
# Alternatives: conda-forge's `libxc-c` (cpu and cuda variants; 7.1.2 for linux-64 and win-64).
set -euo pipefail

version=7.1.2
prefix=
cuda=0
arch=80
maxorder=1
fmad=off
jobs=$(nproc 2> /dev/null || echo 8)
source_tarball=
expected_sha256=
work=
probe=0
# GitLab's archive of a release (https://gitlab.com/libxc/libxc/-/archive/<v>/libxc-<v>.tar.gz).
declare -A known_sha256=([7.1.2]=c517ce61820ea8114664a4280b6a6bc74a4f22f1fd1ea4ddecd6df0caeeae4f4)
while (( $# )); do
  case $1 in
    --version) version=$2; shift 2;;
    --prefix) prefix=$2; shift 2;;
    --cuda) cuda=1; shift;;
    --cuda-arch) arch=$2; shift 2;;
    --maxorder) maxorder=$2; shift 2;;
    --fmad) fmad=$2; shift 2;;
    --jobs) jobs=$2; shift 2;;
    --source) source_tarball=$2; shift 2;;
    --sha256) expected_sha256=$2; shift 2;;
    --work) work=$2; shift 2;;
    --probe) probe=1; shift;;
    -h|--help) sed -n '2,34p' "$0"; exit 0;;
    *) echo "build_libxc: unknown option $1" >&2; exit 2;;
  esac
done
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "build_libxc: version like 7.1.2" >&2; exit 2; }
[[ $maxorder =~ ^[0-4]$ ]] || { echo "build_libxc: --maxorder 0 to 4" >&2; exit 2; }
[[ $fmad =~ ^(on|off)$ ]] || { echo "build_libxc: --fmad on or off" >&2; exit 2; }
label=cpu; (( cuda )) && label=cuda-sm$arch
(( cuda )) && [[ $fmad == on ]] && label=$label-fmad
prefix=${prefix:-$PWD/libxc-$version-$label}

# Perlmutter: the modules of the CUDA toolchain when the shell has them but they are not loaded.
if (( cuda )) && ! command -v nvcc > /dev/null 2>&1 && command -v module > /dev/null 2>&1; then
  module load cudatoolkit > /dev/null 2>&1 || true
fi
if ! command -v cmake > /dev/null 2>&1 && command -v module > /dev/null 2>&1; then
  module load cmake > /dev/null 2>&1 || true
fi

echo "build_libxc: libxc $version, $label, MAXORDER $maxorder, $jobs jobs, prefix $prefix"
for tool in cmake cc gcc nvcc curl; do
  printf '  %-6s %s\n' "$tool" "$(command -v "$tool" 2> /dev/null || echo missing)"
done
command -v cmake > /dev/null && cmake --version | head -n 1
(( cuda )) && command -v nvcc > /dev/null && nvcc --version | tail -n 2
(( probe )) && exit 0

command -v cmake > /dev/null || { echo "build_libxc: cmake is required" >&2; exit 3; }
(( cuda )) && { command -v nvcc > /dev/null || { echo "build_libxc: nvcc is required for --cuda" >&2; exit 3; }; }

work=${work:-$(mktemp -d "${TMPDIR:-/tmp}/libxc-build.XXXXXX")}
mkdir -p "$work"
if [[ -z $source_tarball ]]; then
  source_tarball=$work/libxc-$version.tar.gz
  curl -sSfL -o "$source_tarball" "https://gitlab.com/libxc/libxc/-/archive/$version/libxc-$version.tar.gz"
fi
actual_sha256=$(sha256sum "$source_tarball" | cut -d' ' -f1)
echo "build_libxc: source $source_tarball sha256 $actual_sha256"
expected_sha256=${expected_sha256:-${known_sha256[$version]:-}}
if [[ -z $expected_sha256 ]]; then
  echo "build_libxc: no known sha256 for libxc $version: give --sha256 HASH (or --sha256 any)" >&2
  exit 4
fi
if [[ $expected_sha256 != any && $expected_sha256 != "$actual_sha256" ]]; then
  echo "build_libxc: sha256 mismatch: expected $expected_sha256" >&2
  exit 4
fi
tar xzf "$source_tarball" -C "$work"
src=$work/libxc-$version
[[ -f $src/CMakeLists.txt ]] || { echo "build_libxc: no CMakeLists.txt in $src" >&2; exit 4; }

options=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DENABLE_FORTRAN=OFF
         -DENABLE_PYTHON=OFF -DMAXORDER="$maxorder" -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib)
# The plain GNU compilers, not a vendor wrapper: Cray's cc links the CUDA runtime of the loaded
# craype-accel module into every library, so that a host build would not load on a CPU node.
host_cc=$(command -v gcc || true)
host_cxx=$(command -v g++ || true)
[[ -n $host_cc ]] && options+=(-DCMAKE_C_COMPILER="$host_cc")
(( cuda )) && [[ -n $host_cxx ]] && options+=(-DCMAKE_CUDA_HOST_COMPILER="$host_cxx")
if (( cuda )); then
  # The CUDA runtime as a shared library: the one CuPy loads, so that both see the same current
  # device and default stream; its folder is written into the RPATH of libxc.so.
  options+=(-DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
            -DCMAKE_CUDA_FLAGS="--fmad=$( [[ $fmad == on ]] && echo true || echo false )"
            -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON)
fi
start=$SECONDS
cmake -S "$src" -B "$work/build" "${options[@]}"
cmake --build "$work/build" --parallel "$jobs"
cmake --install "$work/build"
echo "build_libxc: built in $((SECONDS - start)) s"
library=$(ls "$prefix"/lib*/libxc.so 2> /dev/null | head -n 1)
[[ -n $library ]] || { echo "build_libxc: no libxc.so under $prefix" >&2; exit 5; }
# A host build must not need a CUDA runtime; a CUDA build names the one it needs. ldd's output is
# kept whole first (no early exit of a reader under pipefail).
linked=$(ldd "$library")
echo "build_libxc: ldd $(basename "$library"):"
grep -E "cuda|cudart|not found" <<< "$linked" || echo "  (no CUDA libraries)"
if (( ! cuda )) && grep -q cudart <<< "$linked"; then
  echo "build_libxc: the host build links a CUDA runtime (a compiler wrapper?)" >&2
  exit 6
fi
{
  echo "libxc $version ($label), MAXORDER $maxorder, built $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
  echo "source sha256 $actual_sha256"
  echo "cmake ${options[*]}"
  grep -E "cuda|cudart" <<< "$linked" || true
} > "$prefix/BUILD_INFO.txt"
echo "build_libxc: export PARSEC_LIBXC_LIBRARY=$library"
