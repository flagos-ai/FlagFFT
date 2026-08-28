#!/bin/bash
set -e

# FlagFFT (nvidia backend) Debian package build script.
# Usage: ./build-flagfft.sh [--base-image IMAGE] [--output-dir DIR]

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }

# Default values — must match Dockerfile.deb's ARG BASE_IMAGE default.
BASE_IMAGE="nvidia/cuda:12.8.0-devel-ubuntu22.04"   # FlagOS DEB ABI matrix
PYTHON_VERSION="3.12"
TORCH_INDEX="https://resource.flagos.net/repository/flagos-pypi-nvidia/simple"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-image) BASE_IMAGE="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --torch-index) TORCH_INDEX="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--base-image IMAGE] [--python-version X.Y] [--torch-index URL] [--output-dir DIR]"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/debian-packages"
fi
log_step "Check submodule deps/libtriton_jit"
if [ ! -f "${PROJECT_ROOT}/deps/libtriton_jit/CMakeLists.txt" ]; then
    log_error "Submodule deps/libtriton_jit is not initialized"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
IMAGE_TAG="flagfft-deb-builder:$(date +%s)"

log_step "docker build (base: ${BASE_IMAGE})"
docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg TORCH_INDEX="${TORCH_INDEX}" \
    -f "${PROJECT_ROOT}/packaging/debian/build-helpers/Dockerfile.deb" \
    -t "${IMAGE_TAG}" \
    "${PROJECT_ROOT}"

log_step "Extract .deb files into ${OUTPUT_DIR}"
CID="$(docker create "${IMAGE_TAG}" /bin/true)"
docker cp "${CID}:/output/." "${OUTPUT_DIR}/"
docker rm "${CID}" >/dev/null
docker rmi "${IMAGE_TAG}" >/dev/null || true

log_info "Done. Packages in ${OUTPUT_DIR}:"
ls -lh "${OUTPUT_DIR}"/*.deb
