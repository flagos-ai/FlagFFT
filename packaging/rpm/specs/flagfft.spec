Name:           libflagfft-nvidia
Version:        0.1.0
Release:        1%{?dist}
Summary:        FlagFFT — C++ FFT library for FlagOS (NVIDIA backend)

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagFFT
Source0:        %{name}-%{version}.tar.gz

# CUDA/PyTorch/Triton/pybind11 are container-provided.
BuildRequires:  cmake >= 3.18
BuildRequires:  ninja-build
BuildRequires:  gcc-c++
BuildRequires:  python3-devel
BuildRequires:  patchelf
BuildRequires:  sqlite-devel
BuildRequires:  json-devel

%description
FlagFFT is a cuFFT-style FFT library with Triton/TLE code generation,
targeting FlagOS multi-vendor accelerators. The NVIDIA backend uses
CUDA + Triton-JIT for kernel generation. libtriton_jit is bundled
statically into libflagfft.so.

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description devel
Header files and CMake config for building applications against
libflagfft-nvidia.

%prep
%autosetup -n %{name}-%{version}

%build
# Ensure pip-installed packages (torch, triton, pybind11) are visible.
PY3_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=$(python3 -c "import site; print(':'.join(site.getsitepackages()))"):/usr/local/lib/python${PY3_VER}/site-packages:/usr/local/lib64/python${PY3_VER}/site-packages
export PATH=/usr/local/bin:$PATH
%cmake \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLAGFFT_LIBTRITON_JIT_BACKEND=CUDA \
    -DFLAGFFT_BUILD_TESTS=OFF \
    -DFLAGFFT_BUILD_BENCHMARKS=OFF
%cmake_build

%install
%cmake_install
# Fix RPATH if any.
find %{buildroot}/usr/lib -name "*.so*" -type f -exec patchelf --remove-rpath {} \; || true

%files
%license LICENSE
%doc README.md
# FlagFFT CMakeLists hardcodes "lib" (not CMAKE_INSTALL_LIBDIR), so .so
# lands at /usr/lib even on Fedora where %_libdir = /usr/lib64.
/usr/lib/libflagfft.so

%files devel
/usr/include/*

%changelog
* Wed May 21 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM release for the NVIDIA backend.
