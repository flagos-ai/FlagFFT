Name:           libflagfft-nvidia
Version:        0.1.0
Release:        1%{?dist}
Summary:        FlagFFT — C++ FFT library for FlagOS (NVIDIA backend)

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagFFT
Source0:        %{name}-%{version}.tar.gz

# The CUDA driver and pip-installed Torch libraries are runtime prerequisites,
# but they are not owned by RPM packages in the container build environment.
%global __requires_exclude ^(libc10|libcuda|libtorch.*)\\.so.*$

# CUDA/PyTorch/Triton/pybind11 are container-provided.
BuildRequires:  binutils
BuildRequires:  cmake >= 3.18
BuildRequires:  python3-ninja
BuildRequires:  gcc-c++
BuildRequires:  python3-devel
BuildRequires:  fmt-devel >= 8.1.1
BuildRequires:  sqlite-devel
BuildRequires:  nlohmann-json-devel >= 3.10.5

%description
FlagFFT is a cuFFT-style FFT library with Triton/TLE code generation,
targeting FlagOS multi-vendor accelerators. The NVIDIA backend uses
CUDA + Triton-JIT for kernel generation. libtriton_jit is bundled
as the private libflagfft_triton_jit.so runtime. PyTorch 2.5.1 and the
CUDA driver remain external runtime prerequisites.

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description devel
Header files for building applications against libflagfft-nvidia.

%prep
%autosetup -n %{name}-%{version}

%build
# Ensure pip-installed packages (torch, triton, pybind11) are visible.
PY3_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=$(python3 -c "import site; print(':'.join(site.getsitepackages()))"):/usr/local/lib/python${PY3_VER}/site-packages:/usr/local/lib64/python${PY3_VER}/site-packages
export PATH=/usr/local/bin:$PATH
NVIDIA_PY_ROOT=$(python3 -c "import nvidia; print(next(iter(nvidia.__path__)))")
%cmake \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
    -Dnvtx3_dir="${NVIDIA_PY_ROOT}/nvtx/include" \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFLAGFFT_LIBTRITON_JIT_BACKEND=CUDA \
    -DFLAGFFT_BUILD_TESTS=OFF \
    -DFLAGFFT_BUILD_BENCHMARKS=OFF \
    -DBUILD_TESTING=OFF \
    -DTRITON_JIT_USE_EXTERNAL_JSON=ON \
    -DTRITON_JIT_USE_EXTERNAL_FMTLIB=ON \
    -DTRITON_JIT_USE_EXTERNAL_PYBIND11=ON \
    -DTRITON_JIT_BUILD_OPERATORS=OFF \
    -DFMT_INSTALL=OFF
%{__cmake} --build . --parallel %{_smp_build_ncpus}

%install
DESTDIR=%{buildroot} %{__cmake} --install .
install -Dm0644 packaging/common/flagfft-triton-jit.pth \
    %{buildroot}%{python3_sitelib}/flagfft-triton-jit.pth

%check
test -f %{buildroot}%{_libdir}/libflagfft.so
test -f %{buildroot}%{_libdir}/libflagfft_triton_jit.so
test -f %{buildroot}%{_datadir}/flagfft/python/src/codegen/jit_source.py
test -f %{buildroot}%{_datadir}/triton_jit/scripts/standalone_compile.py
test -f %{buildroot}%{python3_sitelib}/flagfft-triton-jit.pth
readelf -d %{buildroot}%{_libdir}/libflagfft.so | grep -q '\[libflagfft_triton_jit.so\]'
! readelf -d %{buildroot}%{_libdir}/libflagfft.so | grep -q '\[libtriton_jit.so\]'

%files
%license LICENSE
%doc README.md
%{_libdir}/libflagfft.so
%{_libdir}/libflagfft_triton_jit.so
%{_datadir}/flagfft/python/src/
%{_datadir}/triton_jit/scripts/
%{python3_sitelib}/flagfft-triton-jit.pth

%files devel
%{_includedir}/flagfft/

%changelog
* Thu May 21 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM release for the NVIDIA backend.
