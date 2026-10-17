# FlagTree Ascend dependency patches

The Ascend tensor FFT generator uses the standard Triton language and Ascend
TLE operations. Its compiler must support the operations used by the selected
exchange configuration; importing TLE alone does not qualify a configuration.

The initial compiler is FlagTree commit
`bb9b00998685047cd1d7eb643f48b6bd9d84c46a`, distributed as
`0.7.0+ascend.gitbb9b0099` with Triton 3.5.1. At that revision, importing the
single-device TLE API eagerly imports `shmem` from the distributed runtime.
`flagtree-ascend-optional-shmem.patch` defers that optional dependency until a
distributed operation is called. Apply it to the matching FlagTree source
before installation:

```bash
git -C FlagTree checkout bb9b00998685047cd1d7eb643f48b6bd9d84c46a
git -C FlagTree apply /path/to/FlagFFT/packaging/patches/flagtree-ascend-optional-shmem.patch
```

This patch changes Python import timing only. It adds no device kernel or
external FFT implementation. Compiler execution and performance qualification
are recorded separately from the dependency import check.
