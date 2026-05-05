# TLE Reference Notes

This file is a local reference for FlagTree/TLE APIs that are useful when
authoring FlagFFT Triton kernels. It is not the FlagFFT architecture guide; see
`docs/architecture.md` for project structure and current implementation status.

### 3.2 TLE-Lite

- Design philosophy: write once, run anywhere.
- Core idea: use high-level semantic hints (instead of hard constraints) to guide compiler heuristics. Keep backward compatibility and achieve cross-platform speedups with minimal code changes.

#### 3.2.1 Memory Management

##### 3.2.1.1 `tle.load`

Extension of `tl.load` with async hint support:

```python
x = tle.load(..., is_async=True)
```

#### 3.2.2 Tensor Slicing

##### 3.2.2.1 `tle.extract_tile`

Split input tensor into a sub-tile grid using a child-tile shape and extract tile at specified coordinates.

- GPU: supports extraction from registers and shared memory.

```python
# x is [4, 4]
# z is [2, 2]
# Split x into shape=[2, 2] sub-tiles and return tile at [0, 0]
z = x.extract_tile(index=[0, 0], shape=[2, 2])
```

##### 3.2.2.2 `tle.insert_tile`

Split input tensor into a sub-tile grid using child-tile shape and update tile at specified coordinates.

- GPU: supports updates in registers and shared memory.

```python
# x is [4, 4], y is [2, 2], z is [4, 4]
# Split x into shape=[2, 2] sub-tiles, update tile [0, 0] with y,
# and return full updated [4, 4] tensor
z = x.insert_tile(y, index=[0, 0])
```

#### 3.2.3 Pipeline

##### 3.2.3.1 `tle.pipeline_group`

Hint-style extension.

Automatic stage partitioning:

```python
for yoff in tl.range(0, ynumel, YBLOCK, num_stages=2):
    Q = tl.load(...)
    K = tl.load(...)
    KT = tl.trans(K)
    V = tl.dot(Q, KT)
```

Manual stage partitioning:

```python
for yoff in tle.range(
    0,
    ynumel,
    YBLOCK,
    num_stages=2,
    pipe_stages=[0, 0, 1] if LOAD_TRANS else [0, 1, 1],
    pipe_orders=[0, 1, 2],
    executors=[0, 0, 0] if ONE_CORE else [0, 0, range(1, 31)],
):
    # Warp specialization or heterogeneous units
    with tle.pipeline_group(0):
        Q = tl.load(...)
        K = tl.load(...)
    with tle.pipeline_group(1):
        KT = tl.trans(K)
    with tle.pipeline_group(2):
        V = tl.dot(Q, KT)
```

#### 3.2.4 Distributed

Triton distributed API has four core parts: device mesh definition, sharding specification, resharding (collective communication), and remote access (point-to-point communication).

##### 3.2.4.1 Device Mesh

`tle.device_mesh` defines physical device topology and serves as the context foundation for distributed operations.

```python
class device_mesh:
    def __init__(self, topology: dict):
        """
        Initialize DeviceMesh.

        Args:
            topology (dict): Hardware hierarchy description.
                             Keys are hierarchy names; values are int (1D)
                             or tuple lists (multi-dimensional).
        """
        self._physical_ids = ... # Internal flattened physical IDs (0..N-1)
        self._shape = ...        # Current logical shape, e.g. (2, 2, 4, 2, 2, 4)
        self._dim_names = ...    # Current dimension names

    @property
    def shape(self):
        """Return logical mesh shape."""
        return self._shape

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    def flatten(self):
        """Flatten mesh to 1D, typically for ring communication."""
        return self.reshape(prod(self._shape))

    def __getitem__(self, key):
        """
        Supports slicing and returns a sub-mesh.
        Supports standard slice and integer indexing.
        """
        return sub_mesh

    def __repr__(self):
        return f"DeviceMesh(shape={self._shape}, names={self._dim_names})"


# Define complex hardware hierarchy
topology = {
    # Cross-node hierarchy (2x2 = 4 nodes)
    "node": [("node_x", 2), ("node_y", 2)],
    # In-node GPUs (4 devices)
    "device": 4,
    # In-GPU cluster (2x2)
    "block_cluster": [("cluster_x", 2), ("cluster_y", 2)],
    # In-cluster blocks (4 blocks)
    "block": 4,
}

# mesh.shape -> (2, 2, 4, 2, 2, 4)
# total size = 256
mesh = tle.device_mesh(topology=topology)
```

##### 3.2.4.2 Sharding Specification

`tle.sharding` declares tensor distribution state on the device mesh:

- `splits`: how each tensor axis is partitioned on mesh axes.
- `partials`: whether tensor is partial-sum state.
- Unspecified mesh axes are treated as broadcast.

Symbols:

- `tle.S(axis)`: split.
- `tle.B`: broadcast/replicate.
- `tle.P(axis)`: partial; requires reduce on specified axis.

```python
def sharding(tensor, splits, partials):
    """
    Annotation only: marks tensor state, emits no direct code,
    but guides compiler checks and optimizations.
    """
    return tensor

# Split axis0 on cluster, axis1 on device, and partial on block axis
x_shard = tle.sharding(
    mesh,
    split=[["cluster_x", "cluster_y"], "device"],
    partial=["block"],
)

# Define a sharded tensor
x = tle.make_sharded_tensor(x_ptr, sharding=x_shard, shape=[4, 4])
```

##### 3.2.4.3 Synchronization

In complex distributed kernels (e.g., ring all-reduce or row/column-independent pipelines), only “same-row” or “same-column” blocks often need synchronization rather than the whole cluster. Global synchronization introduces unnecessary waiting.

```python
def distributed_barrier(mesh):
    """
    If sub_mesh is passed, synchronize only devices in this sub-mesh.
    Devices outside this sub-mesh should treat it as No-Op
    (or compiler guarantees control flow does not enter).
    """
    pass
```

##### 3.2.4.4 Remote Access

`tle.remote` obtains a handle for tensor data located on other devices. This maps to point-to-point communication or direct memory access (RDMA/NVLink load).

```python
def remote(tensor, shard_id, scope):
    """
    Get a RemoteTensor handle to a shard on a target device.

    :param tensor: logically distributed tensor (already marked by tle.sharding)
    :param shard_id: tuple coordinate in device mesh
    :return: RemoteTensor, supporting load/store and related ops
    """
```

##### 3.2.4.5 Resharding

`tle.reshard` is the entrypoint for collectives. Compiler compares source and target specs and inserts communication primitives automatically.

```python
def reshard(tensor, spec):
    """
    Action: transform tensor to a new distribution state.

    Typical transitions:
    1. [ ] -> [S]: Scatter
    2. [S] -> [ ]: Gather
    3. [P] -> [ ]: Reduce
    4. [B] -> [S]: Local slice (no communication)
    5. [S] -> [B]: All-gather
    6. [P] -> [B]: All-reduce
    7. [B] -> [P]: Error
    """
```

##### 3.2.4.6 Distributed GEMM

NVIDIA Hopper (H100) and newer architectures introduce Thread Block Cluster, allowing groups of CTAs to cooperate via DSMEM for high-bandwidth, low-latency exchange.

`tle.distributed_dot` is designed to use this feature so developers can write cross-block matrix multiplication without manually handling DSMEM barriers and data movement.

```python
def distributed_dot(a, b, c=None):
    """
    Execute distributed matrix multiplication within current
    Thread Block Cluster scope.

    Behavior depends on sharding specs of input tensors `a` and `b`
    over the cluster mesh.

    Args:
        a (Tensor): left operand with cluster-level sharding annotation.
        b (Tensor): right operand with cluster-level sharding annotation.
        c (Tensor, optional): accumulator.

    Returns:
        Tensor: result tensor with distribution inferred from inputs.
    """
```

Open question: what additional distributed primitives are needed?

#### 3.2.5 API Reference and Practical Examples

##### 3.2.5.1 `tle.load`

- Signature: `tle.load(ptr, mask=None, other=None, is_async=False)`
- Use case: Keep `tl.load` semantics while adding async scheduling hints.
- Practical guidance:
  - Use `is_async=True` for global-memory reads that are later reused in compute-heavy regions.
  - Keep `mask` and `other` explicit on boundary tiles to avoid undefined values.

Example: guarded async load for tail tiles

```python
offs = base + tl.arange(0, BLOCK)
mask = offs < n_elements
x = tle.load(x_ptr + offs, mask=mask, other=0.0, is_async=True)
```

Example: async load + compute overlap pattern

```python
for k in tl.range(0, K, BK, num_stages=2):
    a = tle.load(a_ptr + k * stride_a, is_async=True)
    b = tle.load(b_ptr + k * stride_b, is_async=True)
    acc = tl.dot(a, b, acc)
```

##### 3.2.5.2 `tle.extract_tile` and `tle.insert_tile`

- `extract_tile`: read a sub-tile view from a larger tile tensor.
- `insert_tile`: write a processed sub-tile back to a larger tile tensor.
- Typical use: local transforms (activation, quant/dequant, normalization) on sub-regions without manual pointer arithmetic.

Example: tilewise post-processing in registers

```python
# x: [4, 4]
sub = x.extract_tile(index=[1, 0], shape=[2, 2])  # rows [2:4], cols [0:2]
sub = tl.maximum(sub, 0.0)  # ReLU on the sub-tile
x = x.insert_tile(sub, index=[1, 0])
```

##### 3.2.5.3 `tle.pipeline_group`

- Use `tle.pipeline_group(stage_id)` to explicitly tag operations into stages.
- Useful when you need deterministic stage control (instead of fully heuristic grouping).

Example: staged load-transform-matmul

```python
for k in tle.range(0, K, BK, num_stages=2, pipe_stages=[0, 0, 1], pipe_orders=[0, 1, 2]):
    with tle.pipeline_group(0):
        a = tl.load(a_ptr + k * stride_a)
        b = tl.load(b_ptr + k * stride_b)
    with tle.pipeline_group(1):
        bt = tl.trans(b)
    with tle.pipeline_group(2):
        acc = tl.dot(a, bt, acc)
```

##### 3.2.5.4 `tle.device_mesh` + `tle.sharding` + `tle.reshard`

- Recommended workflow:
  1. Define topology with `tle.device_mesh`.
  2. Mark tensor layout with `tle.sharding`.
  3. Transform layout with `tle.reshard`.
  4. Keep compute kernels operating on logical tensor views.

Example: split-by-device input, then all-gather before compute

```python
mesh = tle.device_mesh({"node": 2, "device": 4})
x_spec = tle.sharding(mesh, split=["device"], partial=[])
x = tle.make_sharded_tensor(x_ptr, sharding=x_spec, shape=[M, K])

# [S] -> [B] on device axis (all-gather)
x_full = tle.reshard(x, spec=tle.sharding(mesh, split=[], partial=[]))
```

##### 3.2.5.5 `tle.shard_id`

- Signature: `tle.shard_id(mesh, axis)`
- Returns current program's coordinate on a mesh axis.
- `axis` can be a mesh-axis name (e.g. `"node"`, `"device"`, `"cluster_x"`) or an axis index.
- Typical use: build peer shard IDs for ring exchange, staged all-reduce, and cluster-cooperative kernels.

Example: query current program coordinates on node/device axes

```python
mesh = tle.device_mesh({"node": 2, "device": 4})
node_rank = tle.shard_id(mesh, "node")      # 0..1
device_rank = tle.shard_id(mesh, "device")  # 0..3
```

##### 3.2.5.6 `tle.remote` + `tle.distributed_barrier`

- `tle.remote` reads/writes explicit remote shards.
- `tle.distributed_barrier` synchronizes only the mesh/sub-mesh you pass in.

Example: remote read from neighbor shard (ring-like exchange)

```python
node_rank = tle.shard_id(mesh, "node")
device_rank = tle.shard_id(mesh, "device")
next_device = (device_rank + 1) % mesh.shape[1]
remote_x = tle.remote(x, shard_id=(node_rank, next_device), scope=mesh)
tle.distributed_barrier(mesh)
neighbor_vals = tl.load(remote_x)
```

### 3.3 TLE-Struct

- Design philosophy: architecture-aware, fine-grained tuning.
- Core idea: classify backends by hardware-topology families (e.g., GPGPU, DSA), expose common hierarchical parallel/storage structures, and let developers explicitly define structured compute/data mappings (e.g., warp-group control, pipeline scheduling). This decouples algorithm logic from hardware physical implementation at the abstraction level.

#### 3.3.1 GPU

##### 3.3.1.1 Memory Management

###### 3.3.1.1.1 `tle.gpu.memory_space`

Specify tensor `memory_space`:

```python
x = ...
x = tle.gpu.memory_space(x, "shared_memory")
```

###### 3.3.1.1.2 `tle.gpu.alloc`

Allocate memory:

```python
a_smem = tle.gpu.alloc(
    [XBLOCK, YBLOCK],
    dtype=tl.float32,
    layout=None,
    scope=tle.gpu.storage_kind.smem,
)
```

###### 3.3.1.1.3 `tle.gpu.local_ptr`

Get memory pointers:

```python
# pointers for a_smem[0, :]: [(0, 0), (0, 1), ..., (0, YBLOCK-1)]
a_smem_ptrs = tle.gpu.local_ptr(
    a_smem,
    indices=(tl.broadcast(0, [YBLOCK]), tl.arange(0, YBLOCK)),
)
```

- Signature: `tle.gpu.local_ptr(buffer, indices=None) -> tl.tensor | tl.ptr`
- Purpose: Build arbitrary-shaped pointer views over shared memory buffers for `tl.load/tl.store/tl.atomic*`.
- Parameters:
  - `buffer`: buffered tensor returned by `tle.gpu.alloc` (SMEM/TMEM).
  - `indices`: optional tuple of integer tensors. Tuple length must equal `rank(buffer)`, and all tensors must have identical shapes. If omitted/`None`, backend treats it as full indices.
- Semantics:
  - If `indices` is provided: output pointer tensor shape equals common shape of index tensors.
  - For each logical output index `(i0, i1, ...)`, pointer value corresponds to `buffer[indices0(i0,...), indices1(i0,...), ...]`.
  - If `indices=None`: build full-view pointers over `buffer` shape (rank>0 returns pointer tensor with `shape(buffer)`, rank=0 returns scalar pointer).
  - Returned pointers live in shared-memory address space (LLVM addrspace=3). Indices must be integers (i32/i64, etc.; lowered to i32).
  - Linearization is row-major (last dimension fastest); shared-memory layout/encoding follows buffer memdesc.

Example 1: 1D slice

```python
smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
# Slice [offset, offset + SLICE)
idx = offset + tl.arange(0, SLICE)
slice_ptr = tle.gpu.local_ptr(smem, (idx,))
vals = tl.load(slice_ptr)
```

Example 2: K-dimension tiling (matrix slice)

```python
smem_a = tle.gpu.alloc([BM, BK], dtype=tl.float16, scope=tle.gpu.smem)
# Slice (BM, KW), where KW is K-dimension slice
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, KW))
cols = tl.broadcast_to(tl.arange(0, KW)[None, :] + k_start, (BM, KW))
a_slice = tle.gpu.local_ptr(smem_a, (rows, cols))
a_vals = tl.load(a_slice)
```

Example 3: arbitrary gather view

```python
smem = tle.gpu.alloc([H, W], dtype=tl.float32, scope=tle.gpu.smem)
# Take an offset column per row
rows = tl.broadcast_to(tl.arange(0, H)[:, None], (H, SLICE))
cols = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (H, SLICE))
gather_ptr = tle.gpu.local_ptr(smem, (rows, cols))
out = tl.load(gather_ptr)
```

Supported downstream ops:

- `tl.load`
- `tl.store`
- `tl.atomic_add/and/cas/max/min/or/xchg/xor`

Practical notes:

- Atomic ops require element dtype/backend support; use integer/float types supported by target hardware.
- For local-pointer load-after-store hazards, TLE backend pass `TleInsertLocalPointerBarriers` inserts barriers automatically; add manual barriers only for custom synchronization patterns outside pass coverage.

Example 4: load/store/atomic on the same `local_ptr`

```python
smem_i32 = tle.gpu.alloc([BLOCK], dtype=tl.int32, scope=tle.gpu.smem)
ptr = tle.gpu.local_ptr(smem_i32, (tl.arange(0, BLOCK),))

tl.store(ptr, tl.zeros([BLOCK], dtype=tl.int32))
tl.atomic_add(ptr, 1)
vals = tl.load(ptr)
```

###### 3.3.1.1.4 `tle.gpu.local_ptr` (for remote)

- Signature: `tle.gpu.local_ptr(remote_buffer, indices=None) -> tl.tensor | tl.ptr`
- Purpose: materialize pointer views for remote shared/local buffers returned by `tle.remote(...)`.
- Inputs:
  - `remote_buffer`: result of `tle.remote(buffer, shard_id, scope)`, where `buffer` is typically allocated by `tle.gpu.alloc`.
  - `indices`: same rules as local mode (`None` for full view, or tuple of integer tensors with identical shapes).
- Semantics:
  - Pointer shape/linearization rules are identical to local `tle.gpu.local_ptr`.
  - Address resolution targets the remote shard selected by `shard_id`.
  - Use `tle.distributed_barrier(...)` when cross-shard producer/consumer ordering is required.

Example: read remote SMEM tile from neighbor shard

```python
smem = tle.gpu.alloc([BM, BK], dtype=tl.float16, scope=tle.gpu.storage_kind.smem)
remote_smem = tle.remote(smem, shard_id=(node_rank, next_device), scope=mesh)

rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
remote_ptr = tle.gpu.local_ptr(remote_smem, (rows, cols))

vals = tl.load(remote_ptr)
```

###### 3.3.1.1.5 `tle.gpu.copy`

Memory copy:

```python
tle.gpu.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
```

#### 3.3.2 DSA

This section is rewritten from `triton_v3.2.x` (`python/triton/experimental/tle/language/dsa` and its README).
DSA APIs are split into:

- Generic DSA APIs under `tle.dsa.*`
- Backend-specific address spaces under `tle.dsa.ascend.*`

##### 3.3.2.1 Memory and Data Movement

###### 3.3.2.1.1 `tle.dsa.alloc`

- Signature: `tle.dsa.alloc(shape, dtype, mem_addr_space)`
- Purpose: allocate DSA local buffers in a target memory space.

Ascend memory spaces exposed in source:

- `tle.dsa.ascend.UB`
- `tle.dsa.ascend.L1`
- `tle.dsa.ascend.L0A`
- `tle.dsa.ascend.L0B`
- `tle.dsa.ascend.L0C`

```python
a_ub = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
b_l1 = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.L1)
```

###### 3.3.2.1.2 `tle.dsa.copy`

- Signature: `tle.dsa.copy(src, dst, shape, inter_no_alias=False)`
- Purpose: explicit movement between GMEM pointers and DSA local buffers (both directions).

```python
tle.dsa.copy(x_ptrs, a_ub, [tail_m, tail_n])          # GMEM -> local buffer
tle.dsa.copy(a_ub, out_ptrs, [tail_m, tail_n])        # local buffer -> GMEM
```

###### 3.3.2.1.3 `tle.dsa.local_ptr`

- Signature: `tle.dsa.local_ptr(buffer, indices=None) -> tl.tensor | tl.ptr`
- Purpose: build pointer views over DSA local buffers (for example UB/L1) for explicit local-memory access patterns.
- Parameters:
  - `buffer`: DSA buffered tensor, typically from `tle.dsa.alloc`.
  - `indices`: optional tuple of integer tensors. If omitted/`None`, backend treats it as full indices.
- Semantics:
  - Shape and indexing behavior follow `tle.gpu.local_ptr` (same pointer-view model).
  - Intended for DSA-local data access paths that require explicit pointer materialization.

Example:

```python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
a_ptr = tle.dsa.local_ptr(a_ub, (rows, cols))
a_val = tl.load(a_ptr)
```

###### 3.3.2.1.4 `tle.dsa.local_ptr` (for remote)

- Signature: `tle.dsa.local_ptr(remote_buffer, indices=None) -> tl.tensor | tl.ptr`
- Purpose: materialize pointer views over remote DSA local buffers obtained from `tle.remote(...)`.
- Inputs:
  - `remote_buffer`: result of `tle.remote(dsa_buffer, shard_id, scope)`.
  - `indices`: same rules as local DSA mode.
- Semantics:
  - Same pointer-view semantics as local DSA mode.
  - Pointer dereference is routed to the remote shard selected by `shard_id`.
  - Pair with `tle.distributed_barrier` when cross-shard ordering is required.

Example:

```python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
remote_a_ub = tle.remote(a_ub, shard_id=peer_rank, scope=mesh)

rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
remote_ptr = tle.dsa.local_ptr(remote_a_ub, (rows, cols))
remote_val = tl.load(remote_ptr)
```

###### 3.3.2.1.5 `tle.dsa.to_tensor` / `tle.dsa.to_buffer`

- `tle.dsa.to_tensor(buffer, writable=True)`: convert a DSA buffer to a tensor view for tensor expressions.
- `tle.dsa.to_buffer(tensor, space)`: convert a tensor value back to a buffer in a target DSA address space.

```python
c_val = tle.dsa.to_tensor(c_ub, writable=True)
result = c_val * 0.5
d_ub = tle.dsa.to_buffer(result, tle.dsa.ascend.UB)
tle.dsa.copy(d_ub, out_ptrs, [tail_m, tail_n])
```

##### 3.3.2.2 Elementwise Compute Ops (buffer-based)

Builtins provided by source:

- `tle.dsa.add`
- `tle.dsa.sub`
- `tle.dsa.mul`
- `tle.dsa.div`
- `tle.dsa.max`
- `tle.dsa.min`

- Common signature: `tle.dsa.<op>(lhs, rhs, out)`
- Compute model: elementwise binary op over DSA local buffers.
- Shape rules:
  - `lhs`, `rhs`, `out` must have the same rank and shape.
  - No implicit broadcast is assumed in this API layer.
- Dtype rules:
  - Three operands should use the same dtype in practice.
  - Integer dtypes are typical for index/count paths; float dtypes are typical for activation/math paths.
- Memory-space rules:
  - Buffers should be allocated in compatible DSA local spaces (for example UB/L1 combinations allowed by backend).
  - Keep hot operands/results in local space to avoid extra GMEM traffic.

Per-op semantics:

- `tle.dsa.add(lhs, rhs, out)`: `out = lhs + rhs`
- `tle.dsa.sub(lhs, rhs, out)`: `out = lhs - rhs`
- `tle.dsa.mul(lhs, rhs, out)`: `out = lhs * rhs`
- `tle.dsa.div(lhs, rhs, out)`: `out = lhs / rhs` (backend-dependent precision/rounding)
- `tle.dsa.max(lhs, rhs, out)`: `out = max(lhs, rhs)`
- `tle.dsa.min(lhs, rhs, out)`: `out = min(lhs, rhs)`

In-place usage:

- You can reuse the same output buffer across steps, for example `tle.dsa.mul(tmp, b, tmp)`.
- Avoid aliasing inputs/outputs unless backend semantics explicitly allow it.

Example 1: arithmetic chain `((a - b) * b) / scale`

```python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
b_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
scale_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
tmp_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
out_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)

tle.dsa.copy(a_ptrs, a_ub, [BM, BK])
tle.dsa.copy(b_ptrs, b_ub, [BM, BK])
tle.dsa.copy(scale_ptrs, scale_ub, [BM, BK])

tle.dsa.sub(a_ub, b_ub, tmp_ub)      # tmp = a - b
tle.dsa.mul(tmp_ub, b_ub, tmp_ub)    # tmp = tmp * b
tle.dsa.div(tmp_ub, scale_ub, out_ub)  # out = tmp / scale

tle.dsa.copy(out_ub, out_ptrs, [BM, BK])
```

Example 2: clamp by `max` + `min`

```python
x_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
floor_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
ceil_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
tmp_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
y_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)

tle.dsa.copy(x_ptrs, x_ub, [BM, BK])
tle.dsa.copy(floor_ptrs, floor_ub, [BM, BK])
tle.dsa.copy(ceil_ptrs, ceil_ub, [BM, BK])

tle.dsa.max(x_ub, floor_ub, tmp_ub)  # tmp = max(x, floor)
tle.dsa.min(tmp_ub, ceil_ub, y_ub)   # y = min(tmp, ceil)

tle.dsa.copy(y_ub, y_ptrs, [BM, BK])
```

```python
tle.dsa.add(a_ub, b_ub, c_ub)
tle.dsa.mul(c_ub, b_ub, c_ub)
```

##### 3.3.2.3 Loop and Hint APIs

Source includes:

- `tle.dsa.pipeline(...)`
- `tle.dsa.parallel(...)`
- `tle.dsa.hint(...)` (used as `with tle.dsa.hint(...)` compile-time hints)

```python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(x_ptr + offs, a_ub, [tail_size], inter_no_alias=True)
```

##### 3.3.2.4 Slice/View Utilities

Source includes:

- `tle.dsa.extract_slice`
- `tle.dsa.insert_slice`
- `tle.dsa.extract_element`
- `tle.dsa.subview`

```python
sub = tle.dsa.extract_slice(full, offsets=(0, k0), sizes=(BM, BK), strides=(1, 1))
full = tle.dsa.insert_slice(full, sub, offsets=(0, k0), sizes=(BM, BK), strides=(1, 1))
elem = tle.dsa.extract_element(sub, indice=(i, j))
```

#### 3.3.3 Struct API Cookbook

##### 3.3.3.1 Shared-memory staging (`alloc` + `copy` + `local_ptr`)

Use this pattern when data is reused across multiple math operations.

```python
# 1) Allocate SMEM tile
a_smem = tle.gpu.alloc([BM, BK], dtype=tl.float16, scope=tle.gpu.storage_kind.smem)

# 2) Copy GMEM -> SMEM
tle.gpu.copy(a_ptrs, a_smem, [BM, BK])

# 3) Build local pointer view and load
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
a_ptr_local = tle.gpu.local_ptr(a_smem, (rows, cols))
a_tile = tl.load(a_ptr_local)
```

##### 3.3.3.2 Shared-memory atomics with `local_ptr`

Useful for histogram, bucketization, and radix-select style counting.

```python
bins = 256
counts = tle.gpu.alloc([bins], dtype=tl.int32, scope=tle.gpu.storage_kind.smem)
idx = tl.arange(0, BLOCK) % bins
count_ptr = tle.gpu.local_ptr(counts, (idx,))
tl.atomic_add(count_ptr, 1)
```

##### 3.3.3.3 DSA local-buffer flow (`dsa.alloc` + `dsa.copy` + `dsa.to_tensor/to_buffer`)

Use this for DSA backends that expose dedicated local buffer spaces.

```python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
b_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
c_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)

tle.dsa.copy(a_ptrs, a_ub, [BM, BK])
tle.dsa.copy(b_ptrs, b_ub, [BM, BK])
tle.dsa.add(a_ub, b_ub, c_ub)

c_val = tle.dsa.to_tensor(c_ub, writable=True)
out_ub = tle.dsa.to_buffer(c_val, tle.dsa.ascend.UB)
tle.dsa.copy(out_ub, out_ptrs, [BM, BK])
```
