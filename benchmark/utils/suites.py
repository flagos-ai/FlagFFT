"""Benchmark size suites for FlagFFT performance testing."""

# All sizes from ctest k1DSizesSmall + k1DSizesMedium + k1DSizesLarge
ALL_SIZES = [16, 23, 64, 81, 243, 256, 361, 512, 997, 2048, 4096, 8192, 16384]

# All batch values from ctest k1DBatchValues
ALL_BATCHES = [1, 4, 256]

# All APIs from ctest coverage
ALL_APIS = ["c2c", "z2z", "r2c", "c2r", "d2z", "z2d"]

# 2D complex APIs only (rank 2 does not support real transforms yet)
ALL_APIS_2D = ["c2c", "z2z"]

# Direction support per API
API_DIRECTIONS = {
    "c2c": ["forward", "inverse"],
    "z2z": ["forward", "inverse"],
    "r2c": ["forward"],
    "d2z": ["forward"],
    "c2r": ["inverse"],
    "z2d": ["inverse"],
}

# 2D sizes: smooth (CT path), non-square, Bluestein (prime dimensions)
ALL_SIZES_2D = ["16x16", "64x64", "128x256", "256x128", "256x256", "512x512"]
BLUESTEIN_SIZES_2D = ["23x23", "997x16", "16x997"]

# ── Suite definitions ──────────────────────────────────────────

# Smoke (24 1D cases + 4 2D cases): key sizes x batch=1 x all API/direction combos
SMOKE = {
    "sizes": [16, 256, 2048],
    "batches": [1],
    "sizes_2d": ["16x16", "256x256"],
    "batches_2d": [1],
}

# Typical (136 1D + 24 2D cases)
TYPICAL = {
    "sizes": ALL_SIZES,
    "batches": [1],
    "extra": [(256, 4), (256, 256), (2048, 4), (2048, 256)],
    "sizes_2d": ALL_SIZES_2D,
    "batches_2d": [1],
}

# Full (312 1D + 48 2D cases)
FULL = {
    "sizes": ALL_SIZES,
    "batches": ALL_BATCHES,
    "sizes_2d": ALL_SIZES_2D + BLUESTEIN_SIZES_2D,
    "batches_2d": [1, 4],
}

SUITES = {"smoke": SMOKE, "typical": TYPICAL, "full": FULL}


def get_suite(name: str) -> dict:
    """Return suite definition by name."""
    if name not in SUITES:
        raise KeyError(f"Unknown benchmark suite {name!r}. Choices: {list(SUITES)}")
    return SUITES[name]


def expand_params(suite: dict):
    """Yield (size, batch, api, direction) tuples for a suite definition.

    Handles the 'extra' key for TYPICAL suite: additional (size, batch)
    pairs beyond the base sizes x batches cross product.

    Also yields 2D entries from 'sizes_2d' x 'batches_2d' x ALL_APIS_2D.
    2D sizes are strings like "256x256" (detected by the caller to set rank=2).
    """
    apis = ALL_APIS
    seen = set()

    # Base 1D: sizes x batches x apis
    for size in suite["sizes"]:
        for batch in suite["batches"]:
            for api in apis:
                for direction in API_DIRECTIONS[api]:
                    key = (size, batch, api, direction)
                    if key not in seen:
                        seen.add(key)
                        yield key

    # Extra 1D (size, batch) pairs applied to all API/direction combos
    for size, batch in suite.get("extra", []):
        for api in apis:
            for direction in API_DIRECTIONS[api]:
                key = (size, batch, api, direction)
                if key not in seen:
                    seen.add(key)
                    yield key

    # 2D: sizes_2d x batches_2d x ALL_APIS_2D (forward only)
    for size_2d in suite.get("sizes_2d", []):
        for batch in suite.get("batches_2d", [1]):
            for api in ALL_APIS_2D:
                for direction in API_DIRECTIONS[api]:
                    key = (size_2d, batch, api, direction)
                    if key not in seen:
                        seen.add(key)
                        yield key
