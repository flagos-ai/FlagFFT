"""Benchmark size suites for FlagFFT performance testing."""

# All sizes from ctest k1DSizesSmall + k1DSizesMedium + k1DSizesLarge
ALL_SIZES = [16, 23, 64, 81, 243, 256, 361, 512, 997, 2048, 4096, 8192, 16384]

# All batch values from ctest k1DBatchValues
ALL_BATCHES = [1, 4, 256]

# All APIs from ctest coverage
ALL_APIS = ["c2c", "z2z", "r2c", "c2r", "d2z", "z2d"]

# Direction support per API
API_DIRECTIONS = {
    "c2c": ["forward", "inverse"],
    "z2z": ["forward", "inverse"],
    "r2c": ["forward"],
    "d2z": ["forward"],
    "c2r": ["inverse"],
    "z2d": ["inverse"],
}

# ── Suite definitions ──────────────────────────────────────────

# Smoke (24 cases): 3 key sizes x batch=1 x all API/direction combos
SMOKE = {"sizes": [16, 256, 2048], "batches": [1]}

# Typical (136 cases): all sizes x batch=1 + extra multi-batch on key sizes
TYPICAL = {
    "sizes": ALL_SIZES,
    "batches": [1],
    "extra": [(256, 4), (256, 256), (2048, 4), (2048, 256)],
}

# Full (312 cases): all sizes x all batches x all API/direction combos
FULL = {"sizes": ALL_SIZES, "batches": ALL_BATCHES}

SUITES = {"smoke": SMOKE, "typical": TYPICAL, "full": FULL}


def get_suite(name: str) -> dict:
    """Return suite definition by name."""
    return SUITES[name]


def expand_params(suite: dict):
    """Yield (size, batch, api, direction) tuples for a suite definition.

    Handles the 'extra' key for TYPICAL suite: additional (size, batch)
    pairs beyond the base sizes x batches cross product.
    """
    apis = ["c2c", "z2z", "r2c", "c2r", "d2z", "z2d"]
    seen = set()

    # Base: sizes x batches x apis
    for size in suite["sizes"]:
        for batch in suite["batches"]:
            for api in apis:
                for direction in API_DIRECTIONS[api]:
                    key = (size, batch, api, direction)
                    if key not in seen:
                        seen.add(key)
                        yield key

    # Extra (size, batch) pairs applied to all API/direction combos
    for size, batch in suite.get("extra", []):
        for api in apis:
            for direction in API_DIRECTIONS[api]:
                key = (size, batch, api, direction)
                if key not in seen:
                    seen.add(key)
                    yield key
