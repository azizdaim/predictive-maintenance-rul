"""Which engine section each active C-MAPSS sensor measures.

One shared mapping, used by the ablation study and the dashboard's 3D engine,
so the two can never disagree. Source: Saxena et al. (2008), the paper that
introduced the C-MAPSS dataset.
"""

# Short key (used by the 3D engine) -> (display name, sensor numbers)
ENGINE_SECTIONS: dict[str, tuple[str, list[int]]] = {
    "fan": ("Fan", [6, 8, 13, 15]),
    "lpc": ("Low-pressure compressor", [2]),
    "hpc": ("High-pressure compressor", [3, 7, 9, 11, 14]),
    "combustor": ("Combustor", [12, 17]),
    "hpt": ("High-pressure turbine", [20]),
    "lpt": ("Low-pressure turbine", [4, 21]),
}
