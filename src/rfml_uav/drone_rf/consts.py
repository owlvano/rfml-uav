# Parameters
BUI_dict = {
    "Background": ["00000"],  # BUI of RF background activities
    "Bebop": [
        "10000",
        "10001",
        "10010",
        "10011",
    ],  # BUI of the Bebop drone RF activities
    "AR": ["10100", "10101", "10110", "10111"],  # BUI of the AR drone RF activities
    "Phantom": ["11000"],  # BUI of the Phantom drone RF activities
}

BUI = [bui for group in BUI_dict.values() for bui in group]

BUI_name = {
    bui: name
    for name, buis in BUI_dict.items()
    for bui in buis
}

def get_bui_name(bui: str) -> str:
    return BUI_name.get(bui, "Unknown")

M = 2048  # Total number of frequency bins
Q = 10  # Number of returning points for spectral continuity

FS_MHZ = 80  # Sampling frequency in MHz
FS_HZ = FS_MHZ * 1e6
