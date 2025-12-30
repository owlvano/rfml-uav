def get_segment_count(bui: str):
    if bui == "00000":
        return 41  # Number of segments for RF background activities
    elif bui == "10111":
        return 18
    else:
        return 21  # Number of segments for drones RF activities

def get_binary_label(bui: str) -> int:
    """
    0 -> no-signal (background)
    1 -> signal present (any drone)
    """
    return 0 if bui == "00000" else 1


def get_class_label(bui: str):
    match bui:
        case "00000":
            return 0
        case "10000" | "10001" | "10010" | "10011":
            return 1
        case "10100" | "10101" | "10110" | "10111":
            return 2
        case "11000":
            return 3
        case _:
            raise ValueError("unknown BUI value")


def get_input_dim(feature: str) -> int:
    match feature:
        case "PSD":
            return 2048
        case "STFT":
            return 100
        case "Wavelets":
            return 256
        case _:
            raise ValueError("unknown feature vector")
