import torch


def is_mps_available() -> bool:
    """Return True if the Metal backend is built and available."""
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    try:
        return mps_backend.is_available()
    except AttributeError:
        return False


def pick_device(preferred: str | torch.device | None = None) -> torch.device:
    """Pick an appropriate torch.device respecting availability."""
    if preferred is not None:
        requested = str(preferred).strip().lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        if requested == "mps" and is_mps_available():
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_supports_fp16(device: torch.device) -> bool:
    return device.type == "cuda"


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and is_mps_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


__all__ = ["is_mps_available", "pick_device", "device_supports_fp16", "empty_cache"]
