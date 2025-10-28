from typing import List, Dict, Any
import os


class CSVLogger:
    """
    Keras equivalent of the PyTorch CSVLogger for tracking training metrics.
    """

    def __init__(self, path: str, header: List[str]):
        self.path = path
        self.header = list(header)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write(",".join(self.header) + "\n")

    def log(self, row: Dict[str, Any]):
        vals = [row.get(k, "") for k in self.header]
        with open(self.path, "a") as f:
            f.write(",".join(str(v) for v in vals) + "\n")


def format_seconds(s: float) -> str:
    """
    Format seconds into human-readable format.
    """
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
