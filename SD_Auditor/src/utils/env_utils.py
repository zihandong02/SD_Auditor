import os, random, numpy as np
import json
import datetime
from typing import Dict, Any
import pandas as pd

SEED = 2  # default global seed (will be updated by set_global_seed)

def set_global_seed(seed: int | None = None):
    """
    Set all RNG seeds and synchronously update the module-level SEED.

    Parameters
    ----------
    seed : int or None
        New seed value; if None, reuse the current env_utils.SEED.
    """
    global SEED
    if seed is None:
        seed = SEED          # keep existing default
    else:
        SEED = seed          # update the global constant

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Uncomment if using PyTorch
    # try:
    #     import torch
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # except ImportError:
    #     pass

def get_device():
    # Uncomment if using PyTorch
    # try:
    #     import torch
    #     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # except ImportError:
    #     return "cpu"
    
    return "cpu"

# ---------------------------------------------------------------------
def _to_json(obj: Any):
    """
    Fallback function for json.dump.

    Converts numpy scalars / ndarrays into native Python types
    so that the whole parameter dictionary can be written as JSON.
    """
    if isinstance(obj, np.ndarray):                       # array -> list
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int_)):            # np.int64 -> int
        return int(obj)
    if isinstance(obj, (np.floating, np.float_)):         # np.float64 -> float
        return float(obj)
    return str(obj)                                       # anything else -> str


# ---------------------------------------------------------------------
# public helper: save run summary + parameters
# ---------------------------------------------------------------------
def dump_run_simple(
    *,
    df: pd.DataFrame,
    params: Dict[str, Any],
    base_dir: str = "results",
    prefix: str = "",
) -> str:
    """
    Persist one experimental run **without** generating figures.

    Creates a timestamped directory:
        results/<prefix>_<YYYYMMDD_HHMMSS>/

    Inside the folder it writes:
        • summary.csv   – machine-readable table
        • summary.txt   – pretty-printed table
        • params.json   – all hyper-parameters used for the run

    Parameters
    ----------
    df      : pandas.DataFrame
        Aggregated metrics (indexed by τ).
    params  : dict
        Dict holding everything needed to reproduce the run
        (e.g. {"tau_values": [...], "common_args": {...}}).
    base_dir : str, optional
        Top-level directory that will hold all runs.
    prefix   : str, optional
        Prefix for the timestamped subfolder.

    Returns
    -------
    out_dir : str
        Absolute path to the folder that was created.
    """
    # ---------- 1. construct timestamped subfolder -------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"{prefix}{timestamp}")
    os.makedirs(out_dir, exist_ok=True)          # create parent dirs if needed

    # ---------- 2. save summary table --------------------------------
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=True)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f_txt:
        f_txt.write(df.to_string())

    # ---------- 3. save full parameter dictionary --------------------
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f_json:
        json.dump(params, f_json, indent=4, default=_to_json)

    return os.path.abspath(out_dir)