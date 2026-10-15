if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import os
import sys
import json
import time
import tempfile
from typing import Dict, Optional, Any

#
# add src/ to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ablation_config import (
    STATE_FILE, ABLATION_ROOT, RUNS_DIR,
    generate_all_configs, get_all_fold_keys, get_fold_specimens,
)

LOCK_FILE = os.path.join(os.path.dirname(STATE_FILE), ".ablation_state.lock")


def resolve_state_path(path: Optional[str]) -> Optional[str]:
    """Resolve a path from the state file. If relative, resolve relative to the directory of STATE_FILE."""
    if path is not None and not os.path.isabs(path):
        return os.path.normpath(os.path.join(os.path.dirname(STATE_FILE), path))
    return path



# __ file lock (atomic create, NFS-safe) _________________________________

class StateLock:
    """Exclusive lock via atomic file creation."""
    def __init__(self, timeout: float = 30.0):
        os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
        self._timeout = timeout

    def __enter__(self):
        deadline = time.time() + self._timeout
        while True:
            try:
                fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                # stale lock? check age and break if too old.
                try:
                    age = time.time() - os.path.getmtime(LOCK_FILE)
                except OSError:
                    age = 0
                if age > 300:  # stale threshold
                    try:
                        os.remove(LOCK_FILE)
                        print(f"[StateLock] Removed stale lock (age={age:.0f}s)")
                    except OSError:
                        pass
                if time.time() > deadline:
                    raise TimeoutError(f"Could not acquire state lock within {self._timeout}s")
                time.sleep(0.5)
        return self

    def __exit__(self, *_):
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass

# status constants
PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"

# phase keys
PH_PRETRAIN = "pretrain"
PH_FINETUNE = "finetune"
PH_EVAL_PRETRAIN = "eval_pretrain"
PH_EVAL_FINETUNE = "eval_finetune"

ALL_PHASES = [PH_PRETRAIN, PH_FINETUNE, PH_EVAL_PRETRAIN, PH_EVAL_FINETUNE]


def _run_id(config_id: str, fold_key: str) -> str:
    return f"{config_id}__{fold_key}"


def init_state(force: bool = False) -> Dict[str, Any]:
    """Create the state file if it doesn't exist (or if *force* is True)."""
    if os.path.exists(STATE_FILE) and not force:
        return load_state()

    os.makedirs(ABLATION_ROOT, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    configs = generate_all_configs()
    fold_keys = get_all_fold_keys()

    runs: Dict[str, Any] = {}
    for cfg in configs:
        for fk in fold_keys:
            rid = _run_id(cfg["id"], fk)
            runs[rid] = {
                "config_id": cfg["id"],
                "fold_key": fk,
                "eval_specimens": get_fold_specimens(fk),
                "config": cfg,
                "pretrain":        {"status": PENDING, "checkpoint": None, "epochs_done": 0, "started_at": None, "finished_at": None},
                "finetune":        {"status": PENDING, "checkpoint": None, "epochs_done": 0, "started_at": None, "finished_at": None},
                "eval_pretrain":   {"status": PENDING, "results_csv": None, "started_at": None, "finished_at": None},
                "eval_finetune":   {"status": PENDING, "results_csv": None, "started_at": None, "finished_at": None},
            }

    state = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_runs": len(runs),
        "runs": runs,
    }
    save_state(state)
    return state


def load_state() -> Dict[str, Any]:
    """Load the state file, initialising if necessary."""
    if not os.path.exists(STATE_FILE):
        return init_state()
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    """Atomically write the state file (write-then-rename)."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(STATE_FILE), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_FILE)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _set_phase_status(state: Dict, rid: str, phase: str, status: str, **extra) -> None:
    """Update a single phase of a single run."""
    run = state["runs"][rid]
    run[phase]["status"] = status
    run[phase].update(extra)
    if status == IN_PROGRESS and run[phase].get("started_at") is None:
        run[phase]["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if status == COMPLETED:
        run[phase]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_state(state)


def mark_pretrain_started(state: Dict, rid: str) -> None:
    _set_phase_status(state, rid, PH_PRETRAIN, IN_PROGRESS)


def mark_pretrain_completed(state: Dict, rid: str, checkpoint: str) -> None:
    _set_phase_status(state, rid, PH_PRETRAIN, COMPLETED, checkpoint=checkpoint)


def mark_finetune_started(state: Dict, rid: str) -> None:
    _set_phase_status(state, rid, PH_FINETUNE, IN_PROGRESS)


def mark_finetune_completed(state: Dict, rid: str, checkpoint: str) -> None:
    _set_phase_status(state, rid, PH_FINETUNE, COMPLETED, checkpoint=checkpoint)


def mark_eval_started(state: Dict, rid: str, phase: str) -> None:
    assert phase in (PH_EVAL_PRETRAIN, PH_EVAL_FINETUNE)
    _set_phase_status(state, rid, phase, IN_PROGRESS)


def mark_eval_completed(state: Dict, rid: str, phase: str, results_csv: str) -> None:
    assert phase in (PH_EVAL_PRETRAIN, PH_EVAL_FINETUNE)
    _set_phase_status(state, rid, phase, COMPLETED, results_csv=results_csv)


# __ task selection _______________________________________________________

def get_next_pending_pretrain(state: Dict) -> Optional[Dict]:
    """Find the next run whose pretrain is pending (not in-progress)."""
    for rid, run in state["runs"].items():
        if run["pretrain"]["status"] == PENDING:
            return {"rid": rid, **run}
    return None


def get_next_pending_finetune(state: Dict) -> Optional[Dict]:
    """Find the next run whose pretrain is done but finetune is pending."""
    for rid, run in state["runs"].items():
        if run["pretrain"]["status"] == COMPLETED and run["finetune"]["status"] == PENDING:
            return {"rid": rid, **run}
    return None


def get_next_pending_eval(state: Dict) -> Optional[Dict]:
    """Find the next run needing evaluation (pretrain or finetune eval)."""
    # eval_pretrain: pretrain done, eval_pretrain pending
    for rid, run in state["runs"].items():
        if run["pretrain"]["status"] == COMPLETED and run["eval_pretrain"]["status"] == PENDING:
            return {"rid": rid, "phase": PH_EVAL_PRETRAIN, **run}
    # eval_finetune: finetune done, eval_finetune pending
    for rid, run in state["runs"].items():
        if run["finetune"]["status"] == COMPLETED and run["eval_finetune"]["status"] == PENDING:
            return {"rid": rid, "phase": PH_EVAL_FINETUNE, **run}
    return None


def get_progress_summary(state: Dict) -> Dict[str, int]:
    counts = {ph: {"pending": 0, "in_progress": 0, "completed": 0} for ph in ALL_PHASES}
    for run in state["runs"].values():
        for ph in ALL_PHASES:
            counts[ph][run[ph]["status"]] += 1
    return counts


def print_progress(state: Dict) -> None:
    counts = get_progress_summary(state)
    print("\n=== Ablation Pipeline Progress ===")
    for ph in ALL_PHASES:
        c = counts[ph]
        total = c["pending"] + c["in_progress"] + c["completed"]
        print(f"  {ph:20s}  {c['completed']}/{total} done  ({c['in_progress']} in-progress, {c['pending']} pending)")
    print()


# __ atomic claim functions _______________________________________________

def claim_next_pretrain() -> Optional[Dict]:
    """Atomically find and claim the next pending pretrain task."""
    with StateLock():
        state = load_state()
        task = get_next_pending_pretrain(state)
        if task is None:
            return None
        rid = task["rid"]
        mark_pretrain_started(state, rid)
        return task


def claim_next_finetune() -> Optional[Dict]:
    """Atomically find and claim the next pending finetune task."""
    with StateLock():
        state = load_state()
        task = get_next_pending_finetune(state)
        if task is None:
            return None
        rid = task["rid"]
        mark_finetune_started(state, rid)
        return task


def claim_next_eval() -> Optional[Dict]:
    """Atomically find and claim the next pending eval task."""
    with StateLock():
        state = load_state()
        task = get_next_pending_eval(state)
        if task is None:
            return None
        rid = task["rid"]
        phase = task["phase"]
        if state["runs"][rid][phase]["status"] == PENDING:
            mark_eval_started(state, rid, phase)
        return task
