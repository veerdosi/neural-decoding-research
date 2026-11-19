"""Portable locations; frozen experiment metadata is intentionally not rewritten."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "research/phase3-sentence-reconstruction"
WORK = PROJECT / "local"
PUBLICATION = PROJECT / "publication"
PYTHON = PROJECT / ".venv/bin/python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def resolve_sandbox(original):
    """Read relocated completed traces without changing historical case hashes."""
    original = Path(original)
    archived = ROOT / "agentic_traces" / original.name
    if archived.is_dir():
        return archived
    if original.name == "agentic_neural_sandbox":
        return ROOT / "agentic_development_trace"
    if original.exists():
        return original
    # New independent cases belong in local/, never in the frozen archive.
    return WORK / "agentic_cases" / original.name


def resolve_locked(original):
    original = Path(original)
    marker = "outputs/neural_language_reconstruction/"
    relative = str(original).split(marker, 1)[-1]
    if relative.startswith("code/"):
        frozen = ROOT / "code/frozen" / original.name
        if frozen.exists():
            return frozen
    return ROOT / relative
