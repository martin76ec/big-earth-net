from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"


def project_paths() -> ProjectPaths:
    # scripts/ lives at <root>/scripts/
    root = Path(__file__).resolve().parents[1]
    return ProjectPaths(root=root)


def _pretty_list(paths: list[Path]) -> str:
    return "\n".join(f"- {p}" for p in paths)


def find_single(glob_pattern: str, *, search_dir: Path, hint: str) -> Path:
    matches = sorted(search_dir.rglob(glob_pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {hint} under {search_dir} (pattern: {glob_pattern})."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple candidates for {hint} under {search_dir} (pattern: {glob_pattern}):\n"
            + _pretty_list(matches)
        )
    return matches[0]


def find_best_tar_zst(data_dir: Path) -> Path:
    # Prefer canonical filename if present.
    exact = sorted(data_dir.rglob("BigEarthNet-S2.tar.zst"))
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError(
            "Found multiple 'BigEarthNet-S2.tar.zst' files under data/:\n" + _pretty_list(exact)
        )

    # Fallback to any tar.zst in data/.
    matches = sorted(data_dir.rglob("*.tar.zst"))
    if not matches:
        raise FileNotFoundError(
            "Could not find a .tar.zst archive under data/. Put the dataset archive under data/ (any subfolder)."
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Found multiple .tar.zst archives under data/. Please keep only one, or pass an explicit --tar_path.\n"
            + _pretty_list(matches)
        )
    return matches[0]


def find_processed_dataset_dir(processed_dir: Path) -> Path:
    # Identify a processed dataset directory by the presence of metadata.json.
    meta = sorted(processed_dir.rglob("metadata.json"))
    if not meta:
        raise FileNotFoundError(
            f"Could not find any processed dataset metadata.json under {processed_dir}. Run phase0 first."
        )
    if len(meta) > 1:
        raise RuntimeError(
            f"Found multiple processed datasets (multiple metadata.json) under {processed_dir}.\n"
            "Please keep only one, or pass an explicit --data_dir to the script.\n"
            + _pretty_list(meta)
        )
    return meta[0].parent


def find_checkpoint(root: Path) -> Path:
    # Prefer the canonical path used by training.
    preferred = root / "checkpoints" / "ijepa" / "ijepa-best.ckpt"
    if preferred.exists():
        return preferred

    # Fallback to a unique ckpt under checkpoints/ijepa.
    ckpt_dir = root / "checkpoints" / "ijepa"
    if ckpt_dir.exists():
        matches = sorted(ckpt_dir.rglob("*.ckpt"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                "Found multiple checkpoints under checkpoints/ijepa. Please keep one, or pass an explicit --checkpoint.\n"
                + _pretty_list(matches)
            )

    raise FileNotFoundError(
        "Could not find an I-JEPA checkpoint. Expected checkpoints/ijepa/ijepa-best.ckpt (run phase1)."
    )
