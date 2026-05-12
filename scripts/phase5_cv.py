from __future__ import annotations

import argparse
import subprocess
import sys

from _project_paths import project_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5: rigorous repeated stratified CV + Wilcoxon.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny CV job")
    args = parser.parse_args()

    paths = project_paths()
    out_dir = paths.reports_dir / ("hpo_smoke" if args.smoke else "hpo")
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_a = paths.processed_dir / ("ijepa_embeddings_smoke.npz" if args.smoke else "ijepa_embeddings.npz")
    emb_b = paths.processed_dir / ("resnet50_embeddings_smoke.npz" if args.smoke else "resnet50_embeddings.npz")

    cmd = [
        sys.executable,
        str(paths.root / "notebooks" / "04_hpo_and_cv.py"),
        "--mode",
        "cv",
        "--embeddings_a",
        str(emb_a),
        "--embeddings_b",
        str(emb_b),
        "--output_dir",
        str(out_dir),
    ]
    if args.smoke:
        cmd += ["--n_folds", "3", "--n_repeats", "1"]

    print("[phase5]", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
