"""Create evidence-preserving, explicitly incomplete Assignment 2 archives."""

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assignment2/packages"


def included_paths(exercise):
    common = [
        "README.md",
        "AGENTS.md",
        "assignment2/config/experiment.yaml",
        "assignment2/requirements.md",
        "assignment2/RUNBOOK.md",
        "assignment2/report_outline.md",
        "assignment2/runner.py",
        "assignment2/budget.py",
        "assignment2/analysis.py",
        "assignment2/run_experiment.py",
        "assignment2/storage.py",
        "assignment2/scripts/inspect_remote.sh",
        "assignment2/scripts/check_python_environments.sh",
        "assignment2/scripts/setup.sbatch",
        "assignment2/scripts/run_pipeline.sbatch",
        "assignment2/scripts/start_vllm.sh",
        "assignment2/scripts/stop_vllm.sh",
        "balrog/config/config.yaml",
        "balrog/agents/__init__.py",
        "balrog/client.py",
    ]
    if exercise >= 2:
        common.append("assignment2/agents.py")
    if exercise == 1:
        common.append("balrog/agents/naive.py")
    return [ROOT / relative_path for relative_path in common]


def add_results(archive):
    results_directory = ROOT / "assignment2/results"
    if not results_directory.exists():
        return []
    added = []
    for path in results_directory.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".md", ".png", ".pdf"}:
            archive.write(path, path.relative_to(ROOT))
            added.append(str(path.relative_to(ROOT)))
    return added


def create_archive(exercise):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"Ass2-{exercise}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        included = []
        for path in included_paths(exercise):
            if path.exists():
                archive.write(path, path.relative_to(ROOT))
                included.append(str(path.relative_to(ROOT)))
        results = add_results(archive)
        status = {
            "status": "NOT SUBMISSION READY",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": (
                "This archive contains implementation scaffolding. Add real completed evidence and report "
                "answers after final evaluation."
            ),
            "included_source_files": included,
            "included_result_files": results,
            "excluded": ["SECRETS", "credentials", "environments", "model weights", "caches", "reference books"],
        }
        archive.writestr("ARCHIVE_STATUS.json", json.dumps(status, indent=2))
    return target


def main():
    for exercise in (1, 2, 3):
        print(create_archive(exercise))


if __name__ == "__main__":
    main()
