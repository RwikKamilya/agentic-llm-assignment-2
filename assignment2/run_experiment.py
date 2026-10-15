"""Command-line entry point for the fixed Assignment 2 experiment manifest."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from assignment2.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run one fixed phase from assignment2/config/experiment.yaml.")
    parser.add_argument("--phase", required=True, choices=["smoke", "development", "pilot", "final"])
    parser.add_argument("--resume", default=None, help="Existing run directory to resume.")
    parser.add_argument("--models", nargs="+", default=None, help="Optional named model profiles from the manifest.")
    arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    study_config = OmegaConf.load(repository_root / "assignment2/config/experiment.yaml")
    runner = ExperimentRunner(
        repository_root=repository_root,
        study_config=study_config,
        phase=arguments.phase,
        resume_directory=arguments.resume,
        model_names=arguments.models,
    )
    runner.run()


if __name__ == "__main__":
    main()
