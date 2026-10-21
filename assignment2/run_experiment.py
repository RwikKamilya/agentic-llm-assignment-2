"""Command-line entry point for the fixed Assignment 2 experiment manifest."""

import argparse
import os
import signal
from pathlib import Path

from omegaconf import OmegaConf

from assignment2.runner import ExperimentRunner
from assignment2.validation import IncompletePhaseError


def stop_requested(signum, frame):
    # Let the runner save the partial episode and return a failure exit.
    raise InterruptedError("Termination requested; saving incomplete evidence.")


def main():
    parser = argparse.ArgumentParser(description="Run one fixed phase from assignment2/config/experiment.yaml.")
    parser.add_argument("--phase", required=True, choices=["smoke", "development", "pilot", "final"])
    parser.add_argument("--resume", default=None, help="Existing run directory to resume.")
    parser.add_argument(
        "--run-directory",
        default=None,
        help="New persistent result directory. It must not already exist and cannot be combined with --resume.",
    )
    parser.add_argument("--models", nargs=1, required=True, help="One named profile; launch Qwen independently of OpenAI.")
    arguments = parser.parse_args()
    if arguments.resume and arguments.run_directory:
        parser.error("--resume and --run-directory cannot be used together.")

    repository_root = Path(__file__).resolve().parents[1]
    study_config = OmegaConf.load(repository_root / "assignment2/config/experiment.yaml")
    output_root = os.environ.get("ASSIGNMENT2_OUTPUT_ROOT")
    if output_root:
        study_config.output_root = output_root
    runner = ExperimentRunner(
        repository_root=repository_root,
        study_config=study_config,
        phase=arguments.phase,
        resume_directory=arguments.resume,
        run_directory=arguments.run_directory,
        model_names=arguments.models,
    )
    signal.signal(signal.SIGTERM, stop_requested)
    try:
        runner.run()
    except IncompletePhaseError as error:
        print(f"Phase incomplete: {error}", flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
