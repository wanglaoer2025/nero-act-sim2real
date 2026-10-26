"""Repeat the exact ACT evaluation with complete actual-source before/after binding."""
from pathlib import Path

import torch

import evaluate_adaptation as evaluator
from evaluation_binding import actual_sources,bind_snapshot


if __name__=='__main__':
    torch.set_num_threads(2)
    original=evaluator.historical.snapshot
    dependencies=actual_sources(evaluator)
    dependencies.append(Path(__file__).resolve())
    evaluator.historical.snapshot=bind_snapshot(original,dependencies)
    evaluator.evaluate_dense.ROOT=evaluator.ROOT
    evaluator.evaluate_dense.evaluation.main=evaluator.evaluate
    try:
        evaluator.evaluate_dense.main()
    finally:
        evaluator.historical.snapshot=original
