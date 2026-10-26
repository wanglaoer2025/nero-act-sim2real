"""Exponential temporal ensemble for overlapping ACT action chunks."""

from __future__ import annotations

import math

import numpy as np


class TemporalEnsembler:
    def __init__(self, chunk_size: int, action_dim: int, decay: float):
        if chunk_size <= 0 or action_dim <= 0 or decay < 0:
            raise ValueError("chunk/action dimension 必须为正，decay 不得为负")
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.decay = float(decay)
        self.history: list[tuple[int, np.ndarray]] = []

    def add(self, timestep: int, chunk) -> None:
        values = np.asarray(chunk, dtype=np.float32)
        if timestep < 0 or values.shape != (self.chunk_size, self.action_dim) or not np.isfinite(values).all():
            raise ValueError("temporal chunk 必须是有限 [chunk_size, action_dim]")
        self.history.append((timestep, values.copy()))
        self.history = [(start, actions) for start, actions in self.history
                        if start + self.chunk_size > timestep]

    def current(self, timestep: int) -> np.ndarray:
        candidates, weights = [], []
        for start, chunk in self.history:
            offset = timestep - start
            if 0 <= offset < self.chunk_size:
                candidates.append(chunk[offset])
                weights.append(math.exp(-self.decay * offset))
        if not candidates:
            raise RuntimeError(f"timestep {timestep} 没有可聚合的 ACT prediction")
        return np.average(np.stack(candidates), axis=0, weights=np.asarray(weights)).astype(np.float32)


def compare_first_vs_temporal(prediction_chunks, targets, episode_lengths, decay: float) -> dict:
    """Compare current-step predictions without mixing episode boundaries."""
    chunks = np.asarray(prediction_chunks, dtype=np.float32)
    truth = np.asarray(targets, dtype=np.float32)
    if chunks.ndim != 3 or truth.shape != (len(chunks), chunks.shape[2]):
        raise ValueError("prediction chunks/targets shape 不匹配")
    if sum(episode_lengths) != len(chunks):
        raise ValueError("episode lengths 与 prediction 数量不匹配")
    first, aggregated, cursor = [], [], 0
    first_jitter, aggregated_jitter = [], []
    for length in episode_lengths:
        ensembler = TemporalEnsembler(chunks.shape[1], chunks.shape[2], decay)
        episode_first, episode_aggregated = [], []
        for timestep, chunk in enumerate(chunks[cursor:cursor + length]):
            ensembler.add(timestep, chunk)
            episode_first.append(chunk[0])
            episode_aggregated.append(ensembler.current(timestep))
        first.extend(episode_first)
        aggregated.extend(episode_aggregated)
        if length > 1:
            first_jitter.extend(np.linalg.norm(np.diff(np.asarray(episode_first)[:, :7], axis=0), axis=1))
            aggregated_jitter.extend(np.linalg.norm(np.diff(np.asarray(episode_aggregated)[:, :7], axis=0), axis=1))
        cursor += length
    first, aggregated = np.asarray(first), np.asarray(aggregated)

    def metrics(values, jitter):
        jitter = np.asarray(jitter)
        return {
            "mae_real": float(np.abs(values - truth).mean()),
            "arm_jitter_mean_rad": float(jitter.mean()) if len(jitter) else 0.0,
            "arm_jitter_p95_rad": float(np.percentile(jitter, 95)) if len(jitter) else 0.0,
            "arm_jitter_max_rad": float(jitter.max()) if len(jitter) else 0.0,
            "gripper_binary_accuracy": float(np.mean((values[:, 7] >= 0.0475) == (truth[:, 7] >= 0.0475))),
        }

    return {"first_action": metrics(first, first_jitter), "temporal": metrics(aggregated, aggregated_jitter)}
