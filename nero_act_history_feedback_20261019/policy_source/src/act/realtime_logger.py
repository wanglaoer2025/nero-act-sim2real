"""CSV/JSON logger for NERO ACT real-time dry-runs and rollouts."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


class RealtimeLogger:
    FIELDS = [
        "timestamp", "step", "current_qpos", "pred_action_raw", "pred_action_denorm",
        "final_action_command", "actual_qpos", "gripper_prediction", "gripper_state",
        "qpos", "raw_pred", "first_action", "aggregated_pred", "safe_target",
        "mpc_target", "mpc_active", "mpc_blend", "mpc_correction_norm_rad",
        "mpc_fallback", "mpc_solve_latency_s", "mpc_tcp_distance_mm",
        "mpc_predicted_terminal_error_mm", "mpc_response_alpha",
        "joint_limit_clamps", "delta_clamps", "command_lead_clamps", "range_anomalies",
        "gripper_command_m", "grasp_gate_blocked", "grasp_tcp_distance_mm",
        "estimated_tcp_xyz_mm", "grasp_distance_mm", "grasp_tcp_z_offset_mm",
        "grasp_table_clearance_mm",
        "grasp_height_gate_blocked",
        "command_sent", "camera_latency_s", "qpos_latency_s", "preprocess_latency_s",
        "model_latency_s", "denorm_latency_s", "safety_latency_s", "send_latency_s",
        "total_loop_dt_s", "processing_latency_s", "late_cycle",
    ]

    def __init__(self, output_dir: Path, mode: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"act_day6_{mode}_{datetime.now():%Y%m%d_%H%M%S}"
        self.csv_path = output_dir / f"{stem}.csv"
        self.summary_path = output_dir / f"{stem}_summary.json"
        self.stream = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.rows = []

    def log(self, row: dict) -> None:
        serialized = dict(row)
        for key in (
            "current_qpos", "pred_action_raw", "pred_action_denorm",
            "final_action_command", "actual_qpos", "qpos", "raw_pred",
            "first_action", "aggregated_pred", "safe_target",
            "estimated_tcp_xyz_mm", "mpc_target", "mpc_response_alpha",
        ):
            serialized[key] = json.dumps(np.asarray(serialized[key]).tolist())
        self.writer.writerow(serialized)
        self.stream.flush()
        self.rows.append(row)

    def close(self, extra: dict | None = None) -> dict:
        if self.stream.closed:
            return {}
        self.stream.close()
        timings = {}
        for key in ("camera_latency_s", "qpos_latency_s", "preprocess_latency_s", "model_latency_s",
                    "denorm_latency_s", "safety_latency_s", "send_latency_s", "total_loop_dt_s",
                    "processing_latency_s", "mpc_solve_latency_s"):
            values = np.asarray([row[key] for row in self.rows], dtype=np.float64)
            timings[key] = {
                "mean": float(values.mean()) if len(values) else 0.0,
                "p50": float(np.percentile(values, 50)) if len(values) else 0.0,
                "p95": float(np.percentile(values, 95)) if len(values) else 0.0,
                "max": float(values.max()) if len(values) else 0.0,
            }
        elapsed = sum(row["total_loop_dt_s"] for row in self.rows[1:])
        grasp_distances = [
            row["grasp_tcp_distance_mm"] for row in self.rows
            if row.get("grasp_tcp_distance_mm") is not None
        ]
        grasp_z_offsets = [
            row["grasp_tcp_z_offset_mm"] for row in self.rows
            if row.get("grasp_tcp_z_offset_mm") is not None
        ]
        grasp_table_clearances = [
            row["grasp_table_clearance_mm"] for row in self.rows
            if row.get("grasp_table_clearance_mm") is not None
        ]
        summary = {
            "steps": len(self.rows), "timing": timings,
            "actual_frequency_hz": (
                (len(self.rows) - 1) / elapsed if len(self.rows) > 1 and elapsed > 0 else 0.0
            ),
            "joint_limit_clamps": sum(row["joint_limit_clamps"] for row in self.rows),
            "delta_clamps": sum(row["delta_clamps"] for row in self.rows),
            "command_lead_clamps": sum(row["command_lead_clamps"] for row in self.rows),
            "range_anomalies": sum(row["range_anomalies"] for row in self.rows),
            "commands_sent": sum(row.get("command_sent", 0) for row in self.rows),
            "mpc_active_steps": sum(row.get("mpc_active", 0) for row in self.rows),
            "mpc_max_correction_norm_rad": max(
                (row.get("mpc_correction_norm_rad", 0.0) for row in self.rows),
                default=0.0,
            ),
            "grasp_gate_blocks": sum(row.get("grasp_gate_blocked", 0) for row in self.rows),
            "grasp_gate_min_attempt_distance_mm": (
                float(min(grasp_distances)) if grasp_distances else None
            ),
            "grasp_height_gate_blocks": sum(
                row.get("grasp_height_gate_blocked", 0) for row in self.rows
            ),
            "grasp_min_z_offset_mm": (
                float(min(grasp_z_offsets)) if grasp_z_offsets else None
            ),
            "grasp_min_table_clearance_mm": (
                float(min(grasp_table_clearances)) if grasp_table_clearances else None
            ),
            "late_cycles": sum(row["late_cycle"] for row in self.rows),
            "nan_rejections": 0,
            "feedback_failures": 0,
            "no_progress_rejections": 0,
            **(extra or {}),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
