# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic traffic simulation helpers used by the OpenEnv server."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, Iterable, List, Tuple

try:
    from .models import PHASE_NAMES
except ImportError:
    from models import PHASE_NAMES

DIRECTION_NAMES = ("north", "east", "south", "west")
AXIS_TO_LANES = {0: (0, 2), 1: (1, 3)}


@dataclass(frozen=True)
class DemandSegment:
    start_step: int
    end_step: int
    multipliers: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    name: str
    rows: int
    cols: int
    max_steps: int
    min_green_steps: int
    lane_capacity: int
    switch_penalty: int
    base_arrival_rates: Tuple[float, float, float, float]
    segments: Tuple[DemandSegment, ...]
    emergency_schedule: Tuple[int, ...]
    score_scale: float
    throughput_target: int
    emergency_wait_target: float


SCENARIOS: Dict[str, ScenarioConfig] = {
    "task_easy": ScenarioConfig(
        scenario_id="task_easy",
        name="Single Intersection Steady Flow",
        rows=1,
        cols=1,
        max_steps=36,
        min_green_steps=2,
        lane_capacity=3,
        switch_penalty=1,
        base_arrival_rates=(1.2, 0.7, 1.1, 0.8),
        segments=(
            DemandSegment(0, 11, (1.0, 0.9, 1.0, 0.9)),
            DemandSegment(12, 23, (1.2, 0.8, 1.2, 0.8)),
            DemandSegment(24, 35, (0.95, 1.05, 0.95, 1.05)),
        ),
        emergency_schedule=(),
        score_scale=18.0,
        throughput_target=170,
        emergency_wait_target=1.0,
    ),
    "task_medium": ScenarioConfig(
        scenario_id="task_medium",
        name="Two Intersections Corridor",
        rows=1,
        cols=2,
        max_steps=48,
        min_green_steps=3,
        lane_capacity=3,
        switch_penalty=1,
        base_arrival_rates=(1.0, 1.3, 1.0, 1.3),
        segments=(
            DemandSegment(0, 15, (0.9, 1.0, 0.9, 1.0)),
            DemandSegment(16, 31, (1.0, 1.45, 1.0, 1.45)),
            DemandSegment(32, 47, (1.2, 1.1, 1.2, 1.1)),
        ),
        emergency_schedule=(),
        score_scale=70.0,
        throughput_target=280,
        emergency_wait_target=1.0,
    ),
    "task_hard": ScenarioConfig(
        scenario_id="task_hard",
        name="2x2 Grid With Emergency Preemption",
        rows=2,
        cols=2,
        max_steps=60,
        min_green_steps=3,
        lane_capacity=3,
        switch_penalty=1,
        base_arrival_rates=(1.2, 1.2, 1.1, 1.1),
        segments=(
            DemandSegment(0, 19, (1.0, 1.1, 1.0, 1.1)),
            DemandSegment(20, 39, (1.35, 1.15, 1.35, 1.15)),
            DemandSegment(40, 59, (1.1, 1.3, 1.1, 1.3)),
        ),
        emergency_schedule=(8, 19, 31, 44, 53),
        score_scale=140.0,
        throughput_target=430,
        emergency_wait_target=14.0,
    ),
}


def build_intersection_ids(rows: int, cols: int) -> List[str]:
    return [f"r{row}_c{col}" for row in range(rows) for col in range(cols)]


def phase_one_hot(phase_index: int) -> List[int]:
    return [1 if phase_index == index else 0 for index in range(len(PHASE_NAMES))]


def intersection_position(index: int, cols: int) -> Tuple[int, int]:
    return divmod(index, cols)


def downstream_index(
    intersection_index: int,
    lane_index: int,
    rows: int,
    cols: int,
) -> int | None:
    row, col = intersection_position(intersection_index, cols)
    if lane_index == 0 and row + 1 < rows:
        return (row + 1) * cols + col
    if lane_index == 1 and col - 1 >= 0:
        return row * cols + (col - 1)
    if lane_index == 2 and row - 1 >= 0:
        return (row - 1) * cols + col
    if lane_index == 3 and col + 1 < cols:
        return row * cols + (col + 1)
    return None


def demand_multiplier(config: ScenarioConfig, step_index: int, lane_index: int) -> float:
    for segment in config.segments:
        if segment.start_step <= step_index <= segment.end_step:
            return segment.multipliers[lane_index]
    return 1.0


def inbound_rate(
    config: ScenarioConfig,
    intersection_index: int,
    lane_index: int,
    step_index: int,
) -> float:
    row, col = intersection_position(intersection_index, config.cols)
    boundary_bonus = (
        1.18
        if downstream_index(intersection_index, lane_index, config.rows, config.cols)
        is None
        else 0.62
    )
    spatial_bias = 1.0 + 0.06 * row + 0.05 * col
    if config.scenario_id == "task_medium" and lane_index in (1, 3):
        spatial_bias += 0.15 if col == 0 else 0.05
    if config.scenario_id == "task_hard":
        spatial_bias += 0.05 * ((row + col) % 2)
    return (
        config.base_arrival_rates[lane_index]
        * demand_multiplier(config, step_index, lane_index)
        * boundary_bonus
        * spatial_bias
    )


def sample_arrivals(rate: float, rng: Random) -> int:
    base = int(rate)
    fractional = rate - base
    return base + (1 if rng.random() < fractional else 0)


def choose_override_axis(emergency_lanes: List[int], regular_lanes: List[int]) -> int:
    ns_pressure = emergency_lanes[0] * 10 + emergency_lanes[2] * 10 + regular_lanes[0] + regular_lanes[2]
    ew_pressure = emergency_lanes[1] * 10 + emergency_lanes[3] * 10 + regular_lanes[1] + regular_lanes[3]
    return 0 if ns_pressure >= ew_pressure else 1


def network_pressure(
    queue_lengths: Iterable[Iterable[int]],
    emergency_lengths: Iterable[Iterable[int]],
) -> float:
    pressure = 0.0
    for queues, emergencies in zip(queue_lengths, emergency_lengths):
        ns = queues[0] + queues[2] + 4 * (emergencies[0] + emergencies[2])
        ew = queues[1] + queues[3] + 4 * (emergencies[1] + emergencies[3])
        pressure += abs(ns - ew) + ns + ew
    return pressure


def score_from_metrics(config: ScenarioConfig, metrics: Dict[str, float]) -> float:
    avg_queue = metrics["average_queue"]
    queue_score = max(0.0, 1.0 - avg_queue / config.score_scale)
    throughput_score = min(1.0, metrics["throughput"] / config.throughput_target)
    if config.emergency_schedule:
        emergency_score = max(
            0.0,
            1.0 - metrics["average_emergency_wait"] / config.emergency_wait_target,
        )
        clearance_score = (
            metrics["emergency_vehicles_cleared"] / metrics["emergency_vehicles_spawned"]
            if metrics["emergency_vehicles_spawned"] > 0
            else 1.0
        )
        score = (
            0.4 * queue_score
            + 0.2 * throughput_score
            + 0.25 * emergency_score
            + 0.15 * clearance_score
        )
    else:
        score = 0.7 * queue_score + 0.3 * throughput_score
    return round(max(0.0, min(1.0, score)), 4)
