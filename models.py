# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the smart traffic control environment."""

from typing import Dict, List, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator

LANE_NAMES = ("north", "east", "south", "west")
PHASE_NAMES = ("north_south_green", "east_west_green", "emergency_override")
SCENARIO_IDS = ("task_easy", "task_medium", "task_hard")


class SmartTrafficAction(Action):
    """Discrete phase choice for each controlled intersection."""

    phase_indices: List[int] = Field(
        ...,
        min_length=1,
        description=(
            "One discrete action per intersection. "
            "0=north_south_green, 1=east_west_green, 2=emergency_override."
        ),
    )

    @field_validator("phase_indices")
    @classmethod
    def validate_phase_indices(cls, value: List[int]) -> List[int]:
        if any(phase not in (0, 1, 2) for phase in value):
            raise ValueError("phase_indices entries must be 0, 1, or 2")
        return value


class SmartTrafficObservation(Observation):
    """Typed observation returned by reset and step."""

    scenario_id: Literal["task_easy", "task_medium", "task_hard"] = Field(
        ..., description="Scenario identifier for the active task."
    )
    step_index: int = Field(..., ge=0, description="Current decision step.")
    max_steps: int = Field(..., ge=1, description="Episode horizon.")
    grid_rows: int = Field(..., ge=1, description="Number of grid rows.")
    grid_cols: int = Field(..., ge=1, description="Number of grid columns.")
    intersection_ids: List[str] = Field(
        ..., description="Stable identifiers for each controlled intersection."
    )
    phase_names: List[str] = Field(
        default_factory=lambda: list(PHASE_NAMES),
        description="Human-readable phase names.",
    )
    current_phase_indices: List[int] = Field(
        ..., description="Current selected command for each intersection."
    )
    served_axis_indices: List[int] = Field(
        ...,
        description="Operational axis served after min-green and emergency logic. 0=NS, 1=EW.",
    )
    phase_one_hot: List[List[int]] = Field(
        ..., description="One-hot encoded current phase for each intersection."
    )
    min_green_satisfied: List[bool] = Field(
        ..., description="Whether each intersection may switch without override."
    )
    queue_lengths: List[List[int]] = Field(
        ...,
        description="Regular-vehicle queue lengths per intersection and lane [N, E, S, W].",
    )
    emergency_queue_lengths: List[List[int]] = Field(
        ...,
        description="Emergency-vehicle queue lengths per intersection and lane [N, E, S, W].",
    )
    inbound_demand: List[List[float]] = Field(
        ...,
        description="Deterministic demand rates used for the next arrival generation.",
    )
    total_queue_length: int = Field(..., ge=0, description="Total regular queue length.")
    total_emergency_queue: int = Field(
        ..., ge=0, description="Total emergency queue length."
    )
    cumulative_wait: float = Field(
        ..., ge=0.0, description="Accumulated waiting-time proxy over the episode."
    )
    cumulative_emergency_wait: float = Field(
        ..., ge=0.0, description="Accumulated waiting-time proxy for emergency vehicles."
    )
    throughput: int = Field(..., ge=0, description="Vehicles that exited the network.")
    emergency_vehicles_spawned: int = Field(
        ..., ge=0, description="Emergency vehicles injected so far."
    )
    emergency_vehicles_cleared: int = Field(
        ..., ge=0, description="Emergency vehicles that exited the network."
    )
    network_pressure: float = Field(
        ..., description="Aggregate pressure signal used by the heuristic baseline."
    )
    score_hint: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized hint derived from current episode metrics.",
    )


class SmartTrafficState(State):
    """Internal state snapshot exposed through the OpenEnv state endpoint."""

    scenario_id: str = Field(..., description="Active scenario identifier.")
    seed: int = Field(..., description="Seed used for deterministic dynamics.")
    max_steps: int = Field(..., ge=1, description="Episode horizon.")
    grid_rows: int = Field(..., ge=1)
    grid_cols: int = Field(..., ge=1)
    intersection_ids: List[str] = Field(default_factory=list)
    current_phase_indices: List[int] = Field(default_factory=list)
    served_axis_indices: List[int] = Field(default_factory=list)
    steps_since_change: List[int] = Field(default_factory=list)
    queue_lengths: List[List[int]] = Field(default_factory=list)
    emergency_queue_lengths: List[List[int]] = Field(default_factory=list)
    throughput: int = Field(default=0, ge=0)
    emergency_vehicles_spawned: int = Field(default=0, ge=0)
    emergency_vehicles_cleared: int = Field(default=0, ge=0)
    cumulative_wait: float = Field(default=0.0, ge=0.0)
    cumulative_emergency_wait: float = Field(default=0.0, ge=0.0)
    last_reward: float = Field(default=0.0)
    metrics: Dict[str, float] = Field(default_factory=dict)
