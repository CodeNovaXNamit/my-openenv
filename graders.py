# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic grading helpers for the built-in traffic tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

try:
    from .baseline_agent import heuristic_policy
    from .models import SmartTrafficAction, SmartTrafficObservation
    from .server.smart_traffic_env_environment import SmartTrafficEnvironment
    from .tasks import TASKS, TaskDefinition
except ImportError:
    from baseline_agent import heuristic_policy
    from models import SmartTrafficAction, SmartTrafficObservation
    from server.smart_traffic_env_environment import SmartTrafficEnvironment
    from tasks import TASKS, TaskDefinition

Policy = Callable[[SmartTrafficObservation], SmartTrafficAction]


@dataclass(frozen=True)
class GradeResult:
    task_id: str
    score: float
    reward_sum: float
    metrics: Dict[str, float]


def run_task(task: TaskDefinition, policy: Policy) -> GradeResult:
    """Run a full deterministic episode for one task."""

    env = SmartTrafficEnvironment()
    observation = env.reset(seed=task.seed, scenario_id=task.scenario_id)
    reward_sum = 0.0

    while not observation.done:
        action = policy(observation)
        observation = env.step(action)
        reward_sum += float(observation.reward or 0.0)

    metrics = observation.metadata["metrics"]
    return GradeResult(
        task_id=task.task_id,
        score=float(metrics["score"]),
        reward_sum=round(reward_sum, 4),
        metrics=metrics,
    )


def grade_all(policy: Policy = heuristic_policy) -> Dict[str, GradeResult]:
    """Run the same policy across all three tasks."""

    return {
        task_id: run_task(task_definition, policy)
        for task_id, task_definition in TASKS.items()
    }


def grader(task_id: str, policy: Policy = heuristic_policy) -> float:
    """Compatibility wrapper returning the normalized 0-1 score only."""

    return run_task(TASKS[task_id], policy).score
