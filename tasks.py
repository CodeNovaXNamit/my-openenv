# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task definitions used for grading and baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    description: str
    scenario_id: str
    seed: int


TASKS: Dict[str, TaskDefinition] = {
    "task_easy": TaskDefinition(
        task_id="task_easy",
        name="Task 1: Single Intersection",
        description="Minimize queue growth at one steady-flow intersection.",
        scenario_id="task_easy",
        seed=11,
    ),
    "task_medium": TaskDefinition(
        task_id="task_medium",
        name="Task 2: Corridor Coordination",
        description="Coordinate two linked intersections with changing east-west peaks.",
        scenario_id="task_medium",
        seed=23,
    ),
    "task_hard": TaskDefinition(
        task_id="task_hard",
        name="Task 3: Emergency Priority Grid",
        description="Manage a 2x2 grid while clearing emergency vehicles quickly.",
        scenario_id="task_hard",
        seed=37,
    ),
}
