# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Heuristic baseline policies for the smart traffic environment."""

from __future__ import annotations

try:
    from .models import SmartTrafficAction, SmartTrafficObservation
except ImportError:
    from models import SmartTrafficAction, SmartTrafficObservation


def heuristic_policy(observation: SmartTrafficObservation) -> SmartTrafficAction:
    """Serve the busiest axis, with hard priority for emergency queues."""

    phase_indices = []
    for current_phase, queues, emergencies, min_green_ready in zip(
        observation.current_phase_indices,
        observation.queue_lengths,
        observation.emergency_queue_lengths,
        observation.min_green_satisfied,
    ):
        ns_pressure = queues[0] + queues[2] + 8 * (emergencies[0] + emergencies[2])
        ew_pressure = queues[1] + queues[3] + 8 * (emergencies[1] + emergencies[3])
        emergency_present = sum(emergencies) > 0

        if emergency_present:
            phase_indices.append(2)
            continue

        target_axis = 0 if ns_pressure >= ew_pressure else 1
        if not min_green_ready and current_phase in (0, 1):
            phase_indices.append(current_phase)
        else:
            phase_indices.append(target_axis)

    return SmartTrafficAction(phase_indices=phase_indices)
