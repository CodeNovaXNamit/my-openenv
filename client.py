# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the smart traffic OpenEnv package."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SmartTrafficAction, SmartTrafficObservation, SmartTrafficState


class SmartTrafficEnv(
    EnvClient[SmartTrafficAction, SmartTrafficObservation, SmartTrafficState]
):
    """Typed OpenEnv client for the smart traffic control environment."""

    def _step_payload(self, action: SmartTrafficAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SmartTrafficObservation]:
        obs_data = dict(payload.get("observation", {}))
        obs_data["reward"] = payload.get("reward", obs_data.get("reward"))
        obs_data["done"] = payload.get("done", obs_data.get("done", False))
        observation = SmartTrafficObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SmartTrafficState:
        return SmartTrafficState.model_validate(payload)
