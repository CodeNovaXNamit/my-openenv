# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Submission-facing inference script with structured logs."""

from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI

try:
    from .baseline_agent import heuristic_policy
    from .models import SmartTrafficAction, SmartTrafficObservation
    from .server.smart_traffic_env_environment import SmartTrafficEnvironment
    from .tasks import TASKS
except ImportError:
    from baseline_agent import heuristic_policy
    from models import SmartTrafficAction, SmartTrafficObservation
    from server.smart_traffic_env_environment import SmartTrafficEnvironment
    from tasks import TASKS


def _print_start(task_id: str, model_name: str) -> None:
    print(f"[START] task={task_id} env=smart_traffic_env model={model_name}", flush=True)


def _print_step(
    step_index: int,
    action: SmartTrafficAction,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_repr = json.dumps({"phase_indices": action.phase_indices}, separators=(",", ":"))
    error_repr = "null" if error is None else json.dumps(error)
    print(
        f"[STEP] step={step_index} action={action_repr} reward={reward:.4f} "
        f"done={str(done).lower()} error={error_repr}",
        flush=True,
    )


def _print_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_repr = ",".join(f"{reward:.4f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={reward_repr}",
        flush=True,
    )


def _observation_payload(observation: SmartTrafficObservation) -> str:
    return json.dumps(
        {
            "scenario_id": observation.scenario_id,
            "step_index": observation.step_index,
            "max_steps": observation.max_steps,
            "intersection_ids": observation.intersection_ids,
            "current_phase_indices": observation.current_phase_indices,
            "served_axis_indices": observation.served_axis_indices,
            "queue_lengths": observation.queue_lengths,
            "emergency_queue_lengths": observation.emergency_queue_lengths,
            "min_green_satisfied": observation.min_green_satisfied,
            "score_hint": observation.score_hint,
        },
        separators=(",", ":"),
    )


def _openai_action(
    client: OpenAI,
    model_name: str,
    observation: SmartTrafficObservation,
) -> SmartTrafficAction:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You control traffic signals. Return JSON only with schema "
                    '{"phase_indices":[int,...]}. Valid values are 0, 1, and 2. '
                    "Use 2 only when emergency preemption is needed."
                ),
            },
            {
                "role": "user",
                "content": _observation_payload(observation),
            },
        ],
        temperature=0,
        max_tokens=80,
    )
    message = response.choices[0].message.content or ""
    payload = json.loads(message)
    return SmartTrafficAction.model_validate(payload)


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME", "heuristic")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    client = (
        OpenAI(base_url=api_base_url, api_key=api_key)
        if api_base_url and api_key and model_name != "heuristic"
        else None
    )

    for task_id, task_definition in TASKS.items():
        env = SmartTrafficEnvironment()
        observation = env.reset(
            seed=task_definition.seed,
            scenario_id=task_definition.scenario_id,
        )
        rewards: List[float] = []
        _print_start(task_id=task_id, model_name=model_name if client else "heuristic")

        while not observation.done:
            error: Optional[str] = None
            if client is not None:
                try:
                    action = _openai_action(client, model_name, observation)
                except Exception as exc:
                    error = f"fallback:{type(exc).__name__}"
                    action = heuristic_policy(observation)
            else:
                action = heuristic_policy(observation)

            observation = env.step(action)
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            _print_step(
                step_index=observation.step_index,
                action=action,
                reward=reward,
                done=observation.done,
                error=error,
            )

        score = float(observation.metadata["metrics"]["score"])
        _print_end(
            success=True,
            steps=observation.step_index,
            score=score,
            rewards=rewards,
        )


if __name__ == "__main__":
    main()
