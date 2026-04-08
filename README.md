---
title: Smart Traffic Signal Control Environment
emoji: "🚦"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - smart-city
---

# Smart Traffic Signal Control Environment

An OpenEnv-compatible reinforcement-learning environment for adaptive traffic-signal control. The environment simulates a city grid of intersections, deterministic vehicle arrivals, queue spillback between intersections, and emergency vehicles that require preemption.

The package is designed for hackathon submission flow:

- Typed OpenEnv server with `reset()`, `step()`, and `state()`
- Three built-in tasks from easy to hard
- Deterministic graders that return scores in `[0.0, 1.0]`
- Heuristic baseline agent and OpenAI-backed inference script
- Docker packaging for Hugging Face Spaces deployment

## Environment Summary

### Observation

Each observation contains:

- Current phase command per intersection
- Operational axis currently being served
- Queue lengths for regular and emergency vehicles on `[north, east, south, west]`
- Min-green status for each intersection
- Demand rates used for the next arrivals
- Aggregate metrics such as throughput, cumulative wait, emergency wait, and score hint

### Action

The action is discrete for each intersection:

- `0`: `north_south_green`
- `1`: `east_west_green`
- `2`: `emergency_override`

`emergency_override` selects the axis with the strongest emergency pressure and can bypass the normal min-green rule when an emergency vehicle is present.

### Reward

The step reward is dense and shaped from:

- Queue reduction
- Vehicles cleared from the network
- Emergency clearances
- Penalties for growing regular and emergency queues

## Built-in Tasks

### Task 1: `task_easy`

- Topology: `1 x 1`
- Goal: minimize queues at a single steady-flow intersection
- Difficulty: easy

### Task 2: `task_medium`

- Topology: `1 x 2`
- Goal: coordinate a corridor with changing east-west peaks
- Difficulty: medium

### Task 3: `task_hard`

- Topology: `2 x 2`
- Goal: manage network congestion while clearing emergency vehicles quickly
- Difficulty: hard

## Baseline Scores

The built-in heuristic baseline currently scores:

- `task_easy`: `0.4828`
- `task_medium`: `0.4940`
- `task_hard`: `0.6435`

Run the scorer locally:

```bash
python -m smart_traffic_env.inference
```

If `OPENAI_API_KEY` and `MODEL_NAME` are set, the same script will use an OpenAI-compatible model policy and fall back to the heuristic policy on malformed output.

For submission configuration, define:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

An example file is provided at `.env.example`. The submission-facing `inference.py` reads those variables first, falls back to `OPENAI_API_KEY` for local compatibility, and emits only `[START]`, `[STEP]`, and `[END]` stdout lines.

## Project Layout

```text
smart_traffic_env/
├── __init__.py
├── baseline_agent.py
├── client.py
├── graders.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── simulator.py
├── tasks.py
├── tests/
│   └── test_environment.py
└── server/
    ├── __init__.py
    ├── app.py
    ├── Dockerfile
    ├── requirements.txt
    └── smart_traffic_env_environment.py
```

## Local Development

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
python -m pytest -q
```

Validate the environment:

```bash
openenv validate . --verbose
```

Run the local pre-submission validator:

```bash
python validate_presubmit.py
```

Run the server locally:

```bash
uv run server
```

Or directly:

```bash
python -m smart_traffic_env.server.app
```

## Docker

Build the image:

```bash
docker build -t smart-traffic-env:latest -f server/Dockerfile .
```

Run it:

```bash
docker run -p 8000:8000 smart-traffic-env:latest
```

## Using the Typed Client

```python
from smart_traffic_env import SmartTrafficAction, SmartTrafficEnv

client = SmartTrafficEnv(base_url="http://localhost:8000").sync()
with client:
    result = client.reset(seed=23, scenario_id="task_medium")
    action = SmartTrafficAction(phase_indices=[0, 1])
    result = client.step(action)
    print(result.observation.total_queue_length, result.reward)
```

## Grading API

Programmatic grading is available from Python:

```python
from smart_traffic_env import grade_all, grader

print(grader("task_easy"))
print({task_id: result.score for task_id, result in grade_all().items()})
```

## Deployment

Push to Hugging Face Spaces from this directory:

```bash
openenv push --repo-id <your-user>/smart-traffic-env
```

The Space exposes:

- `/reset`
- `/step`
- `/state`
- `/schema`
- `/health`
- `/ws`
- `/web`

## Notes

- The simulator is deterministic for a fixed `(scenario_id, seed)`.
- The implementation uses a pure Python queue model instead of SUMO so it stays lightweight for Docker and Space builds.
- Emergency vehicles travel straight through the network and are counted separately from regular traffic.
