# so101-RL

MuJoCo workspace for SO-101 pick-and-place and teacher-policy training.

The env loads the upstream SO-101 Menagerie MJCF at runtime, builds the task scene on top of it, and keeps the Menagerie gripper collision model. The current default training-scene physics are aligned closely with the standalone grasp-tuning scene in [`scripts/tune_grasp_physics.py`](/home/chris/playground/so101-RL/scripts/tune_grasp_physics.py), except the training env still uses the tabletop task scene instead of a raised support platform.

## Current Scene

- Base robot model: `external/mujoco_menagerie/robotstudio_so101/so101.xml`
- Calibration reference: `external/SO-ARM100/Simulation/SO101/so101_new_calib.xml`
- Table: black tabletop in front of the robot
- Task object: dynamic green box, `28 mm x 28 mm x 28 mm`, `0.03 kg`
- Receptacle: purple round receptacle, driven by mocap during the task
- Cameras: `cam_high` and `wrist_cam`
- Wrist image observation key: `observation.images.cam_right_wrist`

The task language still says "green cylinder" in a few places, but the current task object in code is a box/cube.

## Current Physics Defaults

Training-scene defaults now match the grasp-tuning baseline closely:

- Gripper collision model: Menagerie collision geometry
- Integrator: `implicitfast`
- Solver: `Newton`
- Timestep: `0.002`
- Solver iterations: `30`
- Solver line-search iterations: `60`
- Object mass: `0.03 kg`
- Object contact friction: `1.0 0.005 0.0005`
- Object `condim`: `6`
- Gripper contact friction: `1.0 0.005 0.0005`
- Gripper `condim`: `6`
- Contact `solref`: `0.01 1`
- Table friction: `0.95 0.02 0.001`

These values live in [`so101_rl/config.py`](/home/chris/playground/so101-RL/so101_rl/config.py) and are applied in [`so101_rl/scene.py`](/home/chris/playground/so101-RL/so101_rl/scene.py).

## Task Modes

- `pick_place`: move the green object into the receptacle
- `pick_only`: grasp and lift the object
- `grasp_only`: establish a grasp
- `lift_only`: lift the object after grasping

`pick_place` succeeds when the object is physically inside the receptacle and settled there. The narrower task modes use grasp/lift success criteria instead.

## Observations

The env supports two observation modes:

- `full`: image observations plus state/task keys
- `teacher`: privileged state vector for RL

The current teacher observation is `52D`, not `47D`.

## Model Resolution

The repo does not vendor the Menagerie XML directly into the package. It resolves the robot model from one of:

- `SO101_MJCF_PATH=/absolute/path/to/so101.xml`
- `SO101_MENAGERIE_DIR=/absolute/path/to/robotstudio_so101`
- `SO101_TRS_DIR=/absolute/path/to/Simulation/SO101`
- `robot_descriptions`

The default local search path prefers:

- `external/mujoco_menagerie/robotstudio_so101`
- `external/SO-ARM100/Simulation/SO101`

## Setup

Create a venv and install the repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Install training extras if you want PPO scripts:

```bash
python3 -m pip install -e .[training]
```

Fetch the upstream assets:

```bash
bash scripts/fetch_so101_assets.sh
```

Typical local environment setup:

```bash
export SO101_MENAGERIE_DIR="$PWD/external/mujoco_menagerie/robotstudio_so101"
export SO101_TRS_DIR="$PWD/external/SO-ARM100/Simulation/SO101"
```

For headless rendering:

```bash
export MUJOCO_GL=egl
```

For GUI viewing:

```bash
unset MUJOCO_GL
```

## Main Commands

View the current training scene:

```bash
unset MUJOCO_GL && python3 -m so101_rl.viewer
```

Short headless smoke test:

```bash
MUJOCO_GL=egl python3 -m so101_rl.smoke --steps 20 --policy zero
```

Teacher-task smoke test:

```bash
MUJOCO_GL=egl python3 scripts/pick_teacher_smoke.py --steps 20 --policy zero
```

Standalone grasp-physics tuning scene:

```bash
unset MUJOCO_GL && python3 scripts/tune_grasp_physics.py
```

Headless grasp-physics tuning run:

```bash
MUJOCO_GL=egl python3 scripts/tune_grasp_physics.py --headless --print-every 20
```

## Training And Eval Scripts

Current scripts in [`scripts/`](/home/chris/playground/so101-RL/scripts):

- `control_spec.py`
- `inspect_observations.py`
- `scripted_pick_place.py`
- `eval_scripted_policy.py`
- `collect_teacher_dataset.py`
- `pick_teacher_smoke.py`
- `inspect_pick_teacher_starts.py`
- `train_pick_teacher_ppo.py`
- `eval_pick_teacher_ppo.py`
- `tune_grasp_physics.py`

Typical PPO training command:

```bash
MUJOCO_GL=egl python3 scripts/train_pick_teacher_ppo.py
```

Typical PPO evaluation command:

```bash
MUJOCO_GL=egl python3 scripts/eval_pick_teacher_ppo.py \
  --checkpoint outputs/ppo_pick_teacher/best_model/best_model.zip \
  --deterministic
```

## Important Files

- [`so101_rl/config.py`](/home/chris/playground/so101-RL/so101_rl/config.py)
- [`so101_rl/model_source.py`](/home/chris/playground/so101-RL/so101_rl/model_source.py)
- [`so101_rl/scene.py`](/home/chris/playground/so101-RL/so101_rl/scene.py)
- [`so101_rl/envs/pick_place.py`](/home/chris/playground/so101-RL/so101_rl/envs/pick_place.py)
- [`so101_rl/viewer.py`](/home/chris/playground/so101-RL/so101_rl/viewer.py)
- [`so101_rl/smoke.py`](/home/chris/playground/so101-RL/so101_rl/smoke.py)
- [`scripts/tune_grasp_physics.py`](/home/chris/playground/so101-RL/scripts/tune_grasp_physics.py)

## Notes

- Control is joint-position delta control over the six position actuators.
- With `timestep=0.002` and `frame_skip=10`, the policy/control loop runs at `50 Hz`.
- The receptacle is a mocap body, so the task goal is kinematic even though the object remains fully dynamic.
- The viewer and smoke tooling still depend on the local OpenGL setup. Use `unset MUJOCO_GL` for GUI and `MUJOCO_GL=egl` for headless runs.
