# so101-RL

MuJoCo workspace scaffold for training an SO-101 follower arm to pick up a green cylinder and place it into a purple round receptacle from camera observations.

The current repo sets up:

- A Gymnasium-style MuJoCo environment for SO-101 pick-and-place.
- Camera observations from a top camera and a gripper-mounted wrist camera.
- A task scene with a large black table, one green cylinder, one purple round receptacle, and deterministic reset randomization.
- A small rollout recorder for storing stable observation/action keys.

The current repo does not vendor the Menagerie robot assets. It loads the official SO-101 MuJoCo model at runtime from either:

- `SO101_MJCF_PATH=/absolute/path/to/so101.xml`
- `SO101_MENAGERIE_DIR=/absolute/path/to/robotstudio_so101`
- `SO101_TRS_DIR=/absolute/path/to/Simulation/SO101`
- `robot_descriptions`

## Concrete Build Order

### Phase 1: robot model sanity

Use the MuJoCo Menagerie `robotstudio_so101/so101.xml` model as the base robot model.

Check:

- Joint limits and joint ordering.
- Gravity behavior and the default `home` keyframe.
- Self-collision behavior.
- Gripper motion and jaw sign convention.
- Camera mount convention against the official TRS `Simulation/SO101/so101_new_calib.xml`.

Current repo status:

- Joint and actuator resolution supports the current Menagerie names:
  - `shoulder_pan`
  - `shoulder_lift`
  - `elbow_flex`
  - `wrist_flex`
  - `wrist_roll`
  - `gripper`
- The env still accepts the older `Rotation` / `Pitch` / `Elbow` naming if you point it at an older model.
- The scene builder prefers the existing Menagerie `gripperframe` site and `wrist_cam`, and only injects them if they are missing.

### Phase 2: task scene

The workspace already adds:

- A tabletop workspace in front of the robot.
- One dynamic green cylinder with a free joint.
- One movable purple round receptacle used as the insertion target.
- Reset randomization for cylinder and receptacle positions.
- A top camera named `cam_high`.
- A gripper camera named `cam_right_wrist`.

### Phase 3: environment API

The workspace already exposes:

- `reset()` / `step()` Gymnasium API.
- `render_mode="rgb_array"`.
- Deterministic seeds through Gymnasium seeding.
- Observation keys that are stable and usable for LeRobot-style datasets:
  - `observation.images.cam_high`
  - `observation.images.cam_right_wrist`
  - `observation.state`
  - `observation.velocity`
  - `observation.task_state`
  - `achieved_goal`
  - `desired_goal`

### Phase 4: data + policy

The repo includes a minimal `EpisodeRecorder` and a smoke rollout entrypoint so you can start storing episodes in a stable format before wiring teleop or LeRobot training.

### Phase 5: sim-to-real

This repo does not attempt sim-to-real tuning yet. The remaining work is listed below and should be treated as required before you trust the policy on hardware.

## Files

- [`pyproject.toml`](/home/chris/playground/so101-RL/pyproject.toml)
- [`so101_rl/config.py`](/home/chris/playground/so101-RL/so101_rl/config.py)
- [`so101_rl/model_source.py`](/home/chris/playground/so101-RL/so101_rl/model_source.py)
- [`so101_rl/scene.py`](/home/chris/playground/so101-RL/so101_rl/scene.py)
- [`so101_rl/envs/pick_place.py`](/home/chris/playground/so101-RL/so101_rl/envs/pick_place.py)
- [`so101_rl/record.py`](/home/chris/playground/so101-RL/so101_rl/record.py)
- [`so101_rl/smoke.py`](/home/chris/playground/so101-RL/so101_rl/smoke.py)

## Setup

Create a virtualenv and install the repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Fetch the upstream assets from the exact source repos you provided:

```bash
bash scripts/fetch_so101_assets.sh
```

That script sparse-checks out:

- `https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101`
- `https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101`

Install one robot-model source:

Option 1: use the fetched Menagerie assets

```bash
export SO101_MENAGERIE_DIR="$PWD/external/mujoco_menagerie/robotstudio_so101"
```

Option 2: use the official TRS calibration files directly

```bash
export SO101_TRS_DIR="$PWD/external/SO-ARM100/Simulation/SO101"
export SO101_MJCF_PATH="$SO101_TRS_DIR/so101_new_calib.xml"
```

Option 3: use `robot_descriptions`

```bash
python3 -m pip install robot_descriptions
```

For headless rendering, prefer:

```bash
export MUJOCO_GL=egl
```

## Smoke Test

Run a short rollout:

```bash
python3 -m so101_rl.smoke --steps 100 --policy zero
```

Save a rollout to disk:

```bash
python3 -m so101_rl.smoke --steps 150 --policy random --record outputs/smoke_episode.npz
```

The recorder writes:

- `outputs/smoke_episode.npz`
- `outputs/smoke_episode.json`

## GUI Viewer

To inspect the loaded scene in the native MuJoCo GUI, use:

```bash
unset MUJOCO_GL
python3 -m so101_rl.viewer
```

If you want the robot to move while the viewer is open:

```bash
python3 -m so101_rl.viewer --policy random --reset-every 5
```

Notes:

- For GUI viewing, do not force `MUJOCO_GL=egl`. That is for headless rendering.
- The viewer opens with the free camera. Use the MuJoCo UI to switch to named cameras such as `cam_high` and `wrist_cam`.

## Assumptions

This scaffold makes a few explicit assumptions:

- The Menagerie `robotstudio_so101/so101.xml` model is the correct base simulation model.
- The TRS `Simulation/SO101/so101_new_calib.xml` file is the calibration reference when you compare sim against the official upstream geometry.
- The top camera observation key should be `cam_high`, while the wrist observation key should be `cam_right_wrist` even though the current Menagerie camera object is named `wrist_cam`.
- Actions are normalized joint-space delta commands over the six position actuators.
- The task is solved when the cylinder is physically inside the receptacle.

## Remaining Work

These items still need to be done before the workspace is complete:

1. Verify the SO-101 calibration convention against `Simulation/SO101/so101_new_calib.xml` and your real follower arm.
2. Visually calibrate the wrist camera pose. The current workspace uses the Menagerie `wrist_cam` on the gripper camera mount when available, but the final extrinsics still need to be matched to your real gripper camera.
3. Visually calibrate the top camera height, crop, focal settings, and image resolution to your real top camera.
4. Check that the table height and cylinder/receptacle spawn region are actually reachable and stable with the Menagerie model. The current values are reasonable defaults, not measured geometry.
5. Tune contact/friction and cylinder mass so the gripper can reliably pinch, carry, and insert without unrealistic slip or sticking.
6. Decide whether your training action space should remain joint deltas or switch to Cartesian delta control plus a gripper scalar.
7. Add teleoperation or scripted demonstration collection. The repo only includes a recorder, not a teleop stack.
8. Add a dataset export path that exactly matches the feature schema expected by your intended LeRobot training recipe.
9. Train a small imitation baseline first before attempting RL. This is the right order for a vision pick-and-place stack on SO-101.
10. Add domain randomization for textures, lighting, object pose, camera pose, and friction.
11. Add latency and action-hold modeling so the sim control loop matches the real follower arm and camera pipeline.
12. Add conservative hardware validation steps before running any learned policy on the real robot.

## Resource List

These are the core references this scaffold is based on:

- MuJoCo Menagerie model: `https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101`
- The Robot Studio official simulation files: `https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101`
- Official SO-101 docs on Hugging Face
- SO101-Nexus MuJoCo environment references
- LeRobot repo and docs
- MuJoCo docs, especially renderer and headless backend notes

## Recommended Next Step

Install `mujoco`, `gymnasium`, and one robot-model source, then run the smoke test and inspect the two camera views first. The next code iteration should be driven by what is wrong in those rendered views, not by adding more training code before the geometry is correct.
