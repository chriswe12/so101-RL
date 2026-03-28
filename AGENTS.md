# AGENTS

- Default sim model: `external/mujoco_menagerie/robotstudio_so101/so101.xml`
- Calibration reference: `external/SO-ARM100/Simulation/SO101/so101_new_calib.xml`
- Task: move the green cylinder into the purple round receptacle on the black table.
- Success: cylinder is physically inside the receptacle.
- Cameras: `cam_high` and `wrist_cam`; wrist image key is `observation.images.cam_right_wrist`.
- Core files: `so101_rl/config.py`, `so101_rl/scene.py`, `so101_rl/envs/pick_place.py`
- GUI check: `unset MUJOCO_GL && python3 -m so101_rl.viewer`
- Headless smoke check: `MUJOCO_GL=egl python3 -m so101_rl.smoke --steps 20 --policy zero`
