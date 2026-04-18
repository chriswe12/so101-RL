# AGENTS

- Default sim model: `external/mujoco_menagerie/robotstudio_so101/so101.xml`
- Calibration reference: `external/SO-ARM100/Simulation/SO101/so101_new_calib.xml`
- Training env collision model: Menagerie gripper collision geometry
- Training env physics defaults are aligned with `scripts/tune_grasp_physics.py`, except the env uses the tabletop task scene instead of a raised support platform
- Task object in code: green dynamic box/cube (`task_object_geom`), even if some task language still says cylinder
- Task: move the green object into the purple round receptacle on the black table
- Success: object is physically inside the receptacle
- Cameras: `cam_high` and `wrist_cam`; wrist image key is `observation.images.cam_right_wrist`.
- Teacher observation size: `52D`
- Core files: `so101_rl/config.py`, `so101_rl/scene.py`, `so101_rl/envs/pick_place.py`, `scripts/tune_grasp_physics.py`
- GUI check: `unset MUJOCO_GL && python3 -m so101_rl.viewer`
- Headless smoke check: `MUJOCO_GL=egl python3 -m so101_rl.smoke --steps 20 --policy zero`
