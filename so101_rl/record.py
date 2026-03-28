from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeRecorder:
    def __init__(self) -> None:
        self._data: dict[str, list[np.ndarray]] = defaultdict(list)
        self._step_count = 0

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        for key, value in observation.items():
            self._data[key].append(np.asarray(value))
        self._data["action"].append(np.asarray(action, dtype=np.float32))
        self._data["reward"].append(np.asarray(reward, dtype=np.float32))
        self._data["terminated"].append(np.asarray(terminated, dtype=np.bool_))
        self._data["truncated"].append(np.asarray(truncated, dtype=np.bool_))
        self._data["is_success"].append(np.asarray(info.get("is_success", False), dtype=np.bool_))
        self._step_count += 1

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {key: np.stack(values) for key, values in self._data.items()}
        np.savez_compressed(path, **arrays)

        meta_path = path.with_suffix(".json")
        payload = {
            "steps": self._step_count,
            "keys": sorted(arrays.keys()),
        }
        if metadata:
            payload.update(metadata)
        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
