import os
import sys
import json
import glob
import shutil
import argparse
from typing import List, Tuple, Any

import numpy as np
import imageio.v2 as imageio
import tensorflow as tf
import tensorflow_datasets as tfds

class PseudoPairDataset(tfds.core.GeneratorBasedBuilder):
    """RLDS from a JSON mapping: ego webm ↔ exo frame-dir (OXE-compatible)."""

    VERSION = tfds.core.Version("1.0.0")

    def __init__(self, *, json_path: str, frame_stride: int = 1, max_frames = None,
                 data_dir = None, **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        self._json_path = json_path
        self._frame_stride = int(frame_stride)
        self._max_frames = None if not max_frames or int(max_frames) <= 0 else int(max_frames)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Pseudo-paired sth-sth (ego) ↔ bridge (exo) in OXE-like RLDS.",
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": {
                        "image_exo": tfds.features.Image(encoding_format="png"),
                        "image_ego":   tfds.features.Image(encoding_format="png"),
                    },
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "is_first": tf.bool,
                    "is_last": tf.bool,
                    "is_terminal": tf.bool,
                }),
                "language_instruction": tfds.features.Text(),
                "ego_caption": tfds.features.Text(),
                "exo_caption": tfds.features.Text(),
                "similarity": tfds.features.Tensor(shape=(), dtype=np.float32),
            }),
            homepage="",
        )

    # ---------- helpers ----------
    def _read_ego_frames(self, video_path: str):
        frames = []
        try:
            rdr = imageio.get_reader(video_path, format="ffmpeg")  # <-- 핵심
        except Exception as e_ff:
            # (선택) decord 폴백: 설치되어 있으면 사용, 아니면 스킵
            try:
                import decord
                vr = decord.VideoReader(video_path, num_threads=1)
                for i in range(0, len(vr), max(1, self._frame_stride)):
                    frames.append(vr[i].asnumpy())
                    if self._max_frames and len(frames) >= self._max_frames:
                        break
                return frames
            except Exception as e_dec:
                print(f"[skip] cannot read video {video_path}: {e_ff} / {e_dec}")
                return frames

        try:
            for i, frame in enumerate(rdr):
                if i % self._frame_stride != 0:
                    continue
                frames.append(frame)
                if self._max_frames and len(frames) >= self._max_frames:
                    break
        except Exception as e_iter:
            print(f"[warn] ffmpeg reader stopped on {video_path}: {e_iter}")
        finally:
            try:
                rdr.close()
            except Exception:
                pass
        return frames

    def _read_exo_frames(self, dir_path: str) -> List[np.ndarray]:
        paths = sorted(glob.glob(os.path.join(dir_path, "*")))
        paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        if self._frame_stride > 1:
            paths = paths[::self._frame_stride]
        if self._max_frames:
            paths = paths[: self._max_frames]
        frames: List[np.ndarray] = []
        for p in paths:
            frames.append(imageio.imread(p))
        return frames

    # ---------- TFDS generation ----------
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        with tf.io.gfile.GFile(self._json_path, "r") as f:
            mapping = json.load(f)
        items = list(mapping.items())
        return {"train": self._generate_examples(items)}

    def _generate_examples(self, items: List[Tuple[str, Any]]):
        for idx, (ego_webm, meta) in enumerate(items):
            exo_dir = meta["pseudo_pair"]
            ego_cap = str(meta.get("ego_caption", ""))
            exo_cap = str(meta.get("exo_caption", ""))
            sim = float(meta.get("similarity", 0.0))

            if not tf.io.gfile.exists(ego_webm):
                print(f"[Warning] {ego_webm} does not exist")
                continue
            if not tf.io.gfile.isdir(exo_dir):
                print(f"[Warning] {exo_dir} is not dir")
                continue

            try:
                ego_frames = self._read_ego_frames(ego_webm)
            except Exception as e:
                print(f"[Warning] {e}")
                continue
            exo_frames = self._read_exo_frames(exo_dir)

            n = min(len(ego_frames), len(exo_frames))
            if n == 0:
                print("[Warning] No frame loaded")
                continue

            steps = {
                "observation": {
                    "image_exo": exo_frames[:n],   
                    "image_ego":   ego_frames[:n],   
                },
                "action": np.zeros((n, 7), np.float32),
                "is_first":    [i == 0 for i in range(n)],
                "is_last":     [i == n - 1 for i in range(n)],
                "is_terminal": [i == n - 1 for i in range(n)],
            }

            yield idx, {
                "steps": steps,
                "language_instruction": ego_cap,
                "ego_caption": ego_cap,
                "exo_caption": exo_cap,
                "similarity": sim,
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="mapping.json path (sample format as provided)")
    ap.add_argument("--data_dir", required=True, help="TFDS data_dir to write dataset")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (sample every Nth frame)")
    ap.add_argument("--max_frames", type=int, default=0, help="limit per-episode frames (0=unlimited)")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    manual_target = os.path.join(args.data_dir, "manual", "pseudo_json_rlds")
    os.makedirs(manual_target, exist_ok=True)
    dst_json = os.path.join(manual_target, "mapping.json")
    try:
        shutil.copy2(args.json, dst_json)
        print(f"[info] Copied JSON → {dst_json}")
    except Exception as e:
        print(f"[warn] Could not copy JSON to manual dir ({e}). Using original path.")

    builder = PseudoPairDataset(
        json_path=args.json,
        frame_stride=args.stride,
        max_frames=(None if args.max_frames <= 0 else args.max_frames),
        data_dir=args.data_dir,
    )
    builder.download_and_prepare()
    print(f"[done] Built TFDS at: {builder.data_path}")
    print("\nNext (example usage in your pipeline):")
    print("""
    dataset_kwargs = dict(
        name="pseudo_pair_dataset",
        data_dir="%s",
        image_obs_keys={"primary":"primary","wrist":"wrist"},
        language_key="language_instruction",
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )
    """ % args.data_dir)


if __name__ == "__main__":
    main()
