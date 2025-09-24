"""
Convert various video-text datasets to a unified meta.jsonl format:
Bridge V2, DROID, H2O, Something-Something V2
"""

import argparse, json, csv, os, re
from pathlib import Path
from typing import List, Dict
import decord
from tqdm import tqdm

def _count_frames(path: Path, image=False) -> int:
    if not image:
        return len(decord.VideoReader(str(path), num_threads=1))
    else:
        return sum(1 for p in path.iterdir()
               if p.is_file() and re.match(r"im_\d+\.(jpe?g|png)$", p.name, re.I))

def _read_lang_txt(traj_dir: Path) -> str:
    """
    traj_dir/
        lang.txt
            put the red thing in the silver pot
            confidence: 0.83
    """
    f = traj_dir / "lang.txt"
    if not f.exists():
        return ""

    try:
        with f.open("r", encoding="utf-8") as fp:
            for raw in fp:
                line = raw.strip()
                if not line:
                    continue
                if line.lower().startswith("confidence"):
                    continue
                line = re.split(r"\s*confidence\s*:", line, flags=re.I)[0].strip()
                return line
    except Exception:
        pass
    return ""


def build_bridgev2(root: Path) -> List[Dict]:
    """
    root/
        datacol1_toykitchen1/
            many_skills/
                00/
                    2023-03-15_14-35-28/
                        collection_metadata.json
                        config.json
                        diagnostics.png
                        raw/
                            traj_group0/
                                traj0/
                                    obs_dict.pkl
                                    policy_out.pkl
                                    agent_data.pkl
                                    lang.txt
                                    images0/
                                        im_0.jpg
                                        im_1.jpg
                                        ...
                                ...
                            ...
                01/
                ...
    """
    episodes: List[Dict] = []
    pattern = "**/raw/traj_group*/traj*/images0"
    for img_dir in tqdm(root.glob(pattern), desc="Bridge V2", unit="traj"):
        if not img_dir.is_dir():
            continue

        n_frames = _count_frames(img_dir, image=True)
        if n_frames == 0:
            continue

        traj_dir = img_dir.parent
        caption = _read_lang_txt(traj_dir)

        video_id = img_dir.relative_to(root).as_posix()
        episodes.append(
            dict(
                video=str(img_dir),# data/bridge_v2/raw/bridge_data_v2/datacol1_toykitchen1/many_skills/00/2023-03-15_13-35-31/raw/traj_group0/traj0/images0
                n_frames=n_frames, # 30
                caption=caption,   # put the red thing in the silver pot
            )
        )

    return episodes

def build_droid(root: Path) -> List[Dict]:
    """
    RLDS
        DROID = {
            "episode_metadata": {
                    "recording_folderpath": tf.Text, # path to the folder of recordings
                    "file_path": tf.Text, # path to the original data file
                    },
            "steps": {
                "is_first": tf.Scalar(dtype=bool), # true on first step of the episode
                        "is_last": tf.Scalar(dtype=bool), # true on last step of the episode
                    "is_terminal": tf.Scalar(dtype=bool), # true on last step of the episode if it is a terminal step, True for demos
                                        
                        "language_instruction": tf.Text, # language instruction
                        "language_instruction_2": tf.Text, # alternative language instruction
                        "language_instruction_3": tf.Text, # alternative language instruction
                        "observation": {
                                        "gripper_position": tf.Tensor(1, dtype=float64), # gripper position state
                                        "cartesian_position": tf.Tensor(6, dtype=float64), # robot Cartesian state
                                        "joint_position": tf.Tensor(7, dtype=float64), # joint position state
                                        "wrist_image_left": tf.Image(180, 320, 3, dtype=uint8), # wrist camera RGB left viewpoint        
                                        "exterior_image_1_left": tf.Image(180, 320, 3, dtype=uint8), # exterior camera 1 left viewpoint
                                        "exterior_image_2_left": tf.Image(180, 320, 3, dtype=uint8), # exterior camera 2 left viewpoint
                                },                            
                        "action_dict": {
                                        "gripper_position": tf.Tensor(1, dtype=float64), # commanded gripper position
                                        "gripper_velocity": tf.Tensor(1, dtype=float64), # commanded gripper velocity
                                        "cartesian_position": tf.Tensor(6, dtype=float64), # commanded Cartesian position
                                        "cartesian_velocity": tf.Tensor(6, dtype=float64), # commanded Cartesian velocity
                                        "joint_position": tf.Tensor(7, dtype=float64),  # commanded joint position
                                    "joint_velocity": tf.Tensor(7, dtype=float64), # commanded joint velocity
                                },
                "discount": tf.Scalar(dtype=float32), # discount if provided, default to 1
                        "reward": tf.Scalar(dtype=float32), # reward if provided, 1 on final step for demos
                        "action": tf.Tensor(7, dtype=float64), # robot action, consists of [6x joint velocities, 1x gripper position]
            },
        }

    Raw
        episode:
            |
            |---- metadata_*.json: Episode metadata like building ID, data collector ID etc.
            |---- trajectory.h5: All low-dimensional information like action and proprioception trajectories.
            |---- recordings:
                        |
                        |---- MP4:
                        |      |
                        |      |---- *.mp4: High-res video of single (left) camera view.
                        |      |---- *-stereo.mp4: High-res video of concatenated stereo camera views.
                        |
                        |---- SVO:
                                |
                                |---- *.svo: Raw ZED SVO file with encoded camera recording information (contains some additional metadata)
    """
    entries: List[Dict] = []
    for meta_file in tqdm(root.rglob("episode_metadata.json"),
                          desc="DROID-RLDS", unit="ep"):

        ep_dir = meta_file.parent              # …/episode_xxxx

        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue

        caption = (meta.get("language_instruction", "") or
                   meta.get("language_instruction_2", "") or
                   meta.get("language_instruction_3", ""))

        recording_root = Path(meta["recording_folderpath"]) \
                         if "recording_folderpath" in meta else None
        if not recording_root or not recording_root.exists():
            continue

        mp4_dir = recording_root / "recordings" / "MP4"
        mp4_candidates = sorted(mp4_dir.glob("*ext1_left*.mp4"))
        if not mp4_candidates:
            mp4_candidates = sorted(mp4_dir.glob("*.mp4"))
        if not mp4_candidates:
            continue

        vid = mp4_candidates[0]
        n_frames = _count_frames_mp4(vid)

        entries.append(dict(
            video=str(vid),
            n_frames=n_frames,
            caption=caption,
        ))

    return entries

def build_h2o(root: Path) -> List[Dict]:
    """
        .
    ├── h1
    │   ├── 0
    │   │   │── cam0
    │   │   │   ├── rgb
    │   │   │   ├── depth
    │   │   │   ├── cam_pose
    │   │   │   ├── hand_pose
    │   │   │   ├── hand_pose_MANO
    │   │   │   ├── obj_pose
    │   │   │   ├── obj_pose_RT
    │   │   │   ├── action_label (only in cam4)
    │   │   │   ├── rgb256 (only in cam4)
    │   │   │   ├── verb_label
    │   │   │   └── cam_intrinsics.txt
    │   │   ├── cam1
    │   │   ├── cam2
    │   │   ├── cam3
    │   │   └── cam4
    │   ├── 1
    │   ├── 2
    │   ├── 3
    │   └── ...
    ├── h2
    ├── k1
    ├── k2
    └── ...
    """
    entries: List[Dict] = []
    for cam4_dir in tqdm(root.glob("*/*/cam4"), desc="H2O", unit="episode"):
        if not cam4_dir.is_dir():
            continue

        cam4_rgb = cam4_dir / "rgb"
        if not cam4_rgb.is_dir():
            continue

        cam0_rgb = cam4_dir.parent / "cam0" / "rgb"
        if not cam0_rgb.is_dir():
            continue

        n_frames = _count_frames_rgb(cam0_rgb)
        if n_frames or n_frames != _count_frames_rgb(cam4_rgb):
            continue

        caption = _read_action_label(cam4_dir)

        entries.append(
            dict(
                video=[str(cam0_rgb), str(cam4_rgb)], # exo, ego
                n_frames=n_frames,
                caption=caption,
            )
        )

    return entries

def build_sth(root: Path) -> List[Dict]:
    """
    root/
      ├── raw/<id>.webm
      └── labels/
          ├── labels.json
          ├── train.json
          ├── validation.json
          └── test.json
    """
    vids_dir = root / "raw"
    labels   = json.load(open(root / "labels" / "labels.json"))

    metas = []
    for name in ("train.json", "validation.json"):
        with open(root / "labels" / name) as f:
            metas += json.load(f)

    out: List[Dict] = []
    for m in tqdm(metas, desc="SSv2"):
        vid_path = vids_dir / f"{m['id']}.webm"
        if not vid_path.exists():
            continue

        templ = m["template"].replace("[", "").replace("]", "")
        caption = m['label']

        try:
            n_frames = len(decord.VideoReader(str(vid_path)))
        except Exception:
            n_frames = -1

        out.append(dict(
            video=str(vid_path),
            n_frames=n_frames,
            caption=caption,
        ))
    return out

builders = dict(
    bridge = build_bridgev2,
    droid    = build_droid,
    h2o      = build_h2o,
    sth_sth_v2=build_sth,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
        choices=builders.keys(), help="bridge droid h2o sth_sth_v2")
    ap.add_argument("--root", required=True,
        help="path to dataset root directory")
    ap.add_argument("--out",  default="meta.jsonl",
        help="output meta jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    samples = builders[args.dataset](root)

    with open(args.out, "w") as fp:
        for s in samples:
            fp.write(json.dumps(s)+"\n")

if __name__ == "__main__":
    main()
