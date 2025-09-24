import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import signal, sys
import multiprocessing as mp

# --- OpenCV/FFmpeg 안정 옵션 ---
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")  # Linux에선 무시
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "protocol_whitelist;file,crypto,data,rtp,udp,tcp,https,tls")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------- helpers ----------
PREFERRED_EGO_BASENAMES = [
    "aria01_214-1.mp4",    # rgb (example)
    "aria01_1201-1.mp4",   # slam-left (example)
    "aria01_1201-2.mp4",   # slam-right (example)
    "aria01_211-1.mp4",    # et (example)
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Ego/Exo dense clips (per cam) into frame sequences.')
    parser.add_argument('--denseclips_dir', type=str, required=True,
                        help='Root directory for denseclips output')
    parser.add_argument('--info_clips_json', type=str, required=True,
                        help='Path to info_clips.json created earlier')
    parser.add_argument('--source_videos_dir', type=str, required=True,
                        help='Directory containing source frame-aligned videos (root of takes/...)')
    parser.add_argument('--frame_interval', type=int, default=15,
                        help='Interval between saved frames (default: 15)')
    parser.add_argument('--processes', type=int, default=1,
                        help='Number of parallel processes (default: 1)')
    return parser.parse_args()

def load_takes_index(takes_json_path):
    with open(takes_json_path, "rb") as f:
        raw = json.load(f)

    takes_list = raw
    index = {}

    ego_stream_order = ["rgb", "slam-left", "slam-right", "et"]  # first match wins
    exo_stream_key = "0"  # cam0x entries keep stream "0"

    for take in takes_list:
        take_name = take.get("take_name")
        take_uid = take.get("take_uid")
        fav = take.get("frame_aligned_videos", {})
        best_exo  = str(take.get("best_exo") or "").lower()
        medias = []

        for cam_id, streams in fav.items():
            if cam_id in ["collage", "best_exo"]:
                continue
            if str(cam_id).startswith("cam") and cam_id != best_exo:
                continue

            cam_id_l = str(cam_id).lower()
            if cam_id_l.startswith("aria"):
                role = "ego"
                chosen = None
                for sname in ego_stream_order:
                    if sname in streams:
                        chosen = streams[sname]
                        break
                if chosen is None:
                    if len(streams) > 0:
                        chosen = next(iter(streams.values()))
                    else:
                        continue
            else:
                role = "exo"
                chosen = streams.get(exo_stream_key)
                if chosen is None:
                    if len(streams) > 0:
                        chosen = next(iter(streams.values()))
                    else:
                        continue

            rel = chosen.get("relative_path")
            vid_name = rel.split("/")[-1]
            file_path = os.path.join(take_name, "frame_aligned_videos", "downscaled", "448", vid_name)

            medias.append({
                "role": role,
                "file_path": file_path,
                "fps": 30,
                "num_frames": None,
                "take_id": take_uid,
                "cam_id": cam_id
            })

        if medias:
            index[take_uid] = medias

    return index

def read_segment_frames(video_path, start_idx, end_idx, interval):
    """Read frames [start_idx, end_idx] with interval. Returns list of ndarray."""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return []

    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_idx = max(0, min(start_idx, max(0, total - 1)))
    end_idx   = max(0, min(end_idx,   max(0, total - 1)))
    if end_idx < start_idx:
        cap.release()
        return []

    # 한 번만 정확히 시킹, 그 뒤 연속 read 하며 interval로 샘플링
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    cur = start_idx
    step = max(1, interval)
    pick = 0
    while cur <= end_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if pick % step == 0:
            frames.append(frame)
        pick += 1
        cur  += 1

    cap.release()
    return frames

def _init_worker(takes_json_path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    global TAKES_INDEX
    TAKES_INDEX = load_takes_index(takes_json_path)

def cam_id_to_role(id_str):
    return "exo" if str(id_str).lower().startswith("cam") else "ego"

def process_take(take_id, actions, args):
    """
    For a single take_id:
      - iterate actions
      - for each camera mentioned in action (pre_frame_{cam}, post_frame_{cam})
      - slice frames and save as npy: denseclips_dir/take_id/action_name/{cam_id}/{00001.npy}
    """
    global TAKES_INDEX
    takes_index = TAKES_INDEX

    out_entries = []

    for action in actions:
        action_name = action['action_name']
        lang = action.get('narration_text', '')

        cam_ids = []
        for k in action.keys():
            if k.startswith("pre_frame_"):
                cam_ids.append(k.replace("pre_frame_", ""))
        cam_ids = sorted(set(cam_ids))

        take_medias = takes_index.get(take_id, [])
        for cam_id in cam_ids:
            pre_key  = f"pre_frame_{cam_id}"
            post_key = f"post_frame_{cam_id}"
            if pre_key not in action or post_key not in action:
                print(f"[INFO] Skipping: NO {pre_key} or {post_key}")
                continue

            start = int(action[pre_key]['frame_num'])
            end   = int(action[post_key]['frame_num'])

            matches = [v['file_path'] for v in take_medias if v['cam_id'] == cam_id]
            if not matches:
                print(f"[INFO] Skipping: no video for cam {cam_id} in take {take_id}")
                continue
            video_path = os.path.join(args.source_videos_dir, matches[0])
            if not os.path.exists(video_path):
                print(f"[INFO] Skipping: file not found {video_path}")
                continue

            frames = read_segment_frames(video_path, start, end, args.frame_interval)
            if not frames:
                print(f"[INFO] Skipping: fail to read frames from {video_path}")
                continue

            save_dir = os.path.join(args.denseclips_dir, take_id, action_name)
            os.makedirs(save_dir, exist_ok=True)
            for i, frame in enumerate(frames, start=1):
                npy_name = os.path.join(save_dir, f"{i:05d}_{cam_id_to_role(cam_id)}.npy")
                if not os.path.exists(npy_name):
                    np.save(npy_name, frame)

            out_entries.append({
                'take_id': take_id,
                'action_name': action_name,
                'cam_id': cam_id,
                'source_video': video_path,
                'start_frame': start,
                'end_frame': end,
                'language': lang,
            })

    return out_entries

def _process_item(kv):
    take_id, actions, args = kv
    return process_take(take_id, actions, args)

def main():
    args = parse_arguments()

    with open(args.info_clips_json, 'r') as f:
        info_clips = json.load(f)  # { take_id: [ {action_name, narration_text, pre_frame_camXX, post_frame_camXX, ...}, ...] }

    takes_json_path = os.path.join(args.source_videos_dir, "..", "takes.json")
    items = list(info_clips.items())

    if args.processes > 1:
        ctx = mp.get_context("spawn")
        info = []
        with ctx.Pool(processes=args.processes,
                      initializer=_init_worker,
                      initargs=(takes_json_path,),
                      maxtasksperchild=1) as pool:
            try:
                iterator = pool.imap_unordered(_process_item, [(t, a, args) for t, a in items], chunksize=1)
                for part in tqdm(iterator, total=len(items), desc="Processing takes (mp)"):
                    if part:
                        info.extend(part)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print("\n[WARN] KeyboardInterrupt: terminating pool...", file=sys.stderr)
                pool.terminate()
                pool.join()
            except Exception as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)
                pool.terminate()
                pool.join()

    else:
        _init_worker(takes_json_path)
        info = []
        for take_id, actions in tqdm(items, desc="Processing takes"):
            part = process_take(take_id, actions, args)
            if part:
                info.extend(part)

    os.makedirs(args.denseclips_dir, exist_ok=True)
    with open(os.path.join(args.denseclips_dir, 'annotations.json'), 'w') as f:
        json.dump(info, f, indent=4)

if __name__ == '__main__':
    main()
