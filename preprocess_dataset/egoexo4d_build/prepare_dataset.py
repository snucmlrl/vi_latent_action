import os
import re
import json
import cv2
import argparse
import tqdm

def sanitize(text: str) -> str:
    """Make a safe file/dir name from free text."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^a-zA-Z0-9_\-\.\s]", "", text)
    return text.strip().replace(" ", "_")

def read_frame_by_num(cap, frame_num):
    """Seek to an absolute frame index and read a single frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Error reading frame {frame_num}")
    return frame

def sec_to_frame(sec, fps):
    """Convert seconds to nearest frame index."""
    return int(round(float(sec) * float(fps)))

def resolve_video_path(root_takes_dir, file_path):
    p = os.path.join(root_takes_dir, file_path)
    
    return p

def build_step_name(seg, vocab, taxonomy_by_scenario, scenario):
    """Choose a human-readable step name with fallbacks."""
    name = seg.get("step_name")
    if name:
        return name
    uniq = seg.get("step_unique_id")
    if uniq is not None and str(uniq) in vocab:
        return vocab[str(uniq)]
    tax = taxonomy_by_scenario.get(scenario) if scenario else None
    if tax and uniq is not None:
        node = tax.get(str(uniq))
        if node and "name" in node:
            return node["name"]
    return f"step_{seg.get('step_id', 'unknown')}"

def frames_bounds(cap, meta_num_frames):
    """Get max valid frame index with fallback to cv2 if metadata missing."""
    n_total = meta_num_frames if (meta_num_frames is not None and meta_num_frames > 0) \
        else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return max(0, n_total - 1)

def load_takes_index(takes_json_path, root_dir_for_paths):
    with open(takes_json_path, "rb") as f:
        raw = json.load(f)

    # takes.json is a list in your sample
    takes_list = raw
    index = {}

    # stream preference per camera family
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
            if cam_id.startswith("cam") and cam_id != best_exo:
                continue
            # Decide role from cam_id
            cam_id_l = str(cam_id).lower()
            if cam_id_l.startswith("aria"):
                role = "ego"
                chosen = None
                for sname in ego_stream_order:
                    if sname in streams:
                        chosen = streams[sname]
                        break
                if chosen is None:
                    # fallback: any first stream entry
                    if len(streams) > 0:
                        chosen = next(iter(streams.values()))
                    else:
                        continue
            else:
                role = "exo"
                chosen = streams.get(exo_stream_key)
                if chosen is None:
                    # fallback: any first stream entry
                    if len(streams) > 0:
                        chosen = next(iter(streams.values()))
                    else:
                        continue

            # file_path relative to dataset root (starts with "takes/...")
            rel = chosen.get("relative_path")
            vid_name = rel.split("/")[-1]
            file_path = os.path.join(take_name, "frame_aligned_videos", "downscaled", "448", vid_name)  # e.g., "takes/cmu_bike01_2/frame_aligned_videos/cam04.mp4"

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

def process_one_media(take_uid, media, segments, scenario, taxonomy_by_scenario, vocabulary,
                      takes_dir, out_dir, info_clips,
                      only_essential=False):
    file_path = media.get("file_path")
    fps_meta = media.get("fps")
    num_frames_meta = media.get("num_frames")
    take_id = media.get("take_id") or take_uid
    cam_id = str(media.get("cam_id"))

    video_path = resolve_video_path(takes_dir, file_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps == fps_meta, f"{fps} should be match to {fps_meta}!"

    # Resolve FPS
    # fps = fps_meta
    if fps is None or fps <= 0:
        fps_fallback = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_fallback) if fps_fallback and fps_fallback > 0 else None
    if fps is None:
        cap.release()
        return

    max_idx = frames_bounds(cap, num_frames_meta)

    # Filter segments once (respect only_essential + time fields)
    kept = []
    for i, seg in enumerate(segments):
        if only_essential and not bool(seg.get("is_essential", False)):
            continue
        if seg.get("start_time") is None or seg.get("end_time") is None:
            continue
        kept.append((i, seg))

    # Initialize per-take action list ONCE with narration/action_name (no frames yet)
    if take_id not in info_clips:
        actions_out = []
        for i, seg in kept:
            step_name = build_step_name(seg, vocabulary, taxonomy_by_scenario, scenario)
            step_name_clean = sanitize(step_name)
            action_name = f"action_{str(i).zfill(3)}_{step_name_clean}"
            actions_out.append({
                "action_name": action_name,
                "narration_text": step_name,
                # per-cam frames will be filled below, e.g., pre_frame_{cam}, post_frame_{cam}
            })
        info_clips[take_id] = actions_out

    actions_out = info_clips[take_id]

    # Fill frames for THIS camera into the existing action entries
    act_idx = 0
    for i, seg in kept:
        step_name = build_step_name(seg, vocabulary, taxonomy_by_scenario, scenario)
        step_name_clean = sanitize(step_name)
        save_dir = os.path.join(take_id, f"action_{str(i).zfill(3)}_{step_name_clean}")
        os.makedirs(os.path.join(out_dir, save_dir), exist_ok=True)

        start_sec = float(seg["start_time"])
        end_sec   = float(seg["end_time"])
        pre_f  = max(0, min(sec_to_frame(start_sec, fps), max_idx))
        post_f = max(0, min(sec_to_frame(end_sec,   fps), max_idx))

        try:
            pre_img  = read_frame_by_num(cap, pre_f)
            post_img = read_frame_by_num(cap, post_f)
        except Exception:
            act_idx += 1
            continue

        rel_pre  = os.path.join(save_dir, f"pre_frame_{cam_id}.jpg")
        rel_post = os.path.join(save_dir, f"post_frame_{cam_id}.jpg")
        cv2.imwrite(os.path.join(out_dir, rel_pre),  pre_img)
        cv2.imwrite(os.path.join(out_dir, rel_post), post_img)

        # Merge into the per-take action entry
        entry = actions_out[act_idx]
        entry[f"pre_frame_{cam_id}"]  = {"frame_num": pre_f, "path": rel_pre}
        entry[f"post_frame_{cam_id}"] = {"frame_num": post_f, "path": rel_post}
        act_idx += 1

    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Export pre/post frames from EgoExo4D keystep annotations (no PNR).")
    parser.add_argument("--root_path", type=str, required=True,
                        help="Path to egoexo4d root (expects annotations/ and takes/).")
    parser.add_argument("--only_essential", action="store_true",
                        help="Use only segments with is_essential==True.")
    args = parser.parse_args()

    takes_dir = os.path.join(args.root_path, "takes")
    ann_dir   = os.path.join(args.root_path, "annotations")
    takes_json_path = os.path.join(args.root_path, "takes.json")  # <-- use takes.json (not captures.json)
    out_dir = os.path.join(args.root_path, "clips_jpgs", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build take_uid -> list[media(ego/exo...)] from takes.json
    takes_index = load_takes_index(takes_json_path, takes_dir)

    # 2) Load and merge keystep_train/val into one ks-like dict
    ks = {"annotations": {}, "taxonomy": {}, "vocabulary": {}}
    for p in [os.path.join(ann_dir, "keystep_train.json"),
              os.path.join(ann_dir, "keystep_val.json")]:
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            ks_part = json.load(f)
        for k, v in ks_part.get("annotations", {}).items():
            ks["annotations"].setdefault(k, v)
        for scen, d in ks_part.get("taxonomy", {}).items():
            dst = ks["taxonomy"].setdefault(scen, {})
            for uid, node in d.items():
                dst.setdefault(uid, node)
        for k, v in ks_part.get("vocabulary", {}).items():
            ks["vocabulary"].setdefault(k, v)

    annotations = ks.get("annotations", {})
    taxonomy_all = ks.get("taxonomy", {})
    vocabulary   = ks.get("vocabulary", {})

    taxonomy_by_scenario = {}
    if isinstance(taxonomy_all, dict):
        for scen, d in taxonomy_all.items():
            if isinstance(d, dict):
                taxonomy_by_scenario[scen] = d

    info_clips = {}
    # 3) Iterate keystep takes; for each, process *all* media (ego+exo) listed under takes.json
    for take_uid, take_entry in tqdm.tqdm(annotations.items()):
        scenario = take_entry.get("scenario")
        segments = take_entry.get("segments", [])
        if not segments:
            continue

        medias = takes_index.get(take_uid)
        if medias is None:
            continue
        # process all available media for this take (we want both ego and exo)
        for media in medias:
            process_one_media(
                take_uid=take_uid,
                media=media,
                segments=segments,
                scenario=scenario,
                taxonomy_by_scenario=taxonomy_by_scenario,
                vocabulary=vocabulary,
                takes_dir=takes_dir,
                out_dir=out_dir,
                info_clips=info_clips,
                only_essential=args.only_essential,
            )

    with open(os.path.join(out_dir, "info_clips.json"), "w") as f:
        json.dump(info_clips, f, indent=2)

if __name__ == "__main__":
    main()
