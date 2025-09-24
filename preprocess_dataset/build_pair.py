# scripts/build_pair.py
import argparse, json, random, glob, warnings, pathlib, sys, time
from pathlib import Path
from collections import OrderedDict

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms._transforms_video as T_vid
from PIL import Image
import decord
from tqdm import tqdm
import torchvision.transforms.functional as TF
import numpy as np

# ────────────────────────────────────── LAVILA import ─────────────────────────
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "lavila"))

from lavila.lavila.models import models as lavila_models
from lavila.lavila.models.utils import inflate_positional_embeds
from lavila.lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.lavila.utils.preprocess import generate_tokenizer
from lavila.lavila.data.datasets import get_frame_ids
# ───────────────────────────────────────────────────────────────────────────────

# ------------------------------- helper ---------------------------------------
def load_lavila_ckpt(ckpt_path: str, num_frames: int, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = OrderedDict((k.replace("module.", ""), v) for k, v in ckpt["state_dict"].items())
    cfg  = ckpt["args"]

    model_cls = getattr(lavila_models, cfg.model)
    model = model_cls(
        text_use_cls_token=getattr(cfg, "use_cls_token", False),
        project_embed_dim=getattr(cfg, "project_embed_dim", 256),
        gated_xattn=getattr(cfg, "gated_xattn", False),
        timesformer_gated_xattn=getattr(cfg, "timesformer_gated_xattn", False),
        timesformer_freeze_space=getattr(cfg, "timesformer_freeze_space", False),
        freeze_lm_vclm=getattr(cfg, "freeze_lm_vclm", False),
        freeze_visual_vclm=getattr(cfg, "freeze_visual_vclm", False),
        num_frames=num_frames,
        drop_path_rate=0,
    )
    tokenizer  = generate_tokenizer(cfg.model)

    if "TIMESFORMER" in cfg.model:
        sd = inflate_positional_embeds(model.state_dict(), sd,
                                       num_frames=num_frames,
                                       load_temporal_fix="bilinear")
    msg = model.load_state_dict(sd, strict=True)
    print(f"✓ model loaded (miss {len(msg.missing_keys)}, extra {len(msg.unexpected_keys)})")
    return model.to(device).eval(), tokenizer, cfg


def build_val_transform(cfg):
    crop = 224 if "336PX" not in cfg.model else 336
    norm = (
        T_vid.NormalizeVideo(mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375])
        if "OPENAI" not in cfg.model else
        T_vid.NormalizeVideo(mean=[108.3272985,116.7460125,104.09373615],
                             std=[68.5005327,66.6321579,70.32316305])
    )
    return transforms.Compose([
        Permute([3,0,1,2]),         # T H W C → C T H W
        transforms.Resize(crop),
        transforms.CenterCrop(crop),
        norm,
    ])
# ------------------------------------------------------------------------------
class VideoPairDataset(Dataset):
    def __init__(self, meta_jsonl, num_frames=4, transform = None):
        self.items = [json.loads(l) for l in open(meta_jsonl)]
        self.num_frames = num_frames
        self.tf = transform

    def _load_frames_video(self, p: Path):
        vr   = decord.VideoReader(str(p), num_threads=1)
        tot  = len(vr)
        if tot < self.num_frames:
            raise RuntimeError(f"{p} : {tot} < {self.num_frames}")

        idxs = get_frame_ids(0, len(vr), num_segments=self.num_frames, jitter = False)
        frames = []
        for idx in idxs:
            try:
                frame = vr[idx]
            except Exception as e :
                vr.seek(idx)
                frame = vr.next()
            frames.append(Image.fromarray(frame.asnumpy()))

        return frames 

    def _load_frames_dir(self, d: Path):
        imgs = sorted(glob.glob(str(d / "*.[jp][pn]g")))
        idxs = get_frame_ids(0, len(imgs), num_segments=self.num_frames, jitter = False)
        # idx  = sorted(random.sample(range(len(imgs)), self.num_frames))
        return [Image.open(imgs[i]) for i in idxs]
    
    def _preprocess_image(self, image):
        if isinstance(image, torch.Tensor):
            assert image.ndim==3, f"{image.shape} is unsuported"
            if image.dtype != torch.float32:
                image = image.float()
            if image.max() > 1.0:
                image = torch.clamp(image, 0.0, 255.0)
                image = image / 255.0

        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            image = TF.to_tensor(image)  # ensures (C, H, W), float32, [0,1]

        elif isinstance(image, Image.Image):
            image = TF.to_tensor(image)  # ensures (C, H, W), float32, [0,1]

        else:
            raise TypeError(f"[preprocess_image] Unsupported image type: {type(image)}")
        
        return image.permute(1, 2, 0) if image.shape[-1] !=3 else image # C H W -> H W C


    def __getitem__(self, idx):
        itm  = self.items[idx]
        path = Path(itm["video"])

        frames = ( self._load_frames_video(path)
                    if path.suffix.lower() in [".mp4", ".webm"]
                    else self._load_frames_dir(path) )

        frames = [self._preprocess_image(f) for f in frames]  # (H,W,C), float32, [0,1]
        frames = torch.stack(frames, dim=0)         # → (T, H, W, C)
        try:
            frames = self.tf(frames)
        except Exception as e:
            print(e)
            print(frames.shape)

        return {
            "video": itm["video"],
            "caption": itm["caption"],
            "frames": frames
        }


    def __len__(self): return len(self.items)
# ------------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego_json", required=True)
    ap.add_argument("--exo_json", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_frames", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="pair_result.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg = load_lavila_ckpt(args.ckpt, args.num_frames, device)
    transform  = build_val_transform(cfg)

    ego_ds = VideoPairDataset(args.ego_json, args.num_frames, transform)
    exo_ds = VideoPairDataset(args.exo_json, args.num_frames, transform)

    ego_dl = DataLoader(ego_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    exo_dl = DataLoader(exo_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    if len(ego_ds) <= len(exo_ds):
        query_dl, query_tag = ego_dl, "ego"
        db_dl, db_tag = exo_dl, "exo"
    else:
        query_dl, query_tag = exo_dl, "exo"
        db_dl, db_tag = ego_dl, "ego"

    @torch.no_grad()
    def encode(dl, tag):
        feats, paths, caption, text_feats = [], [], [], []
        for b in tqdm(dl, desc=f"encode-{tag}"):
            img = b["frames"].to(device)
            txt = tokenizer(list(b["caption"])).to(device)
            v   = model.encode_image(img)
            t   = model.encode_text(txt)
            feats.append(torch.cat([v, t], dim=-1).cpu())
            text_feats.append(t.cpu())
            paths.extend(b["video"])
            caption.extend(b["caption"])
        return torch.cat(feats), torch.cat(text_feats), paths, caption
    db_feats, db_text_feats, db_paths, db_caption = encode(db_dl, db_tag)
    query_feats, query_text_feats, query_paths, query_caption = encode(query_dl, query_tag)

    query_feats = F.normalize(query_feats, dim=-1)
    db_feats    = F.normalize(db_feats, dim=-1)
    query_text_feats = F.normalize(query_text_feats, dim = -1)
    db_text_feats = F.normalize(db_text_feats, dim = -1)
    sim = query_feats @ db_feats.T
    text_sim = query_text_feats@db_text_feats.T

    if query_tag == "ego":
        mapping = {
            query_paths[i]: {
                "pseudo_pair": db_paths[sim[i].argmax().item()],
                "ego_caption": query_caption[i],
                "exo_caption": db_caption[sim[i].argmax().item()],
                "similarity": round(text_sim[i].max().item(), 4)
            }
            for i in range(len(query_paths))
        }
    else:
        mapping = {
            db_paths[sim[i].argmax().item()]: {
                "pseudo_pair": query_paths[i],
                "ego_caption": db_caption[sim[i].argmax().item()],
                "exo_caption": query_caption[i],
                "similarity": round(text_sim[i].max().item(), 4)
            }
            for i in range(len(query_paths))
        }


    with open(args.out, "w") as f:
        json.dump(mapping, f, indent=2)
    print("✓ pair_result saved to", args.out)

if __name__ == "__main__":
    main()
