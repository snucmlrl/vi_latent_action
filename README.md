# Dataset Installation


# Dataset Preparation
``` {bash}
conda env create -f vla-scripts/extern/ego4d_rlds_dataset_builder/environment_ubuntu.yml
conda activate rlds_env
```

## Pseudo-paired Dataset
``` {bash}
git clone https://github.com/facebookresearch/LaViLa.git
```

```
python build_meta.py --root DATASET_NAME --out PATH_TO_META_JSON
```

``` {bash}
python preprocess_dataset/build_pair.py --ego_json data/sth_sth_v2/metadata.jsonl --exo_json data/bridge_data_v2/metadata.jsonl --ckpt data/ckpt/lavila/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth --num
_frames 8 --out data/pseudo_pair/metadata.jsonl --bs 128
```

```{bash}
python preprocess_dataset/build_pseudo_pair_to_rlds.py --json /data3/pseudo_pair/metadata.jsonl --data_dir /data3/pseudo_pair
```

## Ego-Exo4D
```{bash}
pip install awscli ego4d
aws configure
egoexo -o /data3/egoexo4d/ --parts annotations metadata downscaled_takes/448
```

```{bash}
cd preprocess_dataset/egoexo4d_build
```

Preparing info_clips.json
```{bash}
python prepare_dataset.py --root_path /home/robot/vi_latent_action/data/egoexo4d --only_essential
```

Converting to .npy
```{bash}
python preprocess_egoexo4d.py --denseclips_dir /home/robot/vi_latent_action/data/egoexo4d/clips_jpgs/processed --info_clips_json /home/robot/vi_latent_action/data/egoexo4d/clips_jpgs/processed/info_clips.json --source_videos_dir /home/robot/vi_latent_action/data/egoexo4d/takes --processes 32
```

```{bash}
bash tfds_build.sh
```

``` {bash}
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.0+cu121 torchvision==0.17.0+cu121
```

# LAM training
``` {bash}
cd latent_action_model
pip install -U jsonargparse[signatures]>=4.27.7
```

``` {bash}
python - <<'PY'                                                                
import torch
torch.hub.set_dir('/home/robot/.cache/torch/hub')
m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', trust_repo=True)
print('DINOv2 ready:', type(m))
PY
```

``` {bash}
bash train.sh
```

