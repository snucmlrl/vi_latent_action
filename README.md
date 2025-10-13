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
python prepare_dataset.py --root /home/robot/vi_latent_action/data/egoexo4d 
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

# Latent Action Pretraining
``` {bash}
cd vla_scripts
```

``` {bash}
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='timm/vit_large_patch14_reg4_dinov2.lvd142m', repo_type='model', local_dir='../data/ckpt/vit_large_patch14_reg4_dinov2.lvd142m', local_dir_use_symlinks=False, allow_patterns=['model.safetensors','pytorch_model.bin'])"
```

``` {bash}
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='timm/ViT-SO400M-14-SigLIP', repo_type='model', local_dir='../data/ckpt/ViT-SO400M-14-SigLIP', local_dir_use_symlinks=False, allow_patterns=['open_clip_model.safetensors','open_clip_pytorch_model.bin','model.safetensors','pytorch_model.bin'])"
```

``` {bash}
bash train.sh
```


``` {bash}
python extern/convert_univla_weights_to_hf.py
```

``` {bash}
pip install flash-attn==2.5.6
```

# LIBERO finetuning
``` {bash}
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash finetune_libero.sh
``` 

# LIBERO eval setting
``` {bash}
conda create -n libero python=3.10 -y
conda activate libero
python -m pip install   --extra-index-url https://download.pytorch.org/whl/cu121   torch==2.2.0+cu121 torchvision==0.17.0+cu121
pip install tensorflow==2.15.0
pip install draccus
pip install transformers==4.40.1
pip install opencv_python==4.10.0.84
pip install numpy==1.26.4
pip install hydra-core==1.3.2
pip install imageio[ffmpeg]
pip install robosuite==1.4.1
pip install rich==14.0.0
pip install timm==0.9.10
pip install tensorflow_graphics==2021.12.3
pip install braceexpand==0.1.7
pip install webdataset==0.2.111
pip install accelerate==0.32.1

ln -s /home/robot/vi_latent_action/LIBERO/libero       /home/robot/vi_latent_action/experiments/robot/libero/libero
```