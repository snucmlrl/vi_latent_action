from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer

from latent_action_model.genie.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer


from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class DINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(DINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )


    def vq_encode(self, videos: Tensor, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dino_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dino_features = rearrange(dino_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dino_features], dim=2)

        # Encode
        z = self.encoder(padded_patches, attention_mask) 
      
        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": dino_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
        }
    
    def forward(self, batch:Dict) -> Dict:
        """
        outputs = {
            "primary": {
                "patches": torch.Tensor,
                "z": torch.Tensor,
                ....
            },
            "secondary": {
                "patches": torch.Tensor,
                "z": torch.Tensor,
                ...
            },
            "wrist": {
                "patches": torch.Tensor,
                "z": torch.Tensor,
                ...
            }
        }
        
        """
        outputs = dict()
        action_patches_dict = dict()
        video_patches_dict = dict()
        video_target_dict = dict()

        for view, batch_per_view in batch.items():
            outputs = self.vq_encode(batch_per_view["videos"])
            video_pathces = self.patch_up(outputs["patches"][:,:-1]) 
            action_patches = self.action_up(outputs["z_q"])
            action_patches_dict[view] = action_patches
            video_patches_dict[view] = video_pathces
            video_target_dict[view] = outputs["patches"][:, [-1]]
        
        for obs_view in batch.keys():
            for act_view in batch.keys():
                video_patches = video_patches_dict[obs_view]
                action_patches = action_patches_dict[act_view]
                video_action_patches = torch.cat([action_patches, video_patches], dim = 2)
                video_recon = self.decoder(video_action_patches)
                video_recon = video_recon[:, :, -video_patches.shape[2]:]
                outputs[obs_view][act_view] = {
                    "recon":video_recon,
                    "target":video_target_dict[obs_view]
                }
            
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
