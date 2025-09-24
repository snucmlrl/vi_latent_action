from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState

OptimizerCallable = Callable[[Iterable], Optimizer]

from genie.modules import DINOLatentActionModel
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



class DINO_LAM(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
    ) -> None:
        super(DINO_LAM, self).__init__()

        lam = DINOLatentActionModel

        self.lam = lam(
                    in_dim=image_channels,
                    model_dim=lam_model_dim,
                    latent_dim=lam_latent_dim,
                    num_latents=lam_num_latents,
                    patch_size=lam_patch_size,
                    enc_blocks=lam_enc_blocks,
                    dec_blocks=lam_dec_blocks,
                    num_heads=lam_num_heads,
                    dropout=lam_dropout,
                )
        
        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']

        def _loss(gt, pred, coeff):
            if coeff.ndim==1:
                coeff = coeff.view(-1, 1, 1, 1, 1)
            mse_loss = ((gt - pred['recon'])**2)*coeff
            q_loss = ((pred['emb'].detach() - pred['z'])**2)*coeff
            commit_loss = ((pred['emb'] - pred['z'].detach())**2)*coeff
            loss = mse_loss.mean() + q_loss.mean() + self.vq_beta*commit_loss.mean()

            unique, counts = torch.unique(outputs["indices"], return_counts=True)
            index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
            index_counts[unique] = counts
            code_usage = (index_counts != 0).float().mean()

            # loss_logs = (
            #     ("mse_loss", mse_loss),
            #     ("q_loss", q_loss),
            #     ("commit_loss", commit_loss),
            #     ("code_usage", code_usage),
            # )
            loss_logs = {
                "mse_loss":mse_loss,
                "q_loss":q_loss,
                "commit_loss":commit_loss,
                "code_usage":code_usage
            }
            return loss, loss_logs

        outputs = self.lam(batch)
        coeff = batch['coeff'].to(self.device, dtype = self.dtype)

        loss = 0.0
        loss_logs = []
        for obs_view in outputs.keys():
            for act_view in outputs[obs_view].keys():
                gt_future_frames = outputs[obs_view][act_view]["target"]
                pred = outputs[obs_view][act_view]["recon"]
                loss_per_view, loss_logs_per_view = _loss(gt_future_frames, pred, coeff)
                loss = loss + loss_per_view
                for k, v in loss_logs_per_view.items():
                    loss_logs.append((f"{obs_view} - {act_view}/{k}", v))
        
        # gt_future_frames = outputs["target"]

        # Compute loss
        # mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        # q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        # commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        # loss = mse_loss + q_loss + self.vq_beta * commit_loss


        
        # Optimize uncontrollable queries in stage-2 (the codebook is frozen though)
        # if "z_q_uncontrol" in outputs.keys():
        #     q_loss_uncontrol = ((outputs["emb_uncontrol"].detach() - outputs["z_uncontrol"]) ** 2).mean()
        #     commit_loss_uncontrol = ((outputs["emb_uncontrol"]- outputs["z_uncontrol"].detach()) ** 2).mean()
        #     loss = loss + q_loss_uncontrol + self.vq_beta * commit_loss_uncontrol

        # Compute code usage
        # unique, counts = torch.unique(outputs["indices"], return_counts=True)
        # index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        # index_counts[unique] = counts
        # code_usage = (index_counts != 0).float().mean()

        # loss_logs = (
        #     ("mse_loss", mse_loss),
        #     ("q_loss", q_loss),
        #     ("commit_loss", commit_loss),
        #     ("code_usage", code_usage),
        # )

        # if "indices_uncontrol" in outputs.keys():
        #     unique, counts = torch.unique(outputs["indices_uncontrol"], return_counts=True)
        #     index_counts = torch.zeros(32, dtype=torch.long).cuda()
        #     index_counts[unique] = counts
        #     uncontrol_code_usage = (index_counts != 0).float().mean()

        #     loss_logs = (
        #         ("mse_loss", mse_loss),
        #         ("q_loss", q_loss),
        #         ("commit_loss", commit_loss),
        #         ("q_loss_uncontrol", q_loss_uncontrol),
        #         ("commit_loss_uncontrol", commit_loss_uncontrol),
        #         ("code_usage", code_usage),
        #         ("code_usage_uncontrol", uncontrol_code_usage),
        #     )

        return outputs, loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)


        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the training loss
        self.log_dict(
            {**{"valid_loss": loss}, **{f"valid/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"valid_loss": loss}, **{f"valid/{k}": v for k, v in aux_losses}})

        return loss


    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def on_train_epoch_end(self):
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
