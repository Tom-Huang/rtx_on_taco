import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn.functional import cross_entropy

from hulc2.models.decoders.action_decoder import ActionDecoder
from hulc2.utils.distributions import State
from hulc2.utils.split_dataset import get_split_sequences, get_start_end_ids

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class Hulc2(pl.LightningModule):
    """
    The lightning module used for training.

    Args:
        perceptual_encoder: DictConfig for perceptual_encoder.
        plan_proposal: DictConfig for plan_proposal network.
        plan_recognition: DictConfig for plan_recognition network.
        language_encoder: DictConfig for language_encoder.
        language_goal: DictConfig for language_goal encoder.
        visual_goal: DictConfig for visual_goal encoder.
        action_decoder: DictConfig for action_decoder.
        kl_beta: Weight for KL loss term.
        kl_balancing_mix: Weight for KL balancing (as in https://arxiv.org/pdf/2010.02193.pdf).
        optimizer: DictConfig for optimizer.
        lr_scheduler: DictConfig for learning rate scheduler.
        distribution: DictConfig for plan distribution (continuous or discrete).
        use_clip_auxiliary_loss: If True, use CLIP contrastive auxiliary loss.
        clip_auxiliary_loss_beta: Weight for CLIP contrastive loss.
        replan_freq: After how many steps generate new plan (only for inference).
        proj_vis_lang: DictConfig for projection network for CLIP contrastive loss.
    """

    def __init__(
        self,
        perceptual_encoder: DictConfig,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        language_encoder: DictConfig,
        language_goal: DictConfig,
        visual_goal: DictConfig,
        action_decoder: DictConfig,
        kl_beta: float,
        kl_balancing_mix: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        distribution: DictConfig,
        use_clip_auxiliary_loss: bool,
        clip_auxiliary_loss_beta: float,
        replan_freq: int = 30,
        proj_vis_lang: Optional[DictConfig] = None,
    ):
        super(Hulc2, self).__init__()

        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder, device=self.device)
        self.setup_input_sizes(
            self.perceptual_encoder,
            plan_proposal,
            plan_recognition,
            visual_goal,
            action_decoder,
            distribution,
        )
        # plan networks
        self.dist = hydra.utils.instantiate(distribution)
        self.plan_proposal = hydra.utils.instantiate(plan_proposal, dist=self.dist)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition, dist=self.dist)

        # goal encoders
        self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.lang_encoder = hydra.utils.instantiate(language_encoder) if language_encoder else None
        self.language_goal = (
            hydra.utils.instantiate(language_goal, lang_net=self.lang_encoder) if language_goal else None
        )
        # policy network
        self.action_decoder: ActionDecoder = hydra.utils.instantiate(action_decoder)

        self.remove_modules()

        # auxiliary losses
        self.use_clip_auxiliary_loss = use_clip_auxiliary_loss
        self.clip_auxiliary_loss_beta = clip_auxiliary_loss_beta
        if use_clip_auxiliary_loss:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.proj_vis_lang = hydra.utils.instantiate(proj_vis_lang)

        self.kl_beta = kl_beta
        self.kl_balancing_mix = kl_balancing_mix
        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()

        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None

        # for clip loss
        if self.use_clip_auxiliary_loss:
            self.encoded_lang_train: Optional[torch.Tensor] = None
            self.encoded_lang_val: Optional[torch.Tensor] = None
            self.train_lang_ann: Optional[torch.Tensor] = None
            self.lang_data_val = None
            self.task_to_id: Optional[Dict] = None
            self.val_dataset = None
            self.train_lang_task_ids: Optional[np.ndarray] = None
            self.val_lang_ann: Optional[torch.Tensor] = None
            self.val_lang_task_ids: Optional[np.ndarray] = None

    def remove_modules(self):
        pass

    @staticmethod
    def setup_input_sizes(
        perceptual_encoder,
        plan_proposal,
        plan_recognition,
        visual_goal,
        action_decoder,
        distribution,
    ):
        """
        Configure the input feature sizes of the respective parts of the network.

        Args:
            perceptual_encoder: DictConfig for perceptual encoder.
            plan_proposal: DictConfig for plan proposal network.
            plan_recognition: DictConfig for plan recognition network.
            visual_goal: DictConfig for visual goal encoder.
            action_decoder: DictConfig for action decoder network.
            distribution: DictConfig for plan distribution (continuous or discrete).
        """
        plan_proposal.perceptual_features = perceptual_encoder.latent_size
        plan_recognition.in_features = perceptual_encoder.latent_size
        visual_goal.in_features = perceptual_encoder.latent_size
        action_decoder.perceptual_features = perceptual_encoder.latent_size

        if distribution.dist == "discrete":
            plan_proposal.plan_features = distribution.class_size * distribution.category_size
            plan_recognition.plan_features = distribution.class_size * distribution.category_size
            action_decoder.plan_features = distribution.class_size * distribution.category_size
        elif distribution.dist == "continuous":
            plan_proposal.plan_features = distribution.plan_features
            plan_recognition.plan_features = distribution.plan_features
            action_decoder.plan_features = distribution.plan_features

    @property
    def num_training_steps(self) -> int:
        return int(self.trainer.estimated_stepping_batches)  # type: ignore

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        """
        Set up warmup steps for learning rate scheduler.

        Args:
            num_training_steps: Number of training steps, if < 0 infer from class attribute.
            num_warmup_steps: Either as absolute number of steps or as percentage of training steps.

        Returns:
            num_training_steps: Number of training steps for learning rate scheduler.
            num_warmup_steps: Number of warmup steps for learning rate scheduler.
        """
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        num_warmup_steps = int(num_warmup_steps)
        return num_training_steps, num_warmup_steps

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        if "num_warmup_steps" in self.lr_scheduler:
            self.lr_scheduler.num_training_steps, self.lr_scheduler.num_warmup_steps = self.compute_warmup(
                num_training_steps=self.lr_scheduler.num_training_steps,
                num_warmup_steps=self.lr_scheduler.num_warmup_steps,
            )
            rank_zero_info(f"Inferring number of training steps, set to {self.lr_scheduler.num_training_steps}")
            rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.lr_scheduler.num_warmup_steps}")
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def lmp_train(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, train_acts: torch.Tensor, robot_obs: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.distributions.Distribution,
        torch.distributions.Distribution,
        NamedTuple,
    ]:
        """
        Main forward pass for training step after encoding raw inputs.

        Args:
            perceptual_emb: Encoded input modalities.
            latent_goal: Goal embedding (visual or language goal).
            train_acts: Ground truth actions.
            robot_obs: Unnormalized proprioceptive state (only used for world to tcp frame conversion in decoder).

        Returns:
            kl_loss: KL loss
            action_loss: Behavior cloning action loss.
            total_loss: Sum of kl_loss and action_loss.
            pp_dist: Plan proposal distribution.
            pr_dist: Plan recognition distribution
            seq_feat: Features of plan recognition network before distribution.
        """
        # ------------Plan Proposal------------ #
        pp_state = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
        pp_dist = self.dist.get_dist(pp_state)

        # ------------Plan Recognition------------ #
        pr_state, seq_feat = self.plan_recognition(perceptual_emb)
        pr_dist = self.dist.get_dist(pr_state)

        sampled_plan = pr_dist.rsample()  # sample from recognition net
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)

        action_loss = self.action_decoder.loss(
            sampled_plan, perceptual_emb, latent_goal, train_acts, robot_obs
        )  # type:  ignore
        kl_loss = self.compute_kl_loss(pp_state, pr_state)
        total_loss = action_loss + kl_loss

        return kl_loss, action_loss, total_loss, pp_dist, pr_dist, seq_feat

    def lmp_val(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor, robot_obs: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        NamedTuple,
    ]:
        """
        Main forward pass for validation step after encoding raw inputs.

        Args:
            perceptual_emb: Encoded input modalities.
            latent_goal: Goal embedding (visual or language goal).
            actions: Groundtruth actions.
            robot_obs: Unnormalized proprioceptive state (only used for world to tcp frame conversion in decoder).

        Returns:
            sampled_plan_pp: Plan sampled from plan proposal network.
            action_loss_pp: Behavior cloning action loss computed with plan proposal network.
            sampled_plan_pr: Plan sampled from plan recognition network.
            action_loss_pr: Behavior cloning action loss computed with plan recognition network.
            kl_loss: KL loss
            mae_pp: Mean absolute error (L1) of action sampled with input from plan proposal network w.r.t ground truth.
            mae_pr: Mean absolute error of action sampled with input from plan recognition network w.r.t ground truth.
            gripper_sr_pp: Success rate of binary gripper action sampled with input from plan proposal network.
            gripper_sr_pr: Success rate of binary gripper action sampled with input from plan recognition network.
            seq_feat: Features of plan recognition network before distribution.
        """
        # ------------Plan Proposal------------ #
        pp_state = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each
        # print("perceptual emb nan num: ", torch.sum(torch.isnan(perceptual_emb[:, 0])))
        # print("latent goal nan num: ", torch.sum(torch.isnan(latent_goal)))
        pp_dist = self.dist.get_dist(pp_state)

        # ------------ Policy network ------------ #
        sampled_plan_pp = self.dist.sample_latent_plan(pp_dist)  # sample from proposal net
        action_loss_pp, sample_act_pp = self.action_decoder.loss_and_act(  # type:  ignore
            sampled_plan_pp, perceptual_emb, latent_goal, actions, robot_obs
        )

        mae_pp = torch.nn.functional.l1_loss(
            sample_act_pp[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pp = torch.mean(mae_pp, 1)  # (batch, 6)
        # gripper action
        gripper_discrete_pp = sample_act_pp[..., -1]
        gt_gripper_act = actions[..., -1]
        m = gripper_discrete_pp > 0
        gripper_discrete_pp[m] = 1
        gripper_discrete_pp[~m] = -1
        gripper_sr_pp = torch.mean((gt_gripper_act == gripper_discrete_pp).float())

        # ------------Plan Recognition------------ #
        # print("percept emb nan num: ", torch.sum(torch.isnan(perceptual_emb)))
        pr_state, seq_feat = self.plan_recognition(perceptual_emb)
        # print("pattern recognition mean nan num: ", torch.sum(torch.isnan(pr_state.mean)))
        # print("pattern recognition std nan num: ", torch.sum(torch.isnan(pr_state.std)))
        pr_dist = self.dist.get_dist(pr_state)
        sampled_plan_pr = self.dist.sample_latent_plan(pr_dist)  # sample from recognition net
        action_loss_pr, sample_act_pr = self.action_decoder.loss_and_act(  # type:  ignore
            sampled_plan_pr, perceptual_emb, latent_goal, actions, robot_obs
        )
        mae_pr = torch.nn.functional.l1_loss(
            sample_act_pr[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pr = torch.mean(mae_pr, 1)  # (batch, 6)
        kl_loss = self.compute_kl_loss(pp_state, pr_state)
        # gripper action
        gripper_discrete_pr = sample_act_pr[..., -1]
        m = gripper_discrete_pr > 0
        gripper_discrete_pr[m] = 1
        gripper_discrete_pr[~m] = -1
        gripper_sr_pr = torch.mean((gt_gripper_act == gripper_discrete_pr).float())

        return (
            sampled_plan_pp,
            action_loss_pp,
            sampled_plan_pr,
            action_loss_pr,
            kl_loss,
            mae_pp,
            mae_pr,
            gripper_sr_pp,
            gripper_sr_pr,
            seq_feat,
        )

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.


        Returns:
            loss tensor
        """
        kl_loss, action_loss, proprio_loss, lang_pred_loss, lang_contrastive_loss, lang_clip_loss, total_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        encoders_dict = {}
        batch_size: Dict[str, int] = {}
        total_bs = 0
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
            else:
                latent_goal = self.visual_goal(perceptual_emb[:, -1])
            kl, act_loss, mod_loss, pp_dist, pr_dist, seq_feat = self.lmp_train(
                perceptual_emb, latent_goal, dataset_batch["actions"], dataset_batch["state_info"]["robot_obs"]
            )
            if "lang" in self.modality_scope:
                if not torch.any(dataset_batch["use_for_aux_lang_loss"]):
                    batch_size["aux_lang"] = 1
                else:
                    batch_size["aux_lang"] = torch.sum(dataset_batch["use_for_aux_lang_loss"]).detach()  # type:ignore
                    if self.use_clip_auxiliary_loss:
                        lang_clip_loss += self.clip_auxiliary_loss(
                            seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                        )
            encoders_dict[self.modality_scope] = [pp_dist, pr_dist]
            kl_loss += kl
            action_loss += act_loss
            total_loss += mod_loss
            batch_size[self.modality_scope] = dataset_batch["actions"].shape[0]
            total_bs += dataset_batch["actions"].shape[0]
            self.log(
                f"train/kl_loss_scaled_{self.modality_scope}",
                kl,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"train/action_loss_{self.modality_scope}",
                act_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"train/total_loss_{self.modality_scope}",
                mod_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
        kl_loss = kl_loss / len(batch)
        action_loss = action_loss / len(batch)
        if self.use_clip_auxiliary_loss:
            total_loss = total_loss + self.clip_auxiliary_loss_beta * lang_clip_loss
            self.log(
                "train/lang_clip_loss",
                self.clip_auxiliary_loss_beta * lang_clip_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size["aux_lang"],
                sync_dist=True,
            )
        self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, batch_size=total_bs)
        return total_loss

    def compute_kl_loss(self, pp_state: State, pr_state: State) -> torch.Tensor:
        """
        Compute the KL divergence loss between the distributions of the plan recognition and plan proposal network.
        We use KL balancing similar to "MASTERING ATARI WITH DISCRETE WORLD MODELS" by Hafner et al.
        (https://arxiv.org/pdf/2010.02193.pdf)

        Args:
            pp_state: Namedtuple containing the parameters of the distribution produced by plan proposal network.
            pr_state: Namedtuple containing the parameters of the distribution produced by plan recognition network.

        Returns:
            Scaled KL loss.
        """
        pp_dist = self.dist.get_dist(pp_state)  # prior
        pr_dist = self.dist.get_dist(pr_state)  # posterior
        # @fixme: do this more elegantly
        kl_lhs = D.kl_divergence(self.dist.get_dist(self.dist.detach_state(pr_state)), pp_dist).mean()
        kl_rhs = D.kl_divergence(pr_dist, self.dist.get_dist(self.dist.detach_state(pp_state))).mean()

        alpha = self.kl_balancing_mix
        kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled

    def set_kl_beta(self, kl_beta):
        """Set kl_beta from Callback"""
        self.kl_beta = kl_beta

    def clip_auxiliary_loss(self, seq_vis_feat, encoded_lang, use_for_aux_loss):
        """
        CLIP style contrastive loss, adapted from 'Learning transferable visual models from natural language
        supervision' by Radford et al.
        We maximize the cosine similarity between the visual features of the sequence i and the corresponding language
        features while, at the same time, minimizing the cosine similarity between the current visual features and other
        language instructions in the same batch.

        Args:
            seq_vis_feat: Visual embedding.
            encoded_lang: Language goal embedding.
            use_for_aux_loss: Mask of which sequences in the batch to consider for auxiliary loss.

        Returns:
            Contrastive loss.
        """
        assert self.use_clip_auxiliary_loss is not None
        if use_for_aux_loss is not None:
            if not torch.any(use_for_aux_loss):
                return torch.tensor(0.0).to(self.device)
            seq_vis_feat = seq_vis_feat[use_for_aux_loss]
            encoded_lang = encoded_lang[use_for_aux_loss]
        image_features, lang_features = self.proj_vis_lang(seq_vis_feat, encoded_lang)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # symmetric loss function
        labels = torch.arange(logits_per_image.shape[0], device=text_features.device)
        loss_i = cross_entropy(logits_per_image, labels)
        loss_t = cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal networks, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
            else:
                latent_goal = self.visual_goal(perceptual_emb[:, -1])

            (
                sampled_plan_pp,
                action_loss_pp,
                sampled_plan_pr,
                action_loss_pr,
                kl_loss,
                mae_pp,
                mae_pr,
                gripper_sr_pp,
                gripper_sr_pr,
                seq_feat,
            ) = self.lmp_val(
                perceptual_emb, latent_goal, dataset_batch["actions"], dataset_batch["state_info"]["robot_obs"]
            )
            if "lang" in self.modality_scope:
                if self.use_clip_auxiliary_loss:
                    val_pred_clip_loss = self.clip_auxiliary_loss(
                        seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                    )
                    self.log("val/val_pred_clip_loss", val_pred_clip_loss, sync_dist=True)
            val_total_act_loss_pp += action_loss_pp
            pr_mae_mean = mae_pr.mean()
            pp_mae_mean = mae_pp.mean()
            pos_mae_pp = mae_pp[..., :3].mean()
            pos_mae_pr = mae_pr[..., :3].mean()
            orn_mae_pp = mae_pp[..., 3:6].mean()
            orn_mae_pr = mae_pr[..., 3:6].mean()
            self.log(f"val_total_mae/{self.modality_scope}_total_mae_pr", pr_mae_mean, sync_dist=True)
            self.log(f"val_total_mae/{self.modality_scope}_total_mae_pp", pp_mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{self.modality_scope}_pos_mae_pr", pos_mae_pr, sync_dist=True)
            self.log(f"val_pos_mae/{self.modality_scope}_pos_mae_pp", pos_mae_pp, sync_dist=True)
            self.log(f"val_orn_mae/{self.modality_scope}_orn_mae_pr", orn_mae_pr, sync_dist=True)
            self.log(f"val_orn_mae/{self.modality_scope}_orn_mae_pp", orn_mae_pp, sync_dist=True)
            self.log(f"val_kl/{self.modality_scope}_kl_loss", kl_loss, sync_dist=True)
            self.log(f"val_act/{self.modality_scope}_act_loss_pp", action_loss_pp, sync_dist=True)
            self.log(f"val_act/{self.modality_scope}_act_loss_pr", action_loss_pr, sync_dist=True)
            self.log(f"val_grip/{self.modality_scope}_grip_sr_pr", gripper_sr_pr, sync_dist=True)
            self.log(f"val_grip/{self.modality_scope}_grip_sr_pp", gripper_sr_pp, sync_dist=True)
            self.log(
                "val_act/action_loss_pp",
                val_total_act_loss_pp / len(self.trainer.datamodule.modalities),  # type:ignore
                sync_dist=True,
            )
            output[f"sampled_plan_pp_{self.modality_scope}"] = sampled_plan_pp
            output[f"sampled_plan_pr_{self.modality_scope}"] = sampled_plan_pr
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]

        return output

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def step(self, obs, goal):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        # replan every replan_freq steps (default 30 i.e every second)
        if self.rollout_step_counter % self.replan_freq == 0:
            if "lang" in goal:
                self.plan, self.latent_goal = self.get_pp_plan_lang(obs, goal)
            else:
                self.plan, self.latent_goal = self.get_pp_plan_vision(obs, goal)
        # use plan to predict actions with current observations
        action = self.predict_with_plan(obs, self.latent_goal, self.plan)
        self.rollout_step_counter += 1
        return action

    def predict_with_plan(
        self,
        obs: Dict[str, Any],
        latent_goal: torch.Tensor,
        sampled_plan: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass observation, goal and plan through decoder to get predicted action.

        Args:
            obs: Observation from environment.
            latent_goal: Encoded goal.
            sampled_plan: Sampled plan proposal plan.

        Returns:
            Predicted action.
        """
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            action = self.action_decoder.act(
                sampled_plan, perceptual_emb, latent_goal, obs["robot_obs_raw"]
            )  # type:  ignore

        return action

    def get_pp_plan_vision(self, obs: dict, goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use plan proposal network to sample new plan using a visual goal embedding.

        Args:
            obs: Observation from environment.
            goal: Goal observation (vision & proprioception).

        Returns:
            sampled_plan: Sampled plan.
            latent_goal: Encoded visual goal.
        """
        assert len(obs["rgb_obs"]) == len(goal["rgb_obs"])
        # assert len(obs["depth_obs"]) == len(goal["depth_obs"])
        imgs = {k: torch.cat([v, goal["rgb_obs"][k]], dim=1) for k, v in obs["rgb_obs"].items()}  # (1, 2, C, H, W)
        state = None  # type:ignore
        depth_imgs = {}  # type:ignore
        for k in obs.keys():
            if "depth" in k:
                depth_imgs = {k: torch.cat([v, goal["depth_obs"][k]], dim=1) for k, v in obs["depth_obs"].items()}
            if "robot_obs" in k:
                state = torch.cat([obs["robot_obs"], goal["robot_obs"]], dim=1)
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(imgs, depth_imgs, state)
            latent_goal = self.visual_goal(perceptual_emb[:, -1])
            # ------------Plan Proposal------------ #
            pp_state = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
            pp_dist = self.dist.get_dist(pp_state)
            sampled_plan = self.dist.sample_latent_plan(pp_dist)
        self.action_decoder.clear_hidden_state()
        return sampled_plan, latent_goal

    def get_pp_plan_lang(self, obs: dict, goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use plan proposal network to sample new plan using a visual goal embedding.

        Args:
            obs: Observation from environment.
            goal: Embedded language instruction.

        Returns:
            sampled_plan: Sampled plan.
            latent_goal: Encoded language goal.
        """
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            latent_goal = self.language_goal(goal["lang"])
            # ------------Plan Proposal------------ #
            pp_state = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
            pp_dist = self.dist.get_dist(pp_state)
            sampled_plan = self.dist.sample_latent_plan(pp_dist)
        self.action_decoder.clear_hidden_state()
        return sampled_plan, latent_goal

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")
