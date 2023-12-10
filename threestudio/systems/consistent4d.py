import os
import random
import shutil
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import numpy as np

@threestudio.register("consistent4d-system")
class Consistent4D(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)
        rgb_as_latents: bool = False
        load_path: str = "none"
        attn_loc: list = field(default_factory=list)
        use_vif: bool = False
        vif_pretrained_root: str = "./extern/RIFE"
        use_same_noise: bool = False
        mid_no_grad: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        if self.cfg.load_path != "none":
            state_dict = torch.load(self.cfg.load_path, map_location='cpu')["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print('missing_keys: ')
            for key in missing_keys:
                print(key)
            assert len(unexpected_keys) == 0, 'loaded ckpt have unexpected keys'

        # depth metric
        # self.pearson = PearsonCorrCoef().to(self.device)
        
        # video interpolateion module
        if self.cfg.use_vif:
            from extern.RIFE.RIFE_HDv3 import Model
            self.vif_module = Model()
            self.vif_module.load_model(self.cfg.vif_pretrained_root, -1)
            self.vif_module.flownet = self.vif_module.flownet.cuda()
            self.vif_module.eval()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # no prompt processor
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        # visualize all training images
        first_image = self.trainer.datamodule.train_dataloader().dataset.get_first_image()
        self.save_image_grid(
            "first_training_image.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in first_image
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "zero123"
        """
        if guidance == "ref":
            # bg_color = torch.rand_like(batch['rays_o'])
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
            bg_color = None
        elif guidance == "zero123":
            batch = batch["random_camera"]
            if self.cfg.rgb_as_latents:
                bg_color = None
                ambient_ratio = 0.1
            else:
                bg_color = None
                ambient_ratio = 0.1 + 0.9 * random.random()

        batch["bg_color"] = bg_color
        batch["ambient_ratio"] = ambient_ratio

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "zero123"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )
        # out["comp_rgb"] would be a list containing cascade outputs, other values remain the same as before
        if guidance == "ref":
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]
            # gt_depth = batch["depth"]

            for idx, (comp_rgb, opacity, depth) in enumerate(zip(out["comp_rgb"], out["opacity"], out["depth"])):
                if self.cfg.rgb_as_latents:
                    # color loss
                    out_rgb_BHWC = self.guidance.decode_latents(comp_rgb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                    set_loss(f"rgb_{idx}", F.mse_loss(gt_rgb, out_rgb_BHWC))

                else:
                    if gt_rgb.shape[2] != opacity.shape[2]:
                        gt_rgb = F.interpolate(gt_rgb.float().permute(0, 3, 1, 2), (opacity.shape[1], opacity.shape[1]), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

                    if gt_mask.shape[2] != opacity.shape[2]:
                        gt_mask = F.interpolate(gt_mask.float().permute(0, 3, 1, 2), (opacity.shape[1], opacity.shape[1]), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                    
                    # color loss
                    gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"][idx] * (
                        1 - gt_mask.float()
                    )
                    set_loss(f"rgb_{idx}", F.mse_loss(gt_rgb, comp_rgb))

                if gt_mask.shape[2] != opacity.shape[2]:
                    gt_mask = F.interpolate(gt_mask.float().permute(0, 3, 1, 2), (opacity.shape[1], opacity.shape[1]), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                # mask loss
                set_loss(f"mask_{idx}", F.mse_loss(gt_mask.float(), opacity))

                # depth loss
                # if self.C(self.cfg.loss.lambda_depth_0) > 0:
                    # valid_gt_depth = gt_depth[gt_mask.squeeze(-1)].unsqueeze(1)
                    # valid_pred_depth = depth[gt_mask].unsqueeze(1)
                    # with torch.no_grad():
                    #     A = torch.cat(
                    #         [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    #     )  # [B, 2]
                    #     X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    #     valid_gt_depth = A @ X  # [B, 1]
                    # set_loss(f"depth_{idx}", F.mse_loss(valid_gt_depth, valid_pred_depth))
                    # relative depth loss
                    # valid_gt_depth = gt_depth[gt_mask.squeeze(-1)]  # [B,]
                    # valid_pred_depth = depth[gt_mask]  # [B,]
                    # set_loss(f"depth_{idx}", 1 - self.pearson(valid_pred_depth, valid_gt_depth))

        elif guidance == "zero123":
            # video interpolation
            # tmp_flag = random.random() > 0.5
            if self.cfg.use_vif and batch["mode"] > 1 and out["comp_rgb"].shape[1] > 2:

                for idx, comp_rgb in enumerate(out["comp_rgb"]):
                    comp_rgb_BCHW = comp_rgb.permute(0, 3, 1, 2)
                    img0 = comp_rgb_BCHW[0:1]
                    img1 = comp_rgb_BCHW[-1:]
                    gt = comp_rgb_BCHW[1:-1]
                    mid_feat = []
                    # loss_mid = 0.
                    for mid_idx in range(gt.shape[0]):
                        if self.cfg.mid_no_grad:
                            with torch.no_grad():
                                mid = self.vif_module.inference(img0, img1, (mid_idx + 1) * 1.0 / (comp_rgb_BCHW.shape[0]-1))
                        else:
                            mid = self.vif_module.inference(img0, img1, (mid_idx + 1) * 1.0 / (comp_rgb_BCHW.shape[0]-1))
                        mid_feat.append(mid)
                    mid_feat = torch.cat(mid_feat, dim=0)
                    set_loss(f"vif_{idx}", F.mse_loss(mid_feat, gt))

            # origin_max_step = self.guidance.max_step
            self.guidance.set_guidance_scale(
                self.C(self.guidance.cfg.guidance_scale)
            )
            noise = None
            for idx, comp_rgb in enumerate(out["comp_rgb"]):
                if idx == 0:
                    self.guidance.set_min_max_steps(
                    self.C(self.guidance.cfg.min_step_percent_coarse),
                    self.C(self.guidance.cfg.max_step_percent_coarse), #remove temporally
                    )
                else:
                    self.guidance.set_min_max_steps(
                    self.C(self.guidance.cfg.min_step_percent_fine),
                    self.C(self.guidance.cfg.max_step_percent_fine), #remove temporally
                    )
                
                # self.guidance.max_step = int(self.guidance.num_train_timesteps * self.guidance.cfg.max_step_percent * (1 - idx/len(out["comp_rgb"]*self.true_global_step/10000)))
                guidance_out, guidance_eval_out, noise = self.guidance(
                    comp_rgb,
                    **batch,
                    rgb_as_latents=self.cfg.rgb_as_latents,
                    guidance_eval=guidance_eval,
                    attn_loc=self.cfg.attn_loc,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                    noise=noise if self.cfg.use_same_noise else None,
                )
                # claforte: TODO: rename the loss_terms keys
                set_loss(f"sds_{idx}", guidance_out["loss_sds"])
                 
        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            set_loss(
                "orient",
                (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum()
                / (out["opacity"] > 0).sum(),
            )

        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())
        
        if guidance != "ref":
            set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        set_loss("opaque", binary_cross_entropy(opacity_clamped, opacity_clamped))
      
        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
        set_loss("z_variance", loss_z_variance)
       
        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        if guidance_eval:
            self.guidance_evaluation_save(out["comp_rgb"].detach(), guidance_eval_out)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        if self.cfg.freq.ref_or_zero123 == "accumulate":
            do_ref = True
            do_zero123 = True
        elif self.cfg.freq.ref_or_zero123 == "alternate":
            do_ref = (
                self.true_global_step < self.cfg.freq.ref_only_steps
                or self.true_global_step % self.cfg.freq.n_ref == 0
            )
            do_zero123 = not do_ref

        total_loss = 0.0
        if do_zero123:
            out = self.training_substep(batch, batch_idx, guidance="zero123")
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref")
            total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)

        # sch = self.lr_schedulers()
        # sch.step()

        return {"loss": total_loss}

    def merge12(self, x):
        return x.reshape(-1, *x.shape[2:])

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"
        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": self.merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": self.merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": self.merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": self.merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": self.merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
        )

        img = Image.open(self.get_save_path(filename))
        draw = ImageDraw.Draw(img)
        for i, n in enumerate(guidance_eval_out["noise_levels"]):
            draw.text((1, (img.size[1] // B) * i + 1), f"{n:.02f}", (255, 255, 255))
            draw.text((0, (img.size[1] // B) * i), f"{n:.02f}", (0, 0, 0))
        img.save(self.get_save_path(filename))

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if self.cfg.rgb_as_latents:
            out["comp_rgb"] = [self.guidance.decode_latents(comp_rgb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) for comp_rgb in out["comp_rgb"]]
        
        batch_images = out["opacity"].shape[1]
        # batch_images = 2 # fixed
        for image_idx in range(batch_images):
            image_id = batch_idx * batch_images + image_idx
            # print(image_id)
            self.save_image_grid(
                f"it{self.true_global_step}-val/{image_id}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][image_idx],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                # + [
                #     {
                #         "type": "rgb",
                #         "img": out["comp_rgb"][0][image_idx],
                #         "kwargs": {"data_format": "HWC"},
                #     },
                # ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][-1][image_idx],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                  )
                # + (
                #     [
                #         {
                #             "type": "rgb",
                #             "img": out["comp_normal"][0][image_idx],
                #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                #         }
                #     ]
                #     if "comp_normal" in out
                #     else []
                # )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][-1][image_idx],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + [{"type": "grayscale", "img": out["depth"][0][image_idx], "kwargs": {}}]
                # + [{"type": "grayscale", "img": out["depth"][-1][image_idx], "kwargs": {}}]
                # + [
                #     {
                #         "type": "grayscale",
                #         "img": out["opacity"][0][image_idx, :, :, 0],
                #         "kwargs": {"cmap": None, "data_range": (0, 1)},
                #     },
                # ]
                # + [
                #     {
                #         "type": "grayscale",
                #         "img": out["opacity"][-1][image_idx, :, :, 0],
                #         "kwargs": {"cmap": None, "data_range": (0, 1)},
                #     },
                # ],
                ,
                # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
                name=f"validation_step_batchidx_{batch_idx}"
                if batch_idx in [0, 7, 15, 23, 29]
                else None,
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=10,
            name="validation_epoch_end",
            step=self.true_global_step,
        )
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
        )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch_idx}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [{"type": "grayscale", "img": out["depth"][0], "kwargs": {}}]
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=10,
            name="test",
            step=self.true_global_step,
        )