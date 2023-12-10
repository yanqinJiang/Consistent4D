from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import get_device
from threestudio.utils.ops import chunk_batch
from threestudio.utils.custom_ops import custom_chunk_batch
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from dataclasses import dataclass, field
import torch.nn as nn

@threestudio.register("dynerf-volume-renderer")
class DyNeRFVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 160000
        grid_prune: bool = True
        return_comp_normal: bool = False
        return_normal_perturb: bool = False
        output_normal: bool = False
    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=self.bbox.view(-1), resolution=32, levels=1
        )
        if not self.cfg.grid_prune:
            self.estimator.occs.fill_(True)
            self.estimator.binaries.fill_(True)
        self.render_step_size = (
            1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        )
        self.randomized = self.cfg.randomized

        self.depth = len(self.geometry.cfg.grid_size)

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        timestamps: Float[Tensor, "B 1"],
        total_images: Float[Tensor, "B 1"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        mode: int = 3,
        guidance: str = "zero123",
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]
        if not self.cfg.grid_prune:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    early_stop_eps=0,
                )
        else:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                )

        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        # TODO: still proceed if the scene is empty
        # normalize timestamps
        timestamps = torch.ones(batch_size, height, width, 1).to(positions) * timestamps[:, None, None, :]
        timestamps = timestamps.reshape(-1, 1)
        timestamps = (timestamps[ray_indices] / total_images) * 2 - 1

        # repeat
        ray_indices_list = [ray_indices + i*n_rays for i in range(self.depth)]
        ray_indices = torch.cat(ray_indices_list, dim=0)
        t_starts = t_starts.repeat(self.depth, 1)
        t_ends = t_ends.repeat(self.depth, 1)
        t_origins = t_origins.repeat(self.depth, 1)
        t_dirs = t_dirs.repeat(self.depth, 1)
        t_light_positions = t_light_positions.repeat(self.depth, 1)
        t_positions = t_positions.repeat(self.depth, 1)
        # positions = positions.repeat(self.depth, 1)
        t_intervals = t_intervals.repeat(self.depth, 1)
        n_rays = n_rays * self.depth
        rays_d_flatten = rays_d_flatten.repeat(self.depth, 1)
        
        if self.training:
            geo_out = self.geometry(
                positions, timestamps, output_normal=self.material.requires_normal or self.cfg.output_normal
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions.repeat(self.depth, 1),
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d_flatten)
        else:
            geo_out = custom_chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                self.depth,
                positions,
                timestamps,
                output_normal=self.material.requires_normal or self.cfg.output_normal,
            )
           
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions.repeat(self.depth, 1),
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d_flatten
            )

        weights: Float[Tensor, "Nr 1"]
        weights_, _, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        t_depth = depth[ray_indices]
        z_variance = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(t_positions - t_depth) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        if bg_color is None:
            bg_color = comp_rgb_bg
        else:
            if bg_color.shape == (batch_size, height, width, 3):
                bg_color = bg_color.reshape(-1, 3)
            elif bg_color.shape == (batch_size, height, width, 4):
                bg_color = bg_color.reshape(-1, 4)
       
        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)
      
        out = {
            "comp_rgb": comp_rgb.view(self.depth, batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(self.depth, batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(self.depth, batch_size, height, width, -1),
            "opacity": opacity.view(self.depth, batch_size, height, width, 1),
            "depth": depth.view(self.depth, batch_size, height, width, 1),
            "z_variance": z_variance.view(self.depth, batch_size, height, width, 1),
        }

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                self.depth, batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        timestamps,
                        output_normal=self.material.requires_normal or self.cfg.output_normal,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb.reshape(-1, 3)})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(self.depth, batch_size, height, width, 3),
                    }
                )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.grid_prune:
            # The logic of occupancy network for dynamic nerf is adapted
            # from https://github.com/liruilong940607/kplanes/blob/40e799039fd073e06329cea54d038a367d13b2fe/plenoxels/models/lowrank_model.py#L184
            def occ_eval_fn(x):
                density = self.geometry.forward_density(x, timestamps=torch.rand_like(x[:, 0:1]) * 2 - 1, occ_eval=True)
                # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                return density * self.render_step_size
            
            if self.training and not on_load_weights:
                self.estimator.update_every_n_steps(
                    step=global_step, occ_eval_fn=occ_eval_fn
                )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()
