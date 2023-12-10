from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
import itertools

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

@threestudio.register("kplanes")
class Kplanes(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        grid_size: Tuple[int, int, int, int] = field(default_factory=lambda: (100, 100, 100, 16))
        hybrid: bool = True
        multi_scale_res: int = 2
        n_input_dims: int = 4
        n_grid_dims: int = 16
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob_magic3d"
        density_blob_scale: float = 10.0
        density_blob_std: float = 0.5
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 32,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference']
        finite_difference_normal_eps: float = 0.01

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = 25.0

    cfg: Config

    def configure(self) -> None:
        super().configure()
        
        # init grids
        multi_scale_res=[2**i for i in range(self.cfg.multi_scale_res)]
        self.grids = nn.ModuleList()
        self.n_hidden_dims = 0
        for idx, res in enumerate(multi_scale_res):
            resolution = [r * res for r in self.cfg.grid_size[:3]] + [self.cfg.grid_size[3+idx]]
           
            gp = self.init_grid_param(
                grid_nd=2,
                in_dim=self.cfg.n_input_dims,
                out_dim=self.cfg.n_grid_dims,
                reso=resolution,
            )
            self.n_hidden_dims += gp[-1].shape[1]
            self.grids.append(gp)

        self.density_network = get_mlp(
            self.n_hidden_dims, 1, self.cfg.mlp_network_config
        )
        self.feature_network = get_mlp(
            self.n_hidden_dims,
            self.cfg.n_feature_dims,
            self.cfg.mlp_network_config,
        )
        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )
        
        # # zero_init
        # for param in self.density_network.parameters():
        #     nn.init.zeros_(param)

    def init_grid_param(
        self,
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
        assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
        has_time_planes = in_dim == 4
        assert grid_nd <= in_dim
        coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        grid_coefs = nn.ParameterList()
        for ci, coo_comb in enumerate(coo_combs):
            new_grid_coef = nn.Parameter(torch.empty(
                [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
            ))
            if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
                nn.init.ones_(new_grid_coef)
            else:
                nn.init.uniform_(new_grid_coef, a=a, b=b)
            grid_coefs.append(new_grid_coef)

        return grid_coefs

    def get_activated_density(
        self, points: Float[Tensor, "*N Di"], density: Float[Tensor, "*N 1"]
    ) -> Tuple[Float[Tensor, "*N 1"], Float[Tensor, "*N 1"]]:
        density_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.density_bias == "blob_dreamfusion":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * torch.exp(
                    -0.5 * (points**2).sum(dim=-1) / self.cfg.density_blob_std**2
                )[..., None]
            )
        elif self.cfg.density_bias == "blob_magic3d":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * (
                    1
                    - torch.sqrt((points**2).sum(dim=-1)) / self.cfg.density_blob_std
                )[..., None]
            )
        elif isinstance(self.cfg.density_bias, float):
            density_bias = self.cfg.density_bias
        else:
            raise ValueError(f"Unknown density bias {self.cfg.density_bias}")
        raw_density: Float[Tensor, "*N 1"] = density + density_bias
        density = get_activation(self.cfg.density_activation)(raw_density)
        return raw_density, density

    def interpolate_ms_features(self, pts: torch.Tensor, num_levels: Optional[int]=None,) -> torch.Tensor:
        
        ms_grids = self.grids
        grid_dimensions = 2
        concat_features = True

        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]), grid_dimensions)
        )
        if num_levels is None:
            num_levels = len(ms_grids)
        multi_scale_interp = [] if concat_features else 0.
        grid: nn.ParameterList
        for scale_id, grid in enumerate(ms_grids[:num_levels]):
            if self.cfg.hybrid:
                interp_space = 0.
                spatialplanes=[0, 1, 3]
                timeplanes = [2, 4, 5]
            
                for ci in spatialplanes:
                    coo_comb = coo_combs[ci]
                    feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                    interp_out_plane = (
                        grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                        .view(-1, feature_dim)
                    )
                    
                    interp_space = interp_space + interp_out_plane
                
                for ci in timeplanes:
                    coo_comb = coo_combs[ci]
                    feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                    interp_out_plane = (
                        grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                        .view(-1, feature_dim)
                    )
                    
                    interp_space = interp_space * interp_out_plane
            else:
                interp_space = 1.
                for ci, coo_comb in enumerate(coo_combs):
                    # interpolate in plane
                    feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                    interp_out_plane = (
                        grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                        .view(-1, feature_dim)
                    )
                    # compute product over planes
                    interp_space = interp_space * interp_out_plane

            # combine over scales
            if concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

        if concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp
    
    def forward(
        self, points: Float[Tensor, "*N Di"], timestamps: Float[Tensor, "*N 1"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()
        
        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        points = points * 2 - 1  # convert to [-1, 1] for grid sample

        points_input = torch.cat([points, timestamps], dim=-1) # timestamps range [-1, 1]
        out = self.interpolate_ms_features(points_input)
        density = self.density_network(out).view(*points.shape[:-1], 1)
        features = self.feature_network(out).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        raw_density, density = self.get_activated_density(points_unscaled, density)

        output = {
            "density": density,
            "features": features,
        }

        if output_normal:
            if self.cfg.normal_type == "finite_difference":
                # TODO: use raw density
                eps = self.cfg.finite_difference_normal_eps
                offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                    [
                        [eps, 0.0, 0.0],
                        [-eps, 0.0, 0.0],
                        [0.0, eps, 0.0],
                        [0.0, -eps, 0.0],
                        [0.0, 0.0, eps],
                        [0.0, 0.0, -eps],
                    ]
                ).to(points_unscaled)
                points_offset: Float[Tensor, "... 6 3"] = (
                    points_unscaled[..., None, :] + offsets
                ).clamp(-self.cfg.radius, self.cfg.radius)
               
                density_offset: Float[Tensor, "... 6 1"] = self.forward_density(
                    points_offset,
                    timestamps[..., None, :].repeat(1, 6, 1),
                )
                normal = (
                    -0.5
                    * (density_offset[..., 0::2, 0] - density_offset[..., 1::2, 0])
                    / eps
                )
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "analytic":
                normal = -torch.autograd.grad(
                    density,
                    points_unscaled,
                    grad_outputs=torch.ones_like(density),
                    create_graph=True,
                )[0]
                normal = F.normalize(normal, dim=-1)
                if not grad_enabled:
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({"normal": normal, "shading_normal": normal})

        torch.set_grad_enabled(grad_enabled)
        return output

    def forward_density(self, points: Float[Tensor, "*N Di"], timestamps: Float[Tensor, "*N 1"],) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample

        points_input = torch.cat([points, timestamps], dim=-1) # timestamps range [-1, 1]
        density = self.density_network(
            self.interpolate_ms_features(points_input.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)

        _, density = self.get_activated_density(points_unscaled, density)
        return density

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        if self.cfg.isosurface_deformable_grid:
            threestudio.warn(
                f"{self.__class__.__name__} does not support isosurface_deformable_grid. Ignoring."
            )
        density = self.forward_density(points)
        return density, None

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return -(field - threshold)

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out
