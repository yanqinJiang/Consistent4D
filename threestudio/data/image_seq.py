import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register

from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

import random
from threestudio.data.uncond_t import (
    HybridCameraDataModuleConfig,
    HybridCameraIterableDataset,
    HybridCameraDataset
)

from torch.utils.data import default_collate
from threestudio.utils.base import Updateable

@dataclass
class ImageSeqDataModuleConfig:
    height: int = 96
    width: int = 96
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_seq_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: Any = 2
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_timestamp: list = field(default_factory=list)
    eval_pose_index: list = field(default_factory=list)
    rgb_as_latents: bool = False
    mode_section: list = field(default_factory=list)
    eval_batch_size: int = 8
    time_interval: float = 1/16

class ImageSeqDataBase:
    def setup(self, cfg, split, size_ind=0):
        self.split = split
        self.rank = get_rank()
        self.cfg: ImageSeqDataModuleConfig = cfg

        self.height = self.cfg.height if isinstance(self.cfg.height, int) else self.cfg.height[size_ind]
        self.width = self.cfg.width if isinstance(self.cfg.width, int) else self.cfg.width[size_ind]

        self.batch_sizes: List[int] = (
            [self.cfg.batch_size] if isinstance(self.cfg.batch_size, int) else self.cfg.batch_size
        )
        self.batch_size = self.batch_sizes[0]
        
        self.resolution_milestones: List[int]
        if len(self.batch_sizes) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        # load image sequence
        assert os.path.exists(self.cfg.image_seq_path)
        image_names = os.listdir(self.cfg.image_seq_path)
        self.total_images: int = len(image_names)

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                HybridCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )

            if split == "train":

                self.random_pose_generator = HybridCameraIterableDataset(
                    random_camera_cfg,
                    total_images=self.total_images
                )

            else:
                self.random_pose_generator = HybridCameraDataset(
                    random_camera_cfg, split,
                    total_images=self.total_images
                )

             
        self.time_interval = int(self.cfg.time_interval * self.total_images)

        image_names = [f'{i}.png' for i in range(self.total_images)] # sort

        rgba_list = [cv2.cvtColor(
            cv2.imread(os.path.join(self.cfg.image_seq_path, image_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        for image_name in image_names]

        rgba_list = [
            cv2.resize(
                rgba, (self.width * 8 if self.cfg.rgb_as_latents else self.height, self.height * 8 if self.cfg.rgb_as_latents else self.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0 for rgba in rgba_list
        ]
        rgb_list = [rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:]) for rgba in rgba_list]
        
        self.rgb_list: List[Float[Tensor, "1 H W 3"]] = [
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank) for rgb in rgb_list
        ]
        self.mask_list: List[Float[Tensor, "1 H W 1"]] = [
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank) for rgba in rgba_list
        ]

        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "1 3"] = camera_position
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )

        # get directions by dividing directions_unit_focal by focal length
        fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))
        focal_length = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions_unit_focal = get_ray_directions(
            H=self.height, W=self.width, focal=1.0
        )
        directions: Float[Tensor, "1 H W 3"] = directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length

        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance

    def get_first_image(self):
        return self.rgb_list[0]


class ImageSeqIterableDataset(IterableDataset, ImageSeqDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.batch_size = self.batch_sizes[size_ind]
        height = self.cfg.height[size_ind] if type(self.cfg.height) != int else self.cfg.height
        if height != self.height:
            self.setup(self.cfg, self.split, size_ind=size_ind)

    def collate(self, batch) -> Dict[str, Any]:
    
        mode = random.random()
        # 1: same timestamp, random camera pose 
        # 2: same timestamp, continuous camera pose
        # 3: same camera pose, continuous timestamp
        if mode < self.cfg.mode_section[0]:
            mode = 1
        elif mode < self.cfg.mode_section[1]:
            mode = 2
        elif mode < self.cfg.mode_section[2]:
            mode = 3
        else:
            mode = -1
        
        if mode == 1 or mode == 2:
            select_timestamp = np.random.randint(0, self.total_images, 1)[0]
        else:
            select_start_timestamp = np.random.randint(0, self.total_images-self.batch_size*self.time_interval, 1)[0]
            select_timestamp = list(range(select_start_timestamp, select_start_timestamp + self.batch_size*self.time_interval, self.time_interval))
        
        if mode == 1 or mode == 2:
            batch = {
                "rays_o": self.rays_o,
                "rays_d": self.rays_d,
                "mvp_mtx": self.mvp_mtx,
                "camera_positions": self.camera_position,
                "light_positions": self.light_position,
                "elevation": self.elevation_deg,
                "azimuth": self.azimuth_deg,
                "camera_distances": self.camera_distance,
                "rgb": self.rgb_list[select_timestamp],
                # "depth": self.depth_list[select_timestamp],
                # "depth": self.depth,
                "mask": self.mask_list[select_timestamp],
                "timestamps": torch.tensor([select_timestamp]).unsqueeze(1),
                "total_images": self.total_images
            }
        else:
            ref_timestamp = np.random.randint(len(select_timestamp))
            ref_timestamp = select_timestamp[ref_timestamp] # only one ref image, since decode_latents comsumes much GPU memory
            batch = {
                "rays_o": self.rays_o,
                "rays_d": self.rays_d,
                "mvp_mtx": self.mvp_mtx,
                "camera_positions": self.camera_position,
                "light_positions": self.light_position,
                "elevation": self.elevation_deg,
                "azimuth": self.azimuth_deg,
                "camera_distances": self.camera_distance,
                "rgb": self.rgb_list[ref_timestamp],
                # "depth": self.depth_list[ref_timestamp],
                # "depth": self.depth,
                "mask": self.mask_list[ref_timestamp],
                "timestamps": torch.tensor([ref_timestamp]).unsqueeze(1),
                "total_images": self.total_images
            }
        
        batch['guidance'] = 'ref'
            
        if mode == 1:
            if self.cfg.use_random_camera:
                batch["random_camera"] = self.random_pose_generator.collate(None, mode)
                batch["random_camera"]["timestamps"] = batch["timestamps"].repeat(self.batch_size, 1)
                batch["random_camera"]["total_images"] = batch["total_images"]
                batch["random_camera"]["rgb_path"] = [os.path.join(self.cfg.image_seq_path, f'{select_timestamp}.png')] * self.batch_size
        elif mode == 2:
            if self.cfg.use_random_camera:
                batch["random_camera"] = self.random_pose_generator.collate(None, mode)
                batch["random_camera"]["timestamps"] = batch["timestamps"].repeat(self.batch_size, 1)
                batch["random_camera"]["total_images"] = batch["total_images"]
                batch["random_camera"]["rgb_path"] = [os.path.join(self.cfg.image_seq_path, f'{select_timestamp}.png')] * self.batch_size
        else:
            if self.cfg.use_random_camera:
                batch["random_camera"] = self.random_pose_generator.collate(None, 3)
                if mode == -1:
                    batch["random_camera"]["timestamps"] = batch["timestamps"].repeat(self.batch_size, 1)
                else:
                    batch["random_camera"]["timestamps"] = torch.tensor(select_timestamp).unsqueeze(1)
                batch["random_camera"]["total_images"] = batch["total_images"]
                if mode == -1:
                    batch["random_camera"]["rgb_path"] = [os.path.join(self.cfg.image_seq_path, f'{ref_timestamp}.png')] * self.batch_size
                else:
                    batch["random_camera"]["rgb_path"] = [os.path.join(self.cfg.image_seq_path, f'{t}.png') for t in select_timestamp]
        
        batch["mode"] = mode
        batch["random_camera"]["mode"] = mode
        batch["random_camera"]["guidance"] = "zero123"

        return batch

    def __iter__(self):
        while True:
            yield {}


class ImageSeqDataset(Dataset, ImageSeqDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        dataset_length = len(self.random_pose_generator) * len(self.cfg.eval_timestamp) + \
            self.total_images * len(self.cfg.eval_pose_index)
        return dataset_length

    def collate(self, batch) -> Dict[str, Any]:
        res = default_collate(batch)
        res['total_images'] = self.total_images
        res["mode"] = 3
        return res
        
    def __getitem__(self, index):
        # the order is timestamp0, timestamp1, ...
        # then pose0, pose1 ...
        if index < len(self.random_pose_generator) * len(self.cfg.eval_timestamp):
            camera_index = index % len(self.random_pose_generator)
            timestamp_index = index // len(self.random_pose_generator)
            timestamp = self.cfg.eval_timestamp[timestamp_index]
        else:
            tmp_iter = index - len(self.random_pose_generator) * len(self.cfg.eval_timestamp) 
            camera_index = self.cfg.eval_pose_index[tmp_iter//self.total_images]
            timestamp = tmp_iter % self.total_images

        camera_info = self.random_pose_generator[camera_index]

        timestamp = torch.as_tensor([timestamp])
        camera_info['timestamps'] = timestamp
        
        return camera_info


@register("image-seq-datamodule")
class ImageSeqDataModule(pl.LightningDataModule):
    cfg: ImageSeqDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ImageSeqDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ImageSeqIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ImageSeqDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ImageSeqDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.train_dataset.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=self.cfg.eval_batch_size, collate_fn=self.val_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=self.cfg.eval_batch_size, collate_fn=self.test_dataset.collate)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=self.cfg.eval_batch_size)