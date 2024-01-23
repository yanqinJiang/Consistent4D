import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

import math

class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # 获取图像文件夹路径
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))  # 获取

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size 应该小于等于加载的图像大小
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        num_images = self.opt.seq_length  # 要获取的图像数量
        start_index = index * num_images  # 起始索引

        data = []  # 存储图像数据的列表
        for i in range(start_index, start_index + num_images):
            if i >= len(self.AB_paths):
                # 如果索引超出数据集范围，复制最后一个数据
                i = len(self.AB_paths) - 1

            # 读取图像
            AB_path = self.AB_paths[i]
            AB = Image.open(AB_path).convert('RGB')
            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))

            # 应用相同的变换
            transform_params = get_params(self.opt, A.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

            A = A_transform(A)
            B = B_transform(B)

            data.append({'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path})

        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return math.ceil(len(self.AB_paths) / self.opt.seq_length)