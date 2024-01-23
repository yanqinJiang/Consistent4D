from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

import os
import math

def sort_images_by_id(image_paths):
    def get_id_from_path(path):
        filename = os.path.basename(path)
        id_str = filename.split('.')[0]
        return int(id_str)

    sorted_paths = sorted(image_paths, key=get_id_from_path)
    return sorted_paths
    
class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sort_images_by_id(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        num_images = self.opt.seq_length  # 要获取的图像数量
        start_index = index * num_images  # 起始索引
        data = []  # 存储图像数据的列表
        for i in range(start_index, start_index + num_images):
            if i >= len(self.A_paths):
                # 如果索引超出数据集范围，复制最后一个数据
                i = len(self.A_paths) - 1
            A_path = self.A_paths[i]
            A_img = Image.open(A_path).convert('RGB')
            A = self.transform(A_img)
            data.append({'A': A, 'A_paths': A_path})

        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return math.ceil(len(self.A_paths) / self.opt.seq_length)
