from datasets.preprocess.builder import build_preprocess
import paddle
from .base_dataset import BaseDataset
from .builder import DATASETS
from .image_folder import is_image_file
from .transforms.builder import build_transforms

@DATASETS.register()
class SingleImageDataset(paddle.io.Dataset):
    """
    """
    def __init__(self, dataroot, preprocess, transforms, **kwargs):
        """Initialize single dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.
        """
        super(SingleImageDataset, self).__init__()
        self.dataroot = dataroot
        if not is_image_file(self.dataroot):
            raise RuntimeError(f"Illegal image data input: {self.dataroot}")
        self.transform = build_transforms(transforms)
        self.preprocess = build_preprocess(preprocess)
        self.scale_factor, self.num_scales, self.images = self.preprocess(self.dataroot)
        self.max_image_size = max(self.images[-1]['image'].size)
        self.min_image_size = max(self.images[0]['image'].size)
        self.scales = [img['image'].size[::-1] for img in self.images]

    def __getitem__(self, index):
        return_dict = self.images[index]
        return_dict['img'] = self.transform(return_dict['image']).transpose(2, 0, 1)

        return return_dict


    def __len__(self):
        return len(self.images)