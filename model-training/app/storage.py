from torch.utils.data import Dataset
from torch import load, device
from PIL import Image
from utils import transform_input


class Storage:
    @staticmethod
    def load_model_from_s3(s3, s3_path, model):
        """
        Load trained model from s3 bucket
        :param s3_path:
        :param model:
        :return:
        """
        with s3.open(s3_path, 'rb') as f:
            model.load_state_dict(load(f, map_location=device('cpu')))


class CarotidUltrasoundDataset(Dataset):
    def __init__(self, s3fs, image_path, mask_path, transform=False):
        self.s3fs = s3fs
        self.transform = transform
        self.transform_input = transform_input
        self.images = [file for file in self.s3fs.ls(image_path) if file.lower().endswith(('.png', '.jpg'))]
        # logger.info(self.images[0])
        self.masks = [file.replace('data/images', 'data/labels').replace('val/images', 'val/labels').replace('.jpg', '.png') for file in self.images]
        # logger.info(self.masks[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        with self.s3fs.open(image_path, 'rb') as f:
            image = Image.open(f).convert("L")

        with self.s3fs.open(mask_path, 'rb') as f:
            mask = Image.open(f).convert("L")

        if self.transform:
            image = self.transform_input(image)
            mask = self.transform_input(mask)

        return image, mask
