import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import zipfile

class aoi_Dataset(Dataset):
    def __init__(self, zip_path, label_path=None, transform=transforms.Compose([transforms.ToTensor()]), phase='train'):
        super(aoi_Dataset, self).__init__()
        self.zip_path = zip_path
        self.label_path = label_path
        self.transform = transform
        self.zip_file = zipfile.ZipFile(zip_path, mode='r')
        self.img_list = []
        for img_name in self.zip_file.namelist():
            if '.png' not in img_name:
                continue
            self.img_list.append(img_name)
        self.num_img = len(self.img_list)
        if phase == 'train':
            self.label_list = pd.read_csv(label_path)['Label']
        else:
            self.label_list = np.zeros((len(pd.read_csv(label_path)['Label'])))
        self.num_class = 6
        assert len(self.label_list) == self.num_img, print('Length of label and image list must be the same.')
        print('number of image:', self.num_img)
        print('number of class:', self.num_class)

    def __getitem__(self, index):
        with self.zip_file.open(self.img_list[index], mode='r') as img_file:
            img_file = img_file.read()
            img = cv2.imdecode(np.asarray(bytearray(img_file), dtype='uint8'), cv2.IMREAD_GRAYSCALE)
            img = Image.fromarray(img)
        return self.transform(img), self.label_list[index]

    def __len__(self):
        return self.num_img
        
    def get_class_ratio(self):
        return [len(self.label_list[self.label_list==i])/len(self.label_list) for i in range(self.num_class)]

    def close_zip(self):
        self.zip_file.close()
