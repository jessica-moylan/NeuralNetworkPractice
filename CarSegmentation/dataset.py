import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        #list all files that are in that folder
        self.images = os.listdir(image_dir)

    #get how many images are in a specified directory
    def __len__(self):
        return len(self.images)
    
    #get a specified item in images at a specified index
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        
        #loading the images
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #we convert the mask to a black and white image
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augumentation = self.transform(image = image, mask = mask)
            image = augumentation["image"]
            mask = augumentation["mask"]

        return image, mask



