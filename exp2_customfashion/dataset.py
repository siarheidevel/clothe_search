from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
from torchvision import transforms as T
import cv2, torch, os, random
from PIL import Image
import numpy as np
import utils

class ClothingDataset(Dataset):
    '''
    similar items from deepfashion dataset    
    '''
    seg_labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8,}
     
    def __init__(self, data: pd.DataFrame, stage:str = 'val') -> None:
        super().__init__()
        self.data = data
        # self.class_count = class_count
        # self.instance_count = instance_count
        # self.gender_count = gender_count
        self.out_size = (224,224)
        self.stage = stage
        self.AUGM_COUNT = 3
        self.bg_color=(0,0,0)
        
        self.tfms = T.Compose([T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if self.stage == 'train':
            # massive augmentation
            self.tfms_augm = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(transforms=[utils.FreeFormMask(nodes=4,fill=self.bg_color)], p=0.5),
                T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2))], p=0.2),
                T.RandomPerspective(distortion_scale=0.7, p=0.8,fill=self.bg_color),
                T.RandomApply(transforms=[T.RandomAffine(degrees=(-10, 10), translate=(0.01, 0.01), scale=(1.2, 1.6))], p=0.7),
                
                # T.RandomApply(transforms=[T.RandomCrop(size=self.out_size)], p=0.5),
                # T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
               
                T.RandomGrayscale(p=0.1),
                T.RandomApply(transforms=[T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1, hue=0.1)], p=0.7),
                
                T.RandomApply(transforms=[utils.ISONoise()], p=0.3),
                # T.RandomApply(transforms=[utils.AddGaussianNoise(mean=0,std=0.01)], p=0.5),
                T.ToTensor(),
                #add gausian noise
                # T.RandomApply(transforms=[T.Lambda(lambda x: x + torch.rand(x.shape))], p=0.5),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                #T.RandomApply(transforms=[utils.AddGaussianNoise(mean=0,std=0.01)], p=0.5),
                ]
            )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row.image_file
        img_rgb = cv2.imread(image_path)[:,:,[2,1,0]]#Image.fromarray(img_rgb.astype(np.uint8)).save('img_res.png')
        # seg = np.load(image_path + '.seg.npz')['mask']
        seg = np.array(Image.open(image_path + '.seg_qanet.render.png'))
        mask = seg==row.seg_id
        bbox_img = utils.bounding_box(seg*mask)
        
        garment_img = np.ones_like(img_rgb) * self.bg_color
        garment_img[mask] = img_rgb[mask]
        # garment_img = img_rgb
        garment_img = garment_img[bbox_img[0]:bbox_img[1],bbox_img[2]:bbox_img[3],:]
        garment_img = utils.resize_to_box(garment_img,
                to_size=self.out_size, keep_aspect=True, bg_color=self.bg_color)
        garment_img_with_bg = utils.resize_to_box(img_rgb[bbox_img[0]:bbox_img[1],bbox_img[2]:bbox_img[3],:],
                to_size=self.out_size, keep_aspect=True, bg_color=self.bg_color)
        if random.random()<0.3:
            garment_img = garment_img_with_bg
        
        garment_tensor = self.tfms(Image.fromarray(garment_img.astype(np.uint8), mode ='RGB'))
        # Image.fromarray(np.transpose((utils.denormalize_tensor(garment_tensor)*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')
        if self.stage == 'train':
            augm_tensors = []
            for a in range(self.AUGM_COUNT-1):
                aug_tensor = self.tfms_augm(Image.fromarray(garment_img.astype(np.uint8), mode ='RGB'))
                augm_tensors.append(aug_tensor)
            augm_tensors.append(self.tfms_augm(Image.fromarray(garment_img_with_bg.astype(np.uint8), mode ='RGB')))
            # Image.fromarray(np.transpose((utils.denormalize_tensor(torch.cat((garment_tensor,*augm_tensors),-1))*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')
            return garment_tensor, row.label_id, row.category_id, row.gender_id,  augm_tensors , idx

        return garment_tensor, row.label_id, row.category_id, row.gender_id, [], idx

# classes_group2idx = {v:k for k,v in enumerate(set(data['image_group'].to_list()))}    
# train_dataset = DeepFashionClothingDataset(train_data,classes_group2idx)
# val_dataset = DeepFashionClothingDataset(test_data, classes_group2idx)

# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# next(iter(train_dataloader))    



