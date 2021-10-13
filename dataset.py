from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms as T
import cv2, torch
from PIL import Image
import numpy as np
from mask import Masks

def bounding_box(img):
    a = np.where(img != 0)
    if a[0].size==0 or a[1].size==0:
        return None
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    if bbox[1]-bbox[0] <=1 or bbox[3] - bbox[2] <= 1:
        return None
    return bbox


def resize_to_box(img, to_size=(224,224),keep_aspect=True, interpolation=cv2.INTER_LINEAR):
    old_size = img.shape[:2]
    ratio = min(to_size[0]/ old_size[0], to_size[1]/ old_size[1])
    if keep_aspect:
        new_size = tuple([round(x * ratio) for x in old_size])
    else:
        new_size = to_size
    if ratio < 1:
        img = cv2.blur(img,(3,3))
    resized_img = cv2.resize(img,(new_size[1],new_size[0]),interpolation=interpolation)
    res_image = np.zeros((*to_size,3))
    res_image[(to_size[0]-new_size[0])//2:(to_size[0]-new_size[0])//2+new_size[0],
        (to_size[1]-new_size[1])//2:(to_size[1]-new_size[1])//2+new_size[1],:] = resized_img
    return res_image


class DeepFashionClothingDataset(Dataset):
    '''
    similar items from deepfashion dataset    
    '''
    seg_labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8,}
    

    __deepfashion_tops =set(['Tees_Tanks', 'Blouses_Shirts', 'Dresses', 'Sweaters',
        'Jackets_Coats', 'Sweatshirts_Hoodies', 'Rompers_Jumpsuits', 'Cardigans',
        'Graphic_Tees', 'Shirts_Polos', 'Jackets_Vests', 'Suiting'])

    __deepfashion_bottoms = set([ 'Shorts', 'Pants', 'Skirts',
        'Denim', 'Leggings', 'Suiting'])

    __category_labels ={**{k:4 for k in __deepfashion_tops},
        **{k:5 for k in __deepfashion_bottoms}}

     
    def __init__(self, data: pd.DataFrame, classes: dict, stage:str = 'val') -> None:
        super().__init__()
        # dataset_path = Path(dir)
        # self.data = pd.read_csv(dataset_path/'annotation_index.csv', sep=';')
        # data[['image_file','image_group']].groupby(['image_group']).count().reset_index()
        self.data = data
        self.classes = classes
        self.num_classes = len(self.classes)
        self.out_size = (224,224)
        self.stage = stage
        
        self.tfms = T.Compose([T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if self.stage == 'train':
            # massive augmentation
            self.tfms_augm = T.Compose([
                T.RandomPerspective(distortion_scale=0.7, p=1.0),
                T.RandomApply(transforms=[T.RandomResizedCrop(size=self.out_size,scale=(0.7, 1.2))], p=0.5),
                # T.RandomApply(transforms=[T.RandomCrop(size=self.out_size)], p=0.5),
                # T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(transforms=[T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1, hue=0.1)], p=0.7),
                T.RandomApply(transforms=[FreeFormMask(nodes=4)], p=0.5),
                T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 3))], p=0.5),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row.image_file
        img_rgb = cv2.imread(image_path)[:,:,[2,1,0]]#Image.fromarray(img_rgb.astype(np.uint8)).save('img_res.png')
        # seg = np.load(image_path + '.seg.npz')['mask']
        seg = np.array(Image.open(image_path + '.seg3.render.png'))
        label = DeepFashionClothingDataset.__category_labels[row.category]
        bbox_img = bounding_box(seg*(seg==label))
        # print(bbox_img, img_rgb.shape)
        if bbox_img is None:
            # TODO return 1 pixel?
            garment_img = np.zeros((*self.out_size,3))
            class_id = 0
        else:
            class_id = self.classes[row.image_group]

            garment_img = ((seg==label)[...,np.newaxis] * img_rgb)[bbox_img[0]:bbox_img[1],bbox_img[2]:bbox_img[3],:]
            garment_img = resize_to_box(garment_img,to_size=self.out_size,keep_aspect=True)
            
            # Image.fromarray(garment_img.astype(np.uint8)).save('img_res.png')
        
        
        # perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        # T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        # T.RandomHorizontalFlip(p=0.5)
        #  T.RandomApply(transforms=[T.RandomCrop(size=(64, 64))], p=0.5)
        # T.RandomResizedCrop(size=(32, 32))
        
        garment_tensor = self.tfms(Image.fromarray(garment_img.astype(np.uint8), mode ='RGB'))
        # Image.fromarray(np.transpose((denormalize_tensor(garment_tensor)*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')
        if self.stage == 'train':
            AUGM_COUNT = 4
            augm_tensors = []
            for a in range(AUGM_COUNT):
                aug_tensor = self.tfms_augm(Image.fromarray(garment_img.astype(np.uint8), mode ='RGB'))
                augm_tensors.append(aug_tensor)
            # Image.fromarray(np.transpose((denormalize_tensor(torch.cat((garment_tensor,*augm_tensors),-1))*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')
            return garment_tensor, class_id, idx, label, augm_tensors

        return garment_tensor, class_id, idx, label

# classes_group2idx = {v:k for k,v in enumerate(set(data['image_group'].to_list()))}    
# train_dataset = DeepFashionClothingDataset(train_data,classes_group2idx)
# val_dataset = DeepFashionClothingDataset(test_data, classes_group2idx)

# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# next(iter(train_dataloader))    

def denormalize_tensor(t:torch.Tensor):
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    x = t * STD[:, None, None] + MEAN[:, None, None]
    return x

class FreeFormMask(object):
    """
    Free form masking
    args: nodes - number of masking nodes
    """

    def __init__(self, nodes = 4, fill=0):
        self.nodes = nodes
        self.fill = fill

    def __call__(self, img):
        h, w = img.size
        mask = Masks.get_ff_mask(h, w, num_v=self.nodes).astype(np.uint8)
        masked_image = img * (1 - mask)[...,np.newaxis]
        return Image.fromarray(masked_image)