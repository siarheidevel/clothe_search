from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
from torchvision import transforms as T
import cv2, torch, os, random
from PIL import Image
import numpy as np
import utils




class DeepFashionClothingDataset(Dataset):
    '''
    similar items from deepfashion dataset    
    '''
    seg_labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8,}
     
    def __init__(self, data: pd.DataFrame, class_count: int, instance_count: int, stage:str = 'val') -> None:
        super().__init__()
        # dataset_path = Path(dir)
        # self.data = pd.read_csv(dataset_path/'annotation_index.csv', sep=';')
        # data[['image_file','image_group']].groupby(['image_group']).count().reset_index()
        self.data = data
        self.class_count = class_count
        self.instance_count = instance_count
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
        if random.random()<0.5:
            garment_img = garment_img_with_bg
        # # print(bbox_img, img_rgb.shape)
        # if bbox_img is None:
        #     # TODO return 1 pixel?
        #     garment_img = np.zeros((*self.out_size,3))
        #     class_id = 0
        # else:
        #     class_id = self.classes[row.image_group]

        #     garment_img = ((seg==label)[...,np.newaxis] * img_rgb)[bbox_img[0]:bbox_img[1],bbox_img[2]:bbox_img[3],:]
        #     garment_img = utils.resize_to_box(garment_img,
        #         to_size=self.out_size, keep_aspect=True, bg_color=(0,0,0))
            
        #     # Image.fromarray(garment_img.astype(np.uint8)).save('img_res.png')
        
        garment_tensor = self.tfms(Image.fromarray(garment_img.astype(np.uint8), mode ='RGB'))
        # Image.fromarray(np.transpose((utils.denormalize_tensor(garment_tensor)*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')
        if self.stage == 'train':
            augm_tensors = []
            for a in range(self.AUGM_COUNT-1):
                aug_tensor = self.tfms_augm(Image.fromarray(garment_img.astype(np.uint8), mode ='RGB'))
                augm_tensors.append(aug_tensor)
            augm_tensors.append(self.tfms_augm(Image.fromarray(garment_img_with_bg.astype(np.uint8), mode ='RGB')))
            # Image.fromarray(np.transpose((utils.denormalize_tensor(torch.cat((garment_tensor,*augm_tensors),-1))*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')
            return garment_tensor, row.cat_id, row.label_id, augm_tensors , idx

        return garment_tensor, row.cat_id, row.label_id, idx

# classes_group2idx = {v:k for k,v in enumerate(set(data['image_group'].to_list()))}    
# train_dataset = DeepFashionClothingDataset(train_data,classes_group2idx)
# val_dataset = DeepFashionClothingDataset(test_data, classes_group2idx)

# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# next(iter(train_dataloader))    

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    data_file = '/home/deeplab/devel/Clothes_search/util/deepfashion_index_qanet.csv'
    data = pd.read_csv(data_file, sep=';')
    # TODO split data
    top_categories = data[['gender','maybe_category']].value_counts().reset_index()
    top_categories.columns = [['gender','maybe_category','cnt']]
    category_map = {category:cat_id for cat_id,category in enumerate(top_categories.apply(lambda x:x.gender+'_'+x.maybe_category, axis=1).to_list())}
    data['cat_id'] = data.apply(lambda  x: category_map[x.gender + '_' + x.maybe_category], axis=1)
    labels_map = {label:label_id for label_id,label in enumerate(set(data.label.to_list()))}
    data['label_id'] = data.apply(lambda  x: labels_map[x.label], axis=1)
    total_cat_sum = int(top_categories.cnt.sum())
    data['cat_weight'] = total_cat_sum / data.cat_id.map(data.cat_id.value_counts())
    shuffled_data = data.sample(frac=1, random_state=42)
    train_data, test_data = shuffled_data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
    class_count = len(data.cat_id.value_counts())
    instance_count = len(data.label_id.value_counts())
    
    train_dataset =DeepFashionClothingDataset(train_data,class_count,instance_count, stage='train')
    unbalanced_sampler = WeightedRandomSampler(weights=train_data.cat_weight, num_samples=len(train_data))
    train_dataloader = DataLoader(train_dataset,sampler=unbalanced_sampler, batch_size=1, 
        num_workers=1 ,prefetch_factor=10, pin_memory=True)
    for i,batch in enumerate(train_dataloader):
        print(batch)
        if i> 10: break
    print('finished')


