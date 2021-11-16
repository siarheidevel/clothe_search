# %%
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim 

import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

# %%
dataset_path = Path('/home/deeplab/datasets/deepfashion/diordataset_custom')
data = pd.read_csv(dataset_path/'annotation_index.csv', sep=';')
data_pairs =pd.read_csv(dataset_path/'annotation_pairs.csv', sep=';')
print(data.shape, data_pairs.shape)
data.head(5)

 
# %%
#catgories
seg_labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8,}
print(data[['category']].groupby(['category']).size().sort_values(ascending=False).reset_index())
data[['category']].groupby(['category']).size().sort_values(ascending=False).reset_index().category.to_list()
deepfashion_tops =set(['Tees_Tanks',
    'Blouses_Shirts',
    'Dresses',
    'Sweaters',
    'Jackets_Coats',
    'Sweatshirts_Hoodies',
    'Rompers_Jumpsuits',
    'Cardigans',
    'Graphic_Tees',
    'Shirts_Polos',
    'Jackets_Vests',
    'Suiting'])

deepfashion_bottoms = set([ 'Shorts',
    'Pants',
    'Skirts',
    'Denim',
    'Leggings',
    'Suiting'])
def label_from_category(category):
    if category in deepfashion_bottoms: return seg_labels['pants']
    if category in deepfashion_tops: return seg_labels['upper-clothes']
    return -1
    


# %%


model = EfficientNet.from_pretrained('efficientnet-b3')
list(reversed(list(model.modules())))[:10]


# %%
# img_path = data.iloc[1100,0]
# print(img_path)
# img_rgb = cv2.imread(img_path)[:,:,[2,1,0]]
# # img_rgb= Image.open(img_path)

# seg_labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
#                'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8}
# # seg = np.load(img_path + '.seg.npz')['mask']
# seg = np.array(Image.open(img_path + '.seg3.render.png'))

# print(seg.shape, np.unique(seg, return_counts=True))
# plt.imshow(img_rgb)
# plt.show()
# label = 4#seg_labels['pants']
# plt.imshow(seg*(seg==label))
# plt.show()
# plt.imshow((seg==label)[...,np.newaxis] * img_rgb)
# garment_img = (seg==label)[...,np.newaxis] * img_rgb


#1 %%
# data[data.image_file.str.contains('_flat', regex=False)][:10]


# %% select anchor file
img_path=data.iloc[8931,0]
print(img_path)
def get_positive_pairs(image_file):
    category=data[data.image_file==image_file][:1].category.item()
    group_id = data[data.image_file==image_file][:1].image_group.item()
    grouped_ = data[data.image_group==group_id]
    files= grouped_[grouped_.image_file!=image_file].image_file.to_list()
    return category, files

category, img_files = get_positive_pairs(img_path)
print(category, img_files)
# data[data['image_group']==data[data['image_file']==img_path]['image_group']
# grouped_=pd.merge(data,data[data.image_file==img_path].image_group)


# %%
def bbox(img):
    a = np.where(img != 0)
    if a[0].size==0 or a[1].size==0 :
        return None
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
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


def garment_embedding(image_path, label):
    img_rgb = cv2.imread(image_path)[:,:,[2,1,0]]
    # seg = np.load(image_path + '.seg.npz')['mask']
    seg = np.array(Image.open(image_path + '.seg3.render.png'))
    bbox_img = bbox(seg*(seg==label))
    # print(bbox_img, img_rgb.shape)
    if bbox_img is None:
        return None
    garment_img= ((seg==label)[...,np.newaxis] * img_rgb)[bbox_img[0]:bbox_img[1],bbox_img[2]:bbox_img[3],:]
    # garment_img = (img_rgb)[bbox_img[0]:bbox_img[1],bbox_img[2]:bbox_img[3],:]
    # print(cv2.resize(img_rgb,(garment_img.shape[1],garment_img.shape[0])).shape,garment_img.shape)
    # plt.imshow(np.concatenate((cv2.resize(img_rgb,(garment_img.shape[1],garment_img.shape[0])),garment_img),1))
    # print(garment_img.shape)
    garment_img = resize_to_box(garment_img,to_size=(224,224),keep_aspect=True)
    # cv2.imwrite('img_res.png',garment_img)
    # plt.show(garment_img)
    # plt.show(Image.open(image_path + '.seg3.render.png'))
    # plt.show()
    # tfms = transforms.Compose([transforms.Resize(size=(224,224)), transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    tfms = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.fromarray(garment_img, mode ='RGB')).unsqueeze(0)
    print(img.shape)
    model.eval()
    with torch.no_grad():
        outputs = model.extract_features(img)
    n,c,h,w = outputs.shape
    return nn.AdaptiveAvgPool2d(output_size=1)(outputs).view(n,-1)

# %%
import pandas as pd
import numpy as np

dataset_path = Path('/home/deeplab/datasets/deepfashion/diordataset_custom')
all_data = pd.read_csv(dataset_path/'annotation_index.csv', sep=';')
group_data = all_data[['image_file','image_group']].groupby(['image_group']).count().reset_index().rename(columns={"image_file": "group_count"})
group_data = group_data.sample(frac=1, random_state=42).reset_index()[['image_group','group_count']]
train_data = data.merge(group_data[1000:],how='inner')
test_data = data.merge(group_data[:1000],how='inner')

print(train_data.shape[0],test_data.shape[0])
print(train_data.head(5))

# %%
from torch.utils.data import DataLoader, Dataset
class DeepFashionClothingDataset(Dataset):
    seg_labels = {'background': 0, 'hat':1, 'hair': 2,  'face': 3, 'upper-clothes':4,
               'pants': 5, 'arm': 6, 'leg': 7, 'shoes':8,}
    

    __deepfashion_tops =set(['Tees_Tanks', 'Blouses_Shirts', 'Dresses', 'Sweaters',
        'Jackets_Coats', 'Sweatshirts_Hoodies', 'Rompers_Jumpsuits', 'Cardigans',
        'Graphic_Tees', 'Shirts_Polos', 'Jackets_Vests', 'Suiting'])

    __deepfashion_bottoms = set([ 'Shorts', 'Pants', 'Skirts',
        'Denim', 'Leggings', 'Suiting'])

    __category_labels ={**{k:4 for k in __deepfashion_tops},
        **{k:5 for k in __deepfashion_bottoms}}

    @staticmethod
    def label_from_category(category:str):
            if category in deepfashion_bottoms: return seg_labels['pants']
            if category in deepfashion_tops: return seg_labels['upper-clothes']
            return -1
    
    def __init__(self, data: pd.DataFrame, classes: dict) -> None:
        super().__init__()
        # dataset_path = Path(dir)
        # self.data = pd.read_csv(dataset_path/'annotation_index.csv', sep=';')
        # data[['image_file','image_group']].groupby(['image_group']).count().reset_index()
        self.data = data
        self.classes = classes
        self.num_classes = len(self.classes)
        self.out_size = (224,224)
        self.tfms = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        
        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = data.iloc[idx]
        image_path = row.image_file
        img_rgb = cv2.imread(image_path)[:,:,[2,1,0]]
        # seg = np.load(image_path + '.seg.npz')['mask']
        seg = np.array(Image.open(image_path + '.seg3.render.png'))
        label = DeepFashionClothingDataset.__category_labels[row.category]
        bbox_img = bbox(seg*(seg==label))
        # print(bbox_img, img_rgb.shape)
        if bbox_img is None:
            # TODO return 1 pixel?
            garment_img = np.zeros((*self.out_size,3))
            class_id = 0
        else:
            class_id = self.classes[row.image_group]

            garment_img = ((seg==label)[...,np.newaxis] * img_rgb)[bbox_img[0]:bbox_img[1],bbox_img[2]:bbox_img[3],:]
            garment_img = resize_to_box(garment_img,to_size=self.out_size,keep_aspect=True)
            
            # cv2.imwrite('img_res.png',garment_img)
        garment_tensor = self.tfms(Image.fromarray(garment_img, mode ='RGB'))
        return garment_tensor, class_id

classes_group2idx = {v:k for k,v in enumerate(set(data['image_group'].to_list()))}    
train_dataset = DeepFashionClothingDataset(train_data,classes_group2idx)
val_dataset = DeepFashionClothingDataset(test_data, classes_group2idx)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

next(iter(train_dataloader))    

#%%
import argparse
parser = argparse.ArgumentParser()
    # program level args
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--save_dir', default='./e00_default_sweep/', type=str)
parser.add_argument('--run_name', default='default_run', type=str)
parser.add_argument('--log_parameters', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_cpus', default=1, type=int)
parser.add_argument('--output_dim', default=64, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
# parser.add_argument('--gpus', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
args, unknown = parser.parse_known_args()
# sim_model =Trainer(hidden_dim=256,output_dim=64, use_dropout=True)
# sim_model
args.gpus = '1'
pl.seed_everything(args.seed)
trainer = pl.Trainer.from_argparse_args(args)

# %%
class Sim_model(nn.Module):
    def __init__(self,hidden_dim=256,output_dim=64,use_dropout=False, norm=None):
        super().__init__()
        self.extractor = EfficientNet.from_pretrained('efficientnet-b3')
        extractor_output_dim = 1536
        self.head = []
        if use_dropout:
            self.head.append(nn.Dropout(0.2))
        self.head.append(nn.Linear(extractor_output_dim,hidden_dim))
        self.head.append(nn.LeakyReLU(0.1))
        if use_dropout:
            self.head.append(nn.Dropout(0.2))
        self.head.append(nn.Linear(hidden_dim,output_dim))
        self.head = nn.Sequential(*self.head)

    
    def forward(self, x):
        with torch.no_grad():
            out = self.extractor.extract_features(x)
            n,_,_,_ = out.shape
            out = nn.AdaptiveAvgPool2d(output_size=1)(out).view(n,-1)
        out = self.head(out)
        return out


class TrainerSimilarity(pl.LightningModule):
    def __init__(self,args,**kwargs):
        super().__init__()
        self.args = args
        self.model = Sim_model(hidden_dim=args.hidden_dim, output_dim=args.output_dim, use_dropout=True)
        # Automatically log all the arguments to the tensorboard 
        # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        self.save_hyperparameters(args)
        # need this to log computational graph to Tensorboard
        # in main(): pl.loggers.TensorBoardLogger(log_graph = True)
        self.example_input_array = torch.empty(4, 3, 224, 224)

    def forward(self, x):
        emb = self.model(x)
        return emb

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.head.parameters(), lr=args.lr)
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.n_cpus)
    
    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.n_cpus)
    
    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("LitModel")
    #     parser.add_argument("--encoder_layers", type=int, default=12)
    #     parser.add_argument("--data_path", type=str, default="/some/path")
    #     return parent_parser

# trainer = pl.Trainer()
model_trainer = TrainerSimilarity(args)
# trainer.fit(model, mnist_train)
trainer.fit(model_trainer)
# %%


# %%
def main():
    parser = ArgumentParser()
    # program level args
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='./e00_default_sweep/', type=str)
    parser.add_argument('--run_name', default='default_run', type=str)
    parser.add_argument('--log_parameters', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--output_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    

    tb_logger = pl.loggers.TensorBoardLogger(save_dir = args.save_dir+'log/',
                                            name = args.run_name,
                                            version = 'fixed_version',
                                            log_graph = True)
    



# %%
emb1=garment_embedding(img_path,4)
emb2=garment_embedding(img_files[0],4)
emb3=garment_embedding(data.iloc[8937,0],4)
# bbox_img = bbox(seg*(seg==label))
# if bbox_img is not None:
#     garment_img= ((seg==label)[...,np.newaxis] * img_rgb)[bbox_img[0]:bbox_img[1],bbox_img[2]:bbox_img[3],:]
#     print(bbox_img)
#     print(garment_img.shape)
# garment_img[bbox_img[0]:bbox_img[0]
# np.sum(np.power(emb1-emb2,2),keepdims=False)
np.linalg.norm(emb1-emb2),np.linalg.norm(emb1-emb3),
