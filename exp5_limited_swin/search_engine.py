import torch
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor, NamedTupleCursor
import logging, time
from pathlib import Path
from collections import OrderedDict
import cv2
from PIL import ImageFont, ImageDraw, Image
from torchvision import transforms as T

CUDA_DEVICE_ID = 1
CUDA_DEVICE= 'cuda:'+str(CUDA_DEVICE_ID)
torch.cuda.set_device(CUDA_DEVICE)

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import model, dataset, utils


DSN = "dbname=postgres user=postgres password=111111 port=5433 host=127.0.0.1"

garment_model = None

@torch.no_grad()
def load_model():
    global garment_model

    garment_model = model.Sim_Swin_model(output_dim=192)
    garment_model.eval()

    emb = garment_model(torch.randn(1,3,224,224))
    checkpoint  = torch.load('/home/deeplab/devel/Clothes_search/exp5_limited_swin/checkpoints/efnetb3s/last.ckpt',
        map_location=lambda storage, loc: storage)
    #load only model parameters, exclude 'model.' from naming
    model_dict = OrderedDict([(n[6:],d) for (n,d) in checkpoint['state_dict'].items() if n.startswith('model.')])
    garment_model.load_state_dict(model_dict)
    
    emb = garment_model(torch.randn(1,3,224,224))
    return garment_model

@torch.no_grad()
def garment_vector(image_file, bg_color=(255,255,255), out_size=(224,224)):
    """
    return 3,224,224 tensor
    """
    img_rgb = np.array(Image.open(image_file))#Image.fromarray(img_rgb.astype(np.uint8)).save('img_res.png')
    if img_rgb.shape[0]!=out_size[0] or img_rgb.shape[1]!=out_size[1]:
        img_rgb = utils.resize_to_box(img_rgb,to_size=out_size, keep_aspect=True, bg_color=bg_color)
    tfms = T.Compose([T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    garment_tensor = tfms(Image.fromarray(img_rgb.astype(np.uint8), mode ='RGB'))
    embedding = garment_model(garment_tensor[None,...])[0]
    return embedding.cpu().numpy() #Image.fromarray(np.transpose((utils.denormalize_tensor(garment_tensor)*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')



if __name__=='__main__':
    # print(db_query('select 1+1'))
    print('')
    # fill_init_data()
    garment_model = load_model()
    garment_model.to(CUDA_DEVICE)
    
    print(garment_model)
    # fill_vector_data()


