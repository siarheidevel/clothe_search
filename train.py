from argparse import ArgumentParser
import argparse
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
from pytorch_lightning.callbacks import ModelCheckpoint


import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))
CUDA_DEVICE= 'cuda:1'
torch.cuda.set_device(CUDA_DEVICE)

import model,dataset

def parse_args():
    parser = argparse.ArgumentParser()
        # program level args
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='./checkpoints/', type=str)
    parser.add_argument('--run_name', default='default_run', type=str)
    parser.add_argument('--log_parameters', default=0, type=int)
    parser.add_argument('--batch_size', default=7, type=int)
    parser.add_argument('--n_cpus', default=4, type=int)
    parser.add_argument('--output_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    # parser.add_argument('--gpus', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    args, unknown = parser.parse_known_args()
    # sim_model =Trainer(hidden_dim=256,output_dim=64, use_dropout=True)
    # sim_model
    args.gpus = '1'
    return args

def prepare_dataset(args):
    dataset_path = Path('/home/deeplab/datasets/deepfashion/diordataset_custom')
    all_data = pd.read_csv(dataset_path/'annotation_index.csv', sep=';')
    group_data = all_data[['image_file','image_group']].groupby(['image_group']).count().reset_index().rename(columns={"image_file": "group_count"})
    group_data = group_data.sample(frac=1, random_state=42).reset_index()[['image_group','group_count']]

    classes_group2idx = {v:k+1 for k,v in group_data.to_dict()['image_group'].items()}

    train_data = all_data.merge(group_data[4000:],how='inner')
    test_data = all_data.merge(group_data[:4000],how='inner')

    train_dataset = dataset.DeepFashionClothingDataset(train_data,classes_group2idx, stage = 'train')
    # for i in range(1000,2000,10):
    #     train_dataset.__getitem__(i)

    val_dataset = dataset.DeepFashionClothingDataset(test_data, classes_group2idx, stage='val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.n_cpus ,prefetch_factor=10, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_cpus,prefetch_factor=10, pin_memory=True)
    
    # next(iter(train_dataloader))
    return train_dataloader,val_dataloader


def run_model(args):
    pl.seed_everything(args.seed)
    # ,resume_from_checkpoint='last_model'

    train_dataloader,val_dataloader = prepare_dataset(args)

    #checkpoint rules
    last_checkpoint =  ModelCheckpoint(dirpath='./checkpoints/',
        save_on_train_epoch_end=True,
        save_last=True)
    every_10_epoch_checkpoint = ModelCheckpoint(
        dirpath='./checkpoints/',
        save_on_train_epoch_end=True,
        filename ='{epoch}',
        every_n_epochs = 5
    )

    trainer = pl.Trainer.from_argparse_args(args,
        # resume_from_checkpoint='/home/deeplab/devel/Clothes_search/lightning_logs/version_8/checkpoints/epoch=7-step=1063.ckpt',
        # track_grad_norm=2,weights_summary="full",
        callbacks=[last_checkpoint,every_10_epoch_checkpoint],
        resume_from_checkpoint='./checkpoints/last.ckpt',
        # fast_dev_run=1,

        )
    model_trained = model.TrainerSimilarity(args)

    trainer.fit(model_trained,train_dataloader,val_dataloader)
    print('Finished')


def main():
    args = parse_args()
    run_model(args)
    pass

if __name__=="__main__":
    main()


