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
from pathlib import Path
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.utils.data as utils
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, GPUStatsMonitor
from pytorch_metric_learning import samplers

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))
CUDA_DEVICE_ID = 0
CUDA_DEVICE= 'cuda:'+str(CUDA_DEVICE_ID)
torch.cuda.set_device(CUDA_DEVICE)

import model, dataset, utils

def parse_args():
    parser = argparse.ArgumentParser()
        # program level args
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='./checkpoints/dense121_category', type=str)
    parser.add_argument('--run_name', default='default_run', type=str)
    parser.add_argument('--log_parameters', default=0, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--n_cpus', default=8, type=int)
    parser.add_argument('--output_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--category_weight', default=0.5, type=float)   
    parser.add_argument('--gender_weight', default=0.3, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    # parser.add_argument('--gpus', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    args, unknown = parser.parse_known_args()
    # sim_model =Trainer(hidden_dim=256,output_dim=64, use_dropout=True)
    # sim_model
    # args.gpus = '1'
    return args


def prepare_dataset(args):
    data_file = '/home/deeplab/devel/Clothes_search/util/customfashion_index_qanet_processed2.csv'
    data = pd.read_csv(data_file, sep=';')

    category_count = data.category_id.max()+1 #len(data.category_id.value_counts())
    instance_count = data.label_id.max()+1  #len(data.label_id.value_counts())
    gender_count = data.gender_id.max()+1#len(data.gender_id.value_counts())

    # shuffle data and split to train and test
    # data = data.sample(frac=1, random_state=42)
    # TODO spit by label
    label_stat = pd.DataFrame(data.label_id.value_counts().reset_index().to_numpy(),
        columns=['label_id', 'label_count']).sample(frac=1, random_state=42)
    label_stat_train, label_stat_test = label_stat[:int(len(label_stat)*0.8)], label_stat[int(len(label_stat)*0.8):]
    # train_data, test_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
    train_data = data.merge(label_stat_train['label_id'], how='inner', on='label_id')
    test_data = data.merge(label_stat_test['label_id'], how='inner', on='label_id')
    
    
    train_dataset = dataset.ClothingDataset(train_data, stage='train')
    # for i in [4,1,100,1000,5000]:
    #     print(train_dataset.__getitem__(i))
    # unbalanced_sampler = WeightedRandomSampler(weights=train_data.category_weight.to_list(), num_samples=len(train_data))
    perclass_sampler = samplers.MPerClassSampler(
            labels=train_data.label_id.to_list(), m=3, 
            # batch_size=args.batch_size, 
            length_before_new_iter=len(train_dataset))
    train_dataloader = DataLoader(train_dataset,sampler=perclass_sampler, batch_size=args.batch_size, 
        num_workers=args.n_cpus ,prefetch_factor=10, pin_memory=True)
    # for i,batch in enumerate(train_dataloader):
    #     print(batch)
    #     if i> 10: break
    
    val_dataset = dataset.ClothingDataset(test_data, stage='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*3, shuffle=False,
        num_workers=args.n_cpus,prefetch_factor=10, pin_memory=True)
    return train_dataloader,val_dataloader, category_count, gender_count, instance_count


def run_model(args):
    pl.seed_everything(args.seed)
    # ,resume_from_checkpoint='last_model'
    Path(args.save_dir).mkdir(parents=True,exist_ok=True)

    train_dataloader,val_dataloader, category_count, gender_count, instance_count = prepare_dataset(args)

    #checkpoint rules
    last_checkpoint =  ModelCheckpoint(dirpath=args.save_dir,
        save_on_train_epoch_end=True,
        save_last=True)
    every_10_epoch_checkpoint = ModelCheckpoint(
        dirpath=args.save_dir,
        save_on_train_epoch_end=True,
        filename ='{epoch}',
        every_n_epochs = 1,
        save_top_k = -1
    )

    every_10000_steps_checkpoint = ModelCheckpoint(
        dirpath=args.save_dir,
        every_n_train_steps =5000,
        filename ='{epoch}-{step}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    device_stats = GPUStatsMonitor(memory_utilization=True,
        gpu_utilization= True,
        intra_step_time = False,
        inter_step_time = False,
        fan_speed = False,
        temperature=True)

    trainer = pl.Trainer.from_argparse_args(args,
        # resume_from_checkpoint='/home/deeplab/devel/Clothes_search/exp4_customfashion/checkpoints/dense121_category/epoch=5-step=179999.ckpt',
        # track_grad_norm=2,weights_summary="full",
        callbacks=[lr_monitor, every_10000_steps_checkpoint,last_checkpoint,every_10_epoch_checkpoint, device_stats],
        # resume_from_checkpoint=args.save_dir+'/last.ckpt',
        fast_dev_run=1,
        gpus = [CUDA_DEVICE_ID],
        max_epochs=args.epochs,
        )
    model_trained = model.TrainerSimilarity(args, label_count=instance_count, category_count=category_count, gender_count=gender_count)
    model_trained.load_model_from_checkpoint(args.save_dir+'/last.ckpt', strict=False)
    # model_trained.load_from_checkpoint(checkpoint_path=args.save_dir+'/last.ckpt', strict=False)
    # init model weights
    trainer.fit(model_trained,train_dataloader,val_dataloader)
    print('Finished')


def main():
    args = parse_args()
    run_model(args)
    pass

if __name__=="__main__":
    main()

