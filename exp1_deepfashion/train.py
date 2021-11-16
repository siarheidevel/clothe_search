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
from pytorch_lightning.callbacks import ModelCheckpoint


import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))
CUDA_DEVICE= '1'
# torch.cuda.set_device(CUDA_DEVICE)

import model, deepfashion_dataset, utils

def parse_args():
    parser = argparse.ArgumentParser()
        # program level args
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='./checkpoints/dense121_category', type=str)
    parser.add_argument('--run_name', default='default_run', type=str)
    parser.add_argument('--log_parameters', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--n_cpus', default=8, type=int)
    parser.add_argument('--output_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--category_weight', default=0.1, type=float)    
    parser.add_argument('--gpus', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    args, unknown = parser.parse_known_args()
    # sim_model =Trainer(hidden_dim=256,output_dim=64, use_dropout=True)
    # sim_model
    # args.gpus = '1'
    return args


def prepare_dataset(args):
    data_file = '/home/deeplab/devel/Clothes_search/util/deepfashion_index_qanet.csv'
    data = pd.read_csv(data_file, sep=';')
    data = data.sample(frac=1, random_state=42)
    # TODO split data
    top_categories = data[['gender','maybe_category']].value_counts().reset_index()
    top_categories.columns = [['gender','maybe_category','cnt']]
    category_map = {category:cat_id for cat_id,category in enumerate(top_categories.apply(lambda x:x.gender+'_'+x.maybe_category, axis=1).to_list())}
    data['cat_id'] = data.apply(lambda  x: category_map[x.gender + '_' + x.maybe_category], axis=1)
    labels_map = {label:label_id for label_id,label in enumerate(set(data.label.to_list()))}
    data['label_id'] = data.apply(lambda  x: labels_map[x.label], axis=1)
    total_cat_sum = int(top_categories.cnt.sum())
    data['cat_weight'] = total_cat_sum / data.cat_id.map(data.cat_id.value_counts())
    shuffled_data = data
    train_data, test_data = shuffled_data[:int(len(shuffled_data)*0.8)], data[int(len(shuffled_data)*0.8):]
    class_count = len(data.cat_id.value_counts())
    instance_count = len(data.label_id.value_counts())
    
    train_dataset = deepfashion_dataset.DeepFashionClothingDataset(train_data,class_count,instance_count, stage='train')
    # for i in [4,1,100,1000,5000]:
    #     print(train_dataset.__getitem__(i))
    unbalanced_sampler = WeightedRandomSampler(weights=train_data.cat_weight.to_list(), num_samples=len(train_data))
    train_dataloader = DataLoader(train_dataset,sampler=unbalanced_sampler, batch_size=args.batch_size, 
        num_workers=args.n_cpus ,prefetch_factor=10, pin_memory=True)
    # for i,batch in enumerate(train_dataloader):
    #     print(batch)
    #     if i> 10: break
    
    val_dataset = deepfashion_dataset.DeepFashionClothingDataset(pd.concat((test_data, train_data)),
        class_count,instance_count, stage='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_cpus,prefetch_factor=10, pin_memory=True)
    return train_dataloader,val_dataloader, (instance_count, class_count, gender_count)


def run_model(args):
    pl.seed_everything(args.seed)
    # ,resume_from_checkpoint='last_model'
    Path(args.save_dir).mkdir(parents=True,exist_ok=True)

    train_dataloader,val_dataloader, class_count, instance_count = prepare_dataset(args)

    #checkpoint rules
    last_checkpoint =  ModelCheckpoint(dirpath=args.save_dir,
        save_on_train_epoch_end=True,
        save_last=True)
    every_10_epoch_checkpoint = ModelCheckpoint(
        dirpath=args.save_dir,
        save_on_train_epoch_end=True,
        filename ='{epoch}',
        every_n_epochs = 5
    )

    trainer = pl.Trainer.from_argparse_args(args,
        # resume_from_checkpoint='/home/deeplab/devel/Clothes_search/lightning_logs/version_8/checkpoints/epoch=7-step=1063.ckpt',
        # track_grad_norm=2,weights_summary="full",
        callbacks=[last_checkpoint,every_10_epoch_checkpoint],
        resume_from_checkpoint=args.save_dir+'/last.ckpt',
        # fast_dev_run=1,
        gpus = [CUDA_DEVICE],
        max_epochs=20,
        )
    model_trained = model.TrainerSimilarity(args, category_count=class_count)

    trainer.fit(model_trained,train_dataloader,val_dataloader)
    print('Finished')


def main():
    args = parse_args()
    run_model(args)
    pass

if __name__=="__main__":
    main()





# def prepare_dataset(args):
#     dataset_path = Path('/home/deeplab/datasets/deepfashion/diordataset_custom')
#     all_data = pd.read_csv(dataset_path/'annotation_index.csv', sep=';')
#     group_data = all_data[['image_file','image_group']].groupby(['image_group']).count().reset_index().rename(columns={"image_file": "group_count"})
#     group_data = group_data.sample(frac=1, random_state=42).reset_index()[['image_group','group_count']]

#     classes_group2idx = {v:k+1 for k,v in group_data.to_dict()['image_group'].items()}

#     train_data = all_data.merge(group_data[4000:],how='inner')
#     test_data = all_data.merge(group_data[:4000],how='inner')

#     train_dataset = dataset.DeepFashionClothingDataset(train_data,classes_group2idx, stage = 'train')
#     # for i in range(1000,2000,10):
#     #     train_dataset.__getitem__(i)

#     val_dataset = dataset.DeepFashionClothingDataset(test_data, classes_group2idx, stage='val')

#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
#         num_workers=args.n_cpus ,prefetch_factor=10, pin_memory=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
#         num_workers=args.n_cpus,prefetch_factor=10, pin_memory=True)
    
#     # next(iter(train_dataloader))
#     return train_dataloader,val_dataloader