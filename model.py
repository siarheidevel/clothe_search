import torch
from torch.functional import Tensor
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import cv2
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, reducers, distances, regularizers
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from dataset import denormalize_tensor

class Sim_model(nn.Module):
    def __init__(self,hidden_dim=256,output_dim=64,dropout=0.0, norm=None):
        super().__init__()
        self.extractor = EfficientNet.from_pretrained('efficientnet-b3')
        # self.extractor.eval()
        # for p in self.extractor.parameters():
        #     p.requires_grad=False
        extractor_output_dim = 1536
        self.head = []
        if dropout > 0:
            self.head.append(nn.Dropout(dropout))
        self.head.append(nn.Linear(extractor_output_dim,hidden_dim))
        self.head.append(nn.LeakyReLU(0.1))
        if dropout:
            self.head.append(nn.Dropout(dropout))
        self.head.append(nn.Linear(hidden_dim,output_dim))
        self.head = nn.Sequential(*self.head)

    
    def forward(self, x):
        # with torch.no_grad():
        out = self.extractor.extract_features(x)
        n,_,_,_ = out.shape
        out = nn.AdaptiveAvgPool2d(output_size=1)(out).view(n,-1)

        out = self.head(out)
        out = F.normalize(out, p=2, dim=1)
        return out


class TrainerSimilarity(pl.LightningModule):
    def __init__(self,args,**kwargs):
        super().__init__()
        self.args = args
        self.holder = {} #defaultdict(list)
        self.model = Sim_model(hidden_dim=args.hidden_dim, output_dim=args.output_dim, dropout=args.dropout)
        # Automatically log all the arguments to the tensorboard 
        # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        self.save_hyperparameters(args)
        # need this to log computational graph to Tensorboard
        # in main(): pl.loggers.TensorBoardLogger(log_graph = True)
        self.example_input_array = torch.empty(4, 3, 224, 224)
        # distance = distances.DotProductSimilarity()
        # distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
        # reducer = reducers.ThresholdReducer(low=0)
        # embedding_regularizer = regularizers.LpRegularizer()
        # self.loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, 
        #     reducer=reducer, embedding_regularizer=embedding_regularizer, embedding_reg_weight = 0.1)

        # self.miner_func = miners.TripletMarginMiner(margin=0.2,type_of_triplets="semihard")

        # self.loss_func = losses.CrossBatchMemory(
        #     loss=self.loss_func,
        #     embedding_size=args.output_dim, memory_size=3072,miner=self.miner_func)
        # self.loss_func = losses.ContrastiveLoss(pos_margin=0.1, neg_margin=1)

        distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
        loss = losses.ContrastiveLoss(pos_margin=0.1, neg_margin = 1, distance=distance)
        miner = miners.PairMarginMiner(pos_margin=0.1, neg_margin = 1, distance=distance)
        self.loss_func = losses.CrossBatchMemory(
            loss=loss,
            embedding_size=args.output_dim, memory_size=4000,miner=miner)
        # self.loss_func = loss
        # self.miner_func = miner

        
        

    def forward(self, x):
        emb = self.model(x)
        return emb

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        garment_tensor, class_id, idx, _, augm_tensors = batch
        embeddings = self.forward(garment_tensor)
        if augm_tensors is not None and len(augm_tensors)>0:
            init_class_id, init_idx =class_id, idx
            for i in range(len(augm_tensors)):
                augm_emb = self.forward(augm_tensors[i])
                embeddings =torch.cat((embeddings, augm_emb), 0)
                class_id = torch.cat((class_id,init_class_id),0)
                idx = torch.cat((idx,init_idx),0)
                garment_tensor = torch.cat((garment_tensor,augm_tensors[i]),0)


        # hard_pairs = self.miner_func(embeddings, class_id)
        # loss = self.loss_func(embeddings, class_id, hard_pairs)

        loss = self.loss_func(embeddings, class_id)
        self._add_holder_data('train',idx, garment_tensor,embeddings.detach(), class_id)
        # Logging to TensorBoard by default
        self.log("train_loss", loss.item())
        return loss
    
    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self._calculate_metrics('train')
        

    def validation_step(self, batch, batch_idx):
        garment_tensor, class_id, idx, label = batch
        embeddings = self.forward(garment_tensor)
        loss = self.loss_func(embeddings, class_id)
        # hard_pairs = self.miner_func(embeddings, class_id)
        # loss = self.loss_func(embeddings, class_id, hard_pairs)

        self._add_holder_data('val',idx, garment_tensor,embeddings.detach(), class_id)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)

    

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)
        self._calculate_metrics('val')


    def _add_holder_data(self, stage:str, idx, garment_tensors, embeddings, class_ids):
        self.__add_holder_data_tensor(f'{stage}_idx',idx.cpu())
        self.__add_holder_data_tensor(f'{stage}_garment_tensors',
            F.interpolate(garment_tensors, scale_factor=1/4, 
                recompute_scale_factor=True,mode='bilinear' ,align_corners=False).cpu())
        self.__add_holder_data_tensor(f'{stage}_embeddings',embeddings.cpu())
        self.__add_holder_data_tensor(f'{stage}_class_ids',class_ids.cpu())
    
    def __add_holder_data_tensor(self, key:str, tensor):
        if not key in self.holder:
            self.holder[key] = tensor
        else:
            self.holder[key] = torch.cat((self.holder[key], tensor), 0)
    
    def _calculate_metrics(self, stage:str):
        idx = self.holder[f'{stage}_idx']
        garment_tensor = self.holder[f'{stage}_garment_tensors']
        # Image.fromarray(np.transpose((denormalize_tensor(garment_tensor[0])*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')
        embeddings = self.holder[f'{stage}_embeddings']
        class_ids = self.holder[f'{stage}_class_ids']
        TOP_K = 14
        # https://davidefiocco.github.io/nearest-neighbor-search-with-faiss/
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        neighbors = NearestNeighbors(n_neighbors=TOP_K,  metric='minkowski', algorithm='brute')        

        QUERIES_K = 1000
        query_emb = embeddings[:QUERIES_K]
        query_classes = class_ids[:QUERIES_K]

        # neighbors.fit(embeddings[QUERIES_K:])
        neighbors.fit(embeddings[:])
        knn_distances, knn_indexes = neighbors.kneighbors(query_emb, return_distance = True)
        # knn_indexes = knn_indexes + QUERIES_K
        knn_indexes = torch.Tensor(knn_indexes).long()
        AP_K = 10
        AP_at_K = torch.zeros((knn_indexes.shape[0]))
        valid_matrix = (class_ids[knn_indexes] == query_classes[:,None]).int()
        for K in range(1, AP_K):
            k_knn_indexes = knn_indexes[:,1:K+1]
            kvalid = torch.sum((class_ids[k_knn_indexes] == query_classes[:,None]).int(),1)
            class_count = torch.unique(class_ids, return_counts=True)
            allvalid = torch.sum(((class_count[0][None] == query_classes[:,None]).int()* class_count[1]),1)
            precision_k = kvalid/K
            recall_k  = kvalid/ allvalid
            AP_at_K += valid_matrix[:,K]*precision_k
        AP_at_K = AP_at_K/AP_K


        DRAW_K = 30
        _,img_c,img_h,img_w = garment_tensor.shape
        anchor = denormalize_tensor(garment_tensor[:DRAW_K])[:,None]
        results = denormalize_tensor(garment_tensor[knn_indexes[:DRAW_K].reshape(-1)]
            ).reshape(knn_indexes[:DRAW_K].shape[0],TOP_K,img_c,img_h,img_w)

        # mark true valid images
        # (class_ids[knn_indexes] == query_classes[:,None]).int()[:DRAW_K]
        real_positive_samples = (class_ids[knn_indexes] == query_classes[:,None]).int()[:DRAW_K]
        sim_template = torch.zeros((img_c,img_h,img_w))
        sim_template[1,...]=1
        sim_template[1,1:img_h-1,1:img_w-1]=0
        sim_mask = sim_template*real_positive_samples[...,None,None,None]
        # results = torch.where(real_positive_samples[...,None,None,None]>0,sim_mask,results)
        results = sim_mask + results#torch.where(sim_mask>0,sim_mask,results)        

        imgs = torch.cat((anchor,results),1)
        imgs_rows = torch.cat(imgs.unbind(1),3)
        one_img = torch.cat(imgs_rows.unbind(0),1)
        # from PIL import Image;Image.fromarray(np.transpose((one_img*254).byte().cpu().numpy(), (1, 2, 0))).save('visual.png')
        # from PIL import Image;
        self.logger.experiment.add_image(f"visual_{stage}", one_img, self.current_epoch)
        self.log_dict({
            f'{stage}_map@{AP_K}':torch.mean(AP_at_K),
            f'{stage}_precision@{AP_K}':torch.mean(precision_k),
            f'{stage}_recall@{AP_K}':torch.mean(recall_k)}) 
        
        
        print(f'{stage} epoch {self.current_epoch}: map@{AP_K}={torch.mean(AP_at_K).item():.4f}, prec@{AP_K}={torch.mean(precision_k).item():.4f}, rec@{AP_K}={torch.mean(recall_k).item():.4f}')
        print(knn_distances[:2])
        from PIL import Image;Image.fromarray(np.transpose((one_img*254).byte().cpu().numpy(), (1, 2, 0))).save(f'./checkpoints/{stage}_visual_{self.current_epoch}.png')
        from PIL import Image;Image.fromarray(np.transpose((one_img*254).byte().cpu().numpy(), (1, 2, 0))).save(f'./{stage}_visual.png')

        del self.holder[f'{stage}_idx']
        del self.holder[f'{stage}_garment_tensors']
        del self.holder[f'{stage}_embeddings']
        del self.holder[f'{stage}_class_ids']


    def configure_optimizers(self):       
        # optimizer = torch.optim.Adam(self.model.head.parameters(), lr=self.args.lr, weight_decay=0.0001)
        optimizer = torch.optim.Adam( [
            {'params': self.model.extractor.parameters(), 'lr': self.args.lr/100},
            {'params': self.model.head.parameters(), 'lr': self.args.lr},
            ], self.args.lr, weight_decay=0.000001)

        # scheduler =torch.optim.lr_scheduler.StepLR(optimizer,step_size = 30, gamma=0.2)
        # CosineAnnealingWarmRestarts 
        # return [optimizer], [scheduler]

        return optimizer
    
    # def load_model_from_checkpoint(self, checkpoint_path: str):
    #     checkpoint = torch.load(checkpoint_path)
    #     state_dict = checkpoint["state_dict"]
    #     # for k in list(state_dict.keys()):
    #     #     if not k.startswith("model."):
    #     #         del state_dict[k]
    #     self.load_state_dict(state_dict, strict=False)



