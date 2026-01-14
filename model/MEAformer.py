import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .Tool_model import AutomaticWeightedLoss
from .MEAformer_tools import MultiModalEncoder
from .MEAformer_loss import CustomMultiLossLayer, icl_loss, SinkhornLoss

from src.utils import pairwise_distances
import os.path as osp
import json


class MEAformer(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        
        # 增加一个 Temperature 参数用于 InfoNCE
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.05))
        
        self.rel_features = torch.Tensor(kgs["rel_features"]).cuda()
        self.att_features = torch.Tensor(kgs["att_features"]).cuda()
        self.name_features = None
        self.char_features = None
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()

        img_dim = self._get_img_dim(kgs)

        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head,
                                                    attr_input_dim=kgs["att_features"].shape[1])

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=6)  # 6
        # 实例化一个专门管理 Sinkhorn Loss 的自动权重层
        # loss_num=1，因为我们只把 Sinkhorn Loss 传给它
        # 由于我们在 CustomMultiLossLayer 里写了 init_values[-1]=-4.6，
        # 即使只有一个元素，它也会被初始化为 -4.6 (即权重100)
        self.sinkhorn_weight_layer = CustomMultiLossLayer(loss_num=1)
        # =================
        self.criterion_cl = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_joint = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2, replay=self.args.replay, neg_cross_kg=self.args.neg_cross_kg)

        tmp = -1 * torch.ones(self.input_idx.shape[0], dtype=torch.int64).cuda()
        self.replay_matrix = torch.stack([self.input_idx, tmp], dim=1).cuda()
        self.replay_ready = 0
        self.idx_one = torch.ones(self.args.batch_size, dtype=torch.int64).cuda()
        self.idx_double = torch.cat([self.idx_one, self.idx_one]).cuda()
        self.last_num = 1000000000000
        # self.idx_one = np.ones(self.args.batch_size, dtype=np.int64)

        if "topo_features" in kgs:
            self.topo_features = kgs["topo_features"].cuda()
        else:
            self.topo_features = None

        # 初始化 Sinkhorn Loss
        self.sinkhorn_loss_fn = SinkhornLoss(tau=0.05, n_iter=10)

    def forward(self, batch):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states = self.joint_emb_generat(only_joint=False)
        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)
        if self.args.replay:
            batch = torch.tensor(batch, dtype=torch.int64).cuda()
            all_ent_batch = torch.cat([batch[:, 0], batch[:, 1]])
            if not self.replay_ready:
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch)
            else:
                neg_l = self.replay_matrix[batch[:, 0], self.idx_one[:batch.shape[0]]]
                neg_r = self.replay_matrix[batch[:, 1], self.idx_one[:batch.shape[0]]]
                neg_l_set = set(neg_l.tolist())
                neg_r_set = set(neg_r.tolist())
                all_ent_set = set(all_ent_batch.tolist())
                neg_l_list = list(neg_l_set - all_ent_set)
                neg_r_list = list(neg_r_set - all_ent_set)
                neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64).cuda()
                neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64).cuda()
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch, neg_l_ipt, neg_r_ipt)

            index = (
                all_ent_batch,
                self.idx_double[:batch.shape[0] * 2],
            )
            new_value = torch.cat([l_neg, r_neg]).cuda()

            self.replay_matrix = self.replay_matrix.index_put(index, new_value)
            if self.replay_ready == 0:
                num = torch.sum(self.replay_matrix < 0)
                if num == self.last_num:
                    self.replay_ready = 1
                    print("-----------------------------------------")
                    print("begin replay!")
                    print("-----------------------------------------")
                else:
                    self.last_num = num
        else:
            loss_joi = self.criterion_cl_joint(joint_emb, batch)

        in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch)
        out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch)        

        # === [修改点: 跨模态 InfoNCE Loss] ===
        # 计算 Graph 和 Image 之间的对比损失
        # 目标: 同一实体的 Graph 和 Image 特征要尽量相似，不同实体的要尽量不相似
        loss_cl_cross = self.compute_contrastive_loss(gph_emb, img_emb, batch)
        
        #loss_all = loss_joi + in_loss + out_loss + self.args.conflict_weight * loss_cl_cross

        # === [新增] 计算 Sinkhorn Training Loss ===
        # 1. 获取当前 Batch 对应的左右实体 Embedding
        if torch.is_tensor(batch):
             batch_idx = batch
        else:
             batch_idx = torch.tensor(batch, dtype=torch.int64).cuda()
             
        # 从 joint_emb (所有实体) 中索引出当前 Batch 的特征
        emb_left = joint_emb[batch_idx[:, 0]]
        emb_right = joint_emb[batch_idx[:, 1]]
        
        # 2. 计算 Loss
        loss_sinkhorn = self.sinkhorn_loss_fn(emb_left, emb_right)
        # =========================================

        #sinkhorn_weight = 1.0
        weighted_sinkhorn_loss = self.sinkhorn_weight_layer([loss_sinkhorn])
        
        loss_all = loss_joi + in_loss + out_loss + \
                   self.args.conflict_weight * loss_cl_cross + \
                   weighted_sinkhorn_loss

        #loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}
        loss_dic = {
            "joint_Intra_modal": loss_joi.item(), 
            "Intra_modal": in_loss.item(),
            "Sinkhorn": loss_sinkhorn.item()
        }
        output = {"loss_dic": loss_dic, "emb": joint_emb}
        return loss_all, output

    # === [新增: InfoNCE 实现] ===
    def compute_contrastive_loss(self, feat1, feat2, batch):
        """
        Input: feat1 (All entities), feat2 (All entities), batch (Train pairs)
        We only optimize for the entities in the current batch to save memory.
        """
        if feat1 is None or feat2 is None:
            return torch.tensor(0.0).cuda()

        # === [修复点] ===
        # 检查 batch 是否为 tensor，如果不是（是 numpy），则转换
        if not torch.is_tensor(batch):
            batch = torch.tensor(batch, dtype=torch.int64).cuda()
        # ===============
            
        # 1. 取出当前 Batch 的实体特征
        # batch: [B, 2] (left_id, right_id)
        # 我们把 left 和 right 都拿出来做对比
        batch_indices = batch.flatten().unique()
        
        f1 = F.normalize(feat1[batch_indices], dim=1)
        f2 = F.normalize(feat2[batch_indices], dim=1)
        
        # 2. 计算相似度矩阵
        # [Batch_Unique, Batch_Unique]
        logits = torch.matmul(f1, f2.t()) * self.logit_scale.exp()
        
        # 3. 标签是对象角线 (0,0), (1,1)... 因为是同一个实体的不同模态
        labels = torch.arange(len(f1)).cuda()
        
        # 4. 双向 Cross Entropy
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        
        return (loss_i + loss_t) / 2    

    def generate_hidden_emb(self, hidden):
        gph_emb = F.normalize(hidden[:, 0, :].squeeze(1))
        rel_emb = F.normalize(hidden[:, 1, :].squeeze(1))
        att_emb = F.normalize(hidden[:, 2, :].squeeze(1))
        img_emb = F.normalize(hidden[:, 3, :].squeeze(1))
        if hidden.shape[1] >= 6:
            name_emb = F.normalize(hidden[:, 4, :].squeeze(1))
            char_emb = F.normalize(hidden[:, 5, :].squeeze(1))
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb], dim=1)
        else:
            name_emb, char_emb = None, None
            loss_name, loss_char = None, None
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb], dim=1)

        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, joint_emb

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):
        # pdb.set_trace()
        loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
        loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
        loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0

        total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char])
        return total_loss

    # --------- necessary ---------------

    def joint_emb_generat(self, only_joint=True):
        
        ret_tuple = self.multimodal_encoder(
            self.input_idx, 
            self.adj, 
            self.img_features,
            self.rel_features, 
            self.att_features,
            self.name_features, 
            self.char_features,
            topo_features=self.topo_features
        )
        
        # 解包
        gph_emb = ret_tuple[0]
        joint_emb = ret_tuple[6]
        
        if only_joint:
            return joint_emb, ret_tuple[-1] # weight_norm
        else:
            return ret_tuple[:-1] # 返回除 weight_norm 外的所有

    # --------- share ---------------

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        return new_links

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            # remove from left/right_non_train
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            if self.args.rank == 0:
                logger.info(f"#new_links_select:{len(new_links_select)}")
                logger.info(f"train_ill.shape:{train_ill.shape}")
                logger.info(f"#true_links: {num_true}")
                logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
                logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links
