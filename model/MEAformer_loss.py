import torch
from torch import nn
import torch.nn.functional as F
import pdb
import numpy as np


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class CustomMultiLossLayer(nn.Module):
    """
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """

    def __init__(self, loss_num):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        #self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)
        # 我们希望初始权重约为 100，所以 log_vars 初始化为 -4.6
        # 假设最后一个 loss 是 Sinkhorn，前几个是普通的
        init_values = torch.zeros(self.loss_num)
        
        # 假设 Sinkhorn 是列表里的最后一个
        init_values[-1] = -4.6  # exp(-(-4.6)) ≈ 99.48
        
        self.log_vars = nn.Parameter(init_values, requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss

class icl_loss(nn.Module):

    def __init__(self, tau=0.05, ab_weight=0.5, n_view=2, intra_weight=1.0, inversion=False, replay=False, neg_cross_kg=False):
        super(icl_loss, self).__init__()
        self.tau = tau
        self.sim = cosine_sim
        self.weight = ab_weight  # the factor of a->b and b<-a
        self.n_view = n_view
        self.intra_weight = intra_weight  # the factor of aa and bb
        self.inversion = inversion
        self.replay = replay
        self.neg_cross_kg = neg_cross_kg

    def softXEnt(self, target, logits, replay=False, neg_cross_kg=False):
        # torch.Size([2239, 4478])

        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        if replay:
            logits = logits
            idx = torch.arange(start=0, end=logprobs.shape[0], dtype=torch.int64).cuda()
            stg_neg = logits.argmax(dim=1)
            new_value = torch.zeros(logprobs.shape[0]).cuda()
            index = (
                idx,
                stg_neg,
            )
            logits = logits.index_put(index, new_value)
            stg_neg_2 = logits.argmax(dim=1)
            tmp = idx.eq_(stg_neg)
            neg_idx = stg_neg - stg_neg * tmp + stg_neg_2 * tmp
            return loss, neg_idx

        return loss

    # train_links[:, 0]: shape: (2239,)
    # array([11303,  2910,  2072, ..., 10504, 13555,  8416], dtype=int32)

    def forward(self, emb, train_links, neg_l=None, neg_r=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
        num_ent = emb.shape[0]
        # Get (normalized) hidden1 and hidden2.
        zis = emb[train_links[:, 0]]
        zjs = emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight
        # 2
        n_view = self.n_view
        LARGE_NUM = 1e9
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]
        hidden1_large = hidden1
        hidden2_large = hidden2

        if neg_l is None:
            num_classes = batch_size * n_view
        else:
            num_classes = batch_size * n_view + neg_l.shape[0]
            num_classes_2 = batch_size * n_view + neg_r.shape[0]

        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
        labels = labels.cuda()
        if neg_l is not None:
            labels_2 = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes_2).float()
            labels_2 = labels_2.cuda()

        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.cuda().float()
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM

        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM

        if neg_l is not None:
            zins = emb[neg_l]
            zjns = emb[neg_r]
            logits_ana = torch.matmul(hidden1, torch.transpose(zins, 0, 1)) / temperature
            logits_bnb = torch.matmul(hidden2, torch.transpose(zjns, 0, 1)) / temperature

        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

        # logits_a = torch.cat([logits_ab, self.intra_weight*logits_aa], dim=1)
        # logits_b = torch.cat([logits_ba, self.intra_weight*logits_bb], dim=1)
        if self.inversion:
            logits_a = torch.cat([logits_ab, logits_bb], dim=1)
            logits_b = torch.cat([logits_ba, logits_aa], dim=1)
        else:
            if neg_l is None:
                logits_a = torch.cat([logits_ab, logits_aa], dim=1)
                logits_b = torch.cat([logits_ba, logits_bb], dim=1)
            else:
                logits_a = torch.cat([logits_ab, logits_aa, logits_ana], dim=1)
                logits_b = torch.cat([logits_ba, logits_bb, logits_bnb], dim=1)

        if self.replay:
            loss_a, a_neg_idx = self.softXEnt(labels, logits_a, replay=True, neg_cross_kg=self.neg_cross_kg)
            if neg_l is not None:
                loss_b, b_neg_idx = self.softXEnt(labels_2, logits_b, replay=True, neg_cross_kg=self.neg_cross_kg)
                #
                a_ea_cand = torch.cat([train_links[:, 1], train_links[:, 0], neg_l]).cuda()
                b_ea_cand = torch.cat([train_links[:, 0], train_links[:, 1], neg_r]).cuda()
            else:
                loss_b, b_neg_idx = self.softXEnt(labels, logits_b, replay=True, neg_cross_kg=self.neg_cross_kg)
                a_ea_cand = torch.cat([train_links[:, 1], train_links[:, 0]]).cuda()
                b_ea_cand = torch.cat([train_links[:, 0], train_links[:, 1]]).cuda()

            a_neg = a_ea_cand[a_neg_idx]
            b_neg = b_ea_cand[b_neg_idx]
            return alpha * loss_a + (1 - alpha) * loss_b, a_neg, b_neg

        else:
            loss_a = self.softXEnt(labels, logits_a)
            loss_b = self.softXEnt(labels, logits_b)
            return alpha * loss_a + (1 - alpha) * loss_b

class SinkhornLoss(nn.Module):
    def __init__(self, tau=0.05, n_iter=5):
        super(SinkhornLoss, self).__init__()
        self.tau = tau
        self.n_iter = n_iter

    def forward(self, emb1, emb2):
        # emb1, emb2: [Batch_Size, Dim]
        
        # 1. 计算相似度矩阵 (Cosine Similarity)
        # 归一化以确保数值稳定
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # S_ij = emb1_i * emb2_j / tau
        sim_matrix = torch.matmul(emb1, emb2.t()) / self.tau

        # 2. 可微分 Sinkhorn 迭代 (Log-space)
        # 将相似度矩阵“双向归一化”，逼近置换矩阵
        P = sim_matrix
        for _ in range(self.n_iter):
            # 行归一化
            P = P - torch.logsumexp(P, dim=1, keepdim=True)
            # 列归一化
            P = P - torch.logsumexp(P, dim=0, keepdim=True)

        # 3. 计算对角线 Loss (最大化正确匹配对的概率)
        # 我们希望 P[i, i] (对角线元素) 尽可能大
        # Loss = -mean(P[i, i])
        
        # 获取 batch 内的对角线索引
        batch_size = emb1.size(0)
        diag_indices = torch.arange(batch_size, device=emb1.device)
        
        # 取出对角线上的 Log Probability
        log_prob_diag = P[diag_indices, diag_indices]
        
        # 最小化负的对数概率
        loss = -torch.mean(log_prob_diag)
        
        return loss