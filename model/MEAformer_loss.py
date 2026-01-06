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
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss


class icl_loss(nn.Module):

    def __init__(self, tau=0.05, ab_weight=0.5, n_view=2, intra_weight=1.0, 
                 inversion=False, replay=False, neg_cross_kg=False, 
                hard_mining=True, hard_k=1024):
        super(icl_loss, self).__init__()
        self.tau = tau
        self.sim = cosine_sim
        self.weight = ab_weight  # the factor of a->b and b<-a
        self.n_view = n_view
        self.intra_weight = intra_weight  # the factor of aa and bb
        self.inversion = inversion
        self.replay = replay
        self.neg_cross_kg = neg_cross_kg

        # [新增] 硬负样本挖掘配置
        self.hard_mining = hard_mining
        self.hard_k = hard_k

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

    def compute_hard_mining_loss(self, logits_aa, logits_bb, logits_ab, logits_ba, 
                                 logits_ana, logits_bnb, batch_size, alpha):
        """
        专门用于计算 Hard Negative Mining 的 Loss
        """
        # 1. 提取正样本 (对角线元素) [Batch, 1]
        pos_a = torch.diag(logits_ab).unsqueeze(1)
        pos_b = torch.diag(logits_ba).unsqueeze(1)

        # 2. 构建负样本候选集
        # 掩码：用于屏蔽 logits_ab/ba 中的正样本（对角线）
        diag_mask = torch.eye(batch_size, device=logits_ab.device).bool()
        
        # --- 处理 View A (Left) ---
        # 负样本来源1: 跨模态 (AB) 的非对角线元素
        neg_ab = logits_ab.clone()
        neg_ab.masked_fill_(diag_mask, -1e9) # 将正样本位置设为极小值，使其不会被 TopK 选中
        
        # 负样本来源2: 同模态 (AA)
        # 注意: logits_aa 在 forward 中已经对其对角线做了 -LARGE_NUM 处理，所以直接用
        neg_aa = logits_aa 
        
        # 负样本来源3: Replay 策略的额外负样本 (如有)
        neg_list_a = [neg_ab, neg_aa]
        if logits_ana is not None:
            neg_list_a.append(logits_ana)
            
        # 拼接所有负样本候选 [Batch, N_Neg]
        all_negs_a = torch.cat(neg_list_a, dim=1)
        
        # 3. 核心步骤: 选取 Top-K 最难负样本
        # 如果候选数量不足 K，则选取全部
        k = min(self.hard_k, all_negs_a.size(1))
        hard_negs_a, _ = torch.topk(all_negs_a, k=k, dim=1)
        
        # 4. 构建最终 Logits: [正样本, 硬负样本1, ..., 硬负样本K]
        final_logits_a = torch.cat([pos_a, hard_negs_a], dim=1)
        
        # --- 处理 View B (Right) ---
        neg_ba = logits_ba.clone()
        neg_ba.masked_fill_(diag_mask, -1e9)
        neg_bb = logits_bb
        
        neg_list_b = [neg_ba, neg_bb]
        if logits_bnb is not None:
            neg_list_b.append(logits_bnb)
            
        all_negs_b = torch.cat(neg_list_b, dim=1)
        hard_negs_b, _ = torch.topk(all_negs_b, k=k, dim=1)
        final_logits_b = torch.cat([pos_b, hard_negs_b], dim=1)
        
        # 5. 计算 Cross Entropy
        # 标签全是 0 (因为正样本被我们拼接在了第0列)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits_ab.device)
        
        loss_a = F.cross_entropy(final_logits_a, labels)
        loss_b = F.cross_entropy(final_logits_b, labels)
        
        # 返回加权 Loss
        return alpha * loss_a + (1 - alpha) * loss_b

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

        # === [修改处开始] ===
        # 必须在这里先初始化为 None，防止后面报错
        logits_ana = None
        logits_bnb = None
        # ===================

        if neg_l is not None:
            zins = emb[neg_l]
            zjns = emb[neg_r]
            logits_ana = torch.matmul(hidden1, torch.transpose(zins, 0, 1)) / temperature
            logits_bnb = torch.matmul(hidden2, torch.transpose(zjns, 0, 1)) / temperature

        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

        # [新增] 硬负样本挖掘分支
        if self.hard_mining:
            return self.compute_hard_mining_loss(
                logits_aa, logits_bb, logits_ab, logits_ba, 
                logits_ana, logits_bnb, batch_size, alpha
            )

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