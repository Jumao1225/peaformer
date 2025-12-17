# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import pdb


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class MultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """

    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head, 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, adj):
        output = []
        for i in range(self.n_head):
            # pdb.set_trace()
            N = input.size()[0]
            edge = adj._indices()
            if self.diag:
                h = torch.mul(input, self.w[i])
            else:
                h = torch.mm(input, self.w[i])

            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1)  # edge: 2*D x E
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze()))  # edge_e: 1 x E
            # e_rowsum: N x 1
            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda() if next(self.parameters()).is_cuda else torch.ones(size=(N, 1)))
            # pdb.set_trace()
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E

            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)

            output.append(h_prime.unsqueeze(0))
        # pdb.set_trace()
        output = torch.cat(output, dim=0)
        # pdb.set_trace()
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        if x is not None:
            x = self.l1(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.l2(x)
            return x

# [File: model/layers.py]
# ... (Keep existing imports and classes like SpecialSpmm, MultiHeadGraphAttention, GraphConvolution, ProjectionHead)

# # ========= [NEW] Hypergraph Convolution Layer =========
# class HypergraphConv(nn.Module):
#     """
#     Hypergraph Convolution Layer
#     Mathematically: X' = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X \Theta
#     Simplified for efficiency: Node -> Hyperedge -> Node aggregation
#     """
#     def __init__(self, in_features, out_features, dropout=0.5):
#         super(HypergraphConv, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
        
#         # Transformation matrix for hyperedge features
#         self.W_hyper = nn.Linear(in_features, out_features)
#         self.dropout = dropout
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.out_features)
#         self.W_hyper.weight.data.uniform_(-stdv, stdv)
#         if self.W_hyper.bias is not None:
#             self.W_hyper.bias.data.uniform_(-stdv, stdv)

#     def forward(self, x, H):
#         """
#         x: Node features [N, in_features]
#         H: Hypergraph Incidence Matrix [N, M] (N nodes, M hyperedges)
#            We assume H is row-normalized or passed as a sparse tensor for efficiency if large, 
#            but here we treat it as the attribute matrix directly.
#         """
#         # 1. Node to Hyperedge Aggregation (Message Passing)
#         # hyperedge_feat = H.T @ x
#         # [M, N] @ [N, D] -> [M, D]
#         hyperedge_feat = torch.matmul(H.t(), x)
        
#         # Normalize by hyperedge degree (number of nodes in the hyperedge)
#         # Avoid division by zero
#         edge_deg = H.sum(dim=0, keepdim=True).clamp(min=1e-5)
#         hyperedge_feat = hyperedge_feat / edge_deg.t()

#         # 2. Hyperedge Feature Transformation
#         hyperedge_feat = self.W_hyper(hyperedge_feat)
#         hyperedge_feat = F.relu(hyperedge_feat)
#         hyperedge_feat = F.dropout(hyperedge_feat, self.dropout, training=self.training)

#         # 3. Hyperedge to Node Aggregation
#         # node_feat = H @ hyperedge_feat
#         # [N, M] @ [M, D_out] -> [N, D_out]
#         node_feat = torch.matmul(H, hyperedge_feat)
        
#         # Normalize by node degree (number of hyperedges a node belongs to)
#         node_deg = H.sum(dim=1, keepdim=True).clamp(min=1e-5)
#         node_feat = node_feat / node_deg

#         return node_feat
# # ======================================================


class HypergraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(HypergraphConv, self).__init__()
        self.W_hyper = nn.Linear(in_features, out_features)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W_hyper.weight.size(1))
        self.W_hyper.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, H):
        # H: [N, M]
        # 1. Node -> Edge
        hyperedge_feat = torch.matmul(H.t(), x)
        # Norm
        edge_deg = H.sum(dim=0, keepdim=True).clamp(min=1e-5)
        hyperedge_feat = hyperedge_feat / edge_deg.t()
        
        # Transform
        hyperedge_feat = self.W_hyper(hyperedge_feat)
        hyperedge_feat = F.relu(hyperedge_feat)
        hyperedge_feat = F.dropout(hyperedge_feat, self.dropout, training=self.training)
        
        # 2. Edge -> Node
        node_feat = torch.matmul(H, hyperedge_feat)
        node_deg = H.sum(dim=1, keepdim=True).clamp(min=1e-5)
        node_feat = node_feat / node_deg
        
        return node_feat