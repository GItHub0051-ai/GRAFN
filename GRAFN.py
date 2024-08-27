import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import arange
from config.global_configs import *
import numpy as np
from sklearn.cluster import KMeans
import math
import joblib
from thop import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class GRAFN(nn.Module):
    def __init__(self, ):
        super(GRAFN, self).__init__()
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.hv = SelfAttention(hidden_size)
        self.ha = SelfAttention(hidden_size)
        self.ht = SelfAttention(hidden_size)
        self.ffn = FFN()
        self.gcn = GcnNet(hidden_size)
        self.gat = GraghAttention(hidden_size)
        self.cat_connect = nn.Linear(2 * TEXT_DIM, TEXT_DIM)
        self.weight = nn.Linear(hidden_size, hidden_size)
        self.null = None
        self.null1 = None
        self.lv = selfAttentionV(hidden_size)
        self.la = selfAttentionA(hidden_size)
        self.ll = selfAttentionT(hidden_size)
        self.attn = nn.Sigmoid()

    def GCN(self, adj, x):
        list1 = []
        B, _, _ = adj.shape
        b, _, _ = x.shape
        self.null = torch.zeros(50, 768)
        for i, j in zip(range(B), range(b)):
            shift_gcn = self.gcn(adj[i], x[j]) + self.null.to(device)
            self.null = shift_gcn
            list1.append(shift_gcn.unsqueeze(0))
        connect_shift_gcn = torch.cat(list1, dim=0)

        return connect_shift_gcn

    def GAT(self, adj_a, x_a):
        list2 = []
        A, _, _ = adj_a.shape
        a, _, _ = x_a.shape
        self.null1 = torch.zeros(50, 768)
        for m, n in zip(range(A), range(a)):
            shift_gat = self.gat(adj_a[m], x_a[n]) + self.null1.to(device)
            self.null1 = shift_gat
            list2.append(shift_gat.unsqueeze(0))
        connect_shift_gat = torch.cat(list2, dim=0)

        return connect_shift_gat

    def forward(self, text_embedding, visual_ids=None, acoustic_ids=None):
        # 视觉表示学习
        visual_ = self.visual_embedding(visual_ids)

        # 定义邻接矩阵
        adj_v = self.lv(text_embedding, visual_)

        shift_gcn_vv = self.GAT(visual_, adj_v) + visual_

        # 声学表示学习
        acoustic_ = self.acoustic_embedding(acoustic_ids)

        # 定义邻接矩阵
        adj_aa = self.la(text_embedding, acoustic_)

        shift_gcn_aa = self.GAT(acoustic_, adj_aa) + acoustic_

        # 文本图卷积表示学习
        adj_t = self.ll(text_embedding, text_embedding)
        # exit()
        shift_gcn_tt = self.GCN(adj_t, text_embedding)

        shift_gcn_tt = self.GAT(shift_gcn_tt, adj_t) + shift_gcn_tt

        visual_ = self.hv(shift_gcn_tt, shift_gcn_vv)
        visual_ = self.ffn(visual_)

        # 声学嵌入文本
        acoustic_ = self.ha(shift_gcn_tt, shift_gcn_aa)
        acoustic_ = self.ffn(acoustic_)

        # 基于文本的声视融合
        visual_acoustic = torch.cat((visual_, acoustic_), dim=-1)
        shift = self.cat_connect(visual_acoustic)

        # 残差连接
        embedding_shift = shift + text_embedding

        return embedding_shift

# 图注意力
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout  # dropout参数
        self.in_features = in_features  # 结点向量的特征维度
        self.out_features = out_features  # 经过GAT之后的特征维度
        self.alpha = alpha  # LeakyReLU参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # Learnable weight matrices for linear transformations
        self.W = nn.Parameter(torch.randn(in_features, out_features))
        self.a = nn.Parameter(torch.randn(in_features * 2, 1))
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W.data)
        nn.init.kaiming_uniform_(self.a.data)

    def forward(self, x, adj_matrix):
        # Linear transformation
        h = torch.mm(x, self.W)
        N = h.size()[0]   # 50
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features) #   (50, 50, 1536)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # (50, 50)
        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj_matrix > 0, e, zero_vec)  # [N, N]
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GraghAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GraghAttention, self).__init__()
        self.layer1 = GraphAttentionLayer(hidden_size, hidden_size, 0.1, 0.01)

    def forward(self, x, adj_matrix):
        x = self.layer1(x, adj_matrix)

        return x

# 图卷积实现
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.data)

    def forward(self, adjacency, input_feature):
        # 图卷积公式
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)

        return output

# 图卷积封装
class GcnNet(nn.Module):
    def __init__(self, hidden_size):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(hidden_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, hidden_size)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency.permute(1, 0), h)

        return logits

# 位置编码
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        if d_model % 2 != 0:
            raise ValueError("ERROR!".format(d_model))

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, emb, step=None):

        emb = emb + self.pe[:emb.size(0)]
        emb = self.dropout(emb)
        return emb

# 自注意力
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=1):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.position = PositionalEncoding(TEXT_DIM)
        self.null_32 = torch.zeros(32, 50, 768)
        self.null_46 = torch.zeros(46, 50, 768)
        self.null_101 = torch.zeros(101, 50, 768)
        self.null_128 = torch.zeros(128, 50, 768)
        self.null_14 = torch.zeros(14, 50, 768)
        # self.null_32 = torch.zeros(32, 50, 768)
        # self.null_79 = torch.zeros(79, 50, 768)
        # self.null_51 = torch.zeros(51, 50, 768)
        # self.null_128 = torch.zeros(128, 50, 768)
        self.gama = 1 - np.power(2., (-5 - arange(0, 1)))
        self.linear = nn.Linear(768, 50)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        # 位置编码
        Q = self.position(Q)
        K = self.Wk(embedding)
        # 位置编码
        K = self.position(K)
        V = self.Wv(embedding)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        Q_1 = Q.squeeze(1)
        Q_trans = self.linear(Q_1)

        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score * 8)

        context_layer = torch.matmul(weight_prob, V)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 引入情感衰减因子  MOSI-ONLY
        self.gama = self.gama.astype(np.float32)
        list_context = []
        b, _, _ = context_layer.shape
        for i in range(b):
            context = context_layer[i, ...].reshape(1, 50, 768)
            if i == 0:
                if Q_trans.shape[0] == 32:
                    context_layer_before = self.null_32.cpu().detach().numpy() * self.gama
                elif Q_trans.shape[0] == 46:
                    context_layer_before = self.null_46.cpu().detach().numpy() * self.gama
                elif Q_trans.shape[0] == 14:
                    context_layer_before = self.null_14.cpu().detach().numpy() * self.gama
                elif Q_trans.shape[0] == 101:
                    context_layer_before = self.null_101.cpu().detach().numpy() * self.gama
                elif Q_trans.shape[0] == 128:
                    context_layer_before = self.null_128.cpu().detach().numpy() * self.gama
                else:
                    print('error')
                context_layer_b = torch.matmul(Q_trans, torch.tensor(context_layer_before).to(device)) + context
            else:
                context_layer_before = list_context[i-1].cpu().detach().numpy() * self.gama
                context_layer_b = torch.matmul(Q_trans, torch.tensor(context_layer_before).to(device)) + context
            list_context.append(context_layer_b)
            if len(list_context) <= 1 or len(list_context) <= 128:
                continue
            else:
                context_layer = torch.cat(list_context, dim=1)
        #引入情感衰减因子  MOSEI-ONLY
        # self.gama = self.gama.astype(np.float32)
        # list_context = []
        # b, _, _ = context_layer.shape
        # for i in range(b):
        #     context = context_layer[i, ...].reshape(1, 50, 768)
        #     if i == 0:
        #         if Q_trans.shape[0] == 32:
        #             context_layer_before = self.null_32.cpu().detach().numpy() * self.gama
        #         elif Q_trans.shape[0] == 51:
        #             context_layer_before = self.null_51.cpu().detach().numpy() * self.gama
        #         elif Q_trans.shape[0] == 79:
        #             context_layer_before = self.null_79.cpu().detach().numpy() * self.gama
        #         elif Q_trans.shape[0] == 128:
        #             context_layer_before = self.null_128.cpu().detach().numpy() * self.gama
        #         else:
        #             print('error')
        #         context_layer_b = torch.matmul(Q_trans, torch.tensor(context_layer_before).to(device)) + context
        #     else:
        #         context_layer_before = list_context[i-1].cpu().detach().numpy() * self.gama
        #         context_layer_b = torch.matmul(Q_trans, torch.tensor(context_layer_before).to(device)) + context
        #     list_context.append(context_layer_b)
        #     if len(list_context) <= 51 or len(list_context) <= 128:
        #         continue
        #     else:
        #         context_layer = torch.cat(list_context, dim=1)

        return context_layer


class selfAttentionV(nn.Module):
    def __init__(self, hidden_size):
        super(selfAttentionV, self).__init__()

        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.position = PositionalEncoding(TEXT_DIM)

        # import os
        # print(os.getcwd())

        self.kmeans = joblib.load(filename='kmeans_model.joblib')
        self.linear = nn.Linear(50, 100)

    def forward(self, text_embedding, embedding):
        b, _, _ = text_embedding.shape
        Q = self.Wq(text_embedding)
        # 位置编码
        Q = self.position(Q)
        K = self.Wk(embedding)
        # 位置编码
        K = self.position(K)

        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score)

        weight_prob = self.linear(weight_prob)

        # 使用Kmeans聚类，讲权值划分为0或1
        weight_prob = weight_prob.cpu().detach().numpy()
        weight_prob = weight_prob.reshape(-1, 2)
        adj_matrix = self.kmeans.fit_predict(weight_prob)

        # 还原尺度
        adj_matrix = torch.FloatTensor(adj_matrix.reshape(b, 50, 50)).to(device)

        return adj_matrix

class selfAttentionA(nn.Module):
    def __init__(self, hidden_size):
        super(selfAttentionA, self).__init__()

        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.position = PositionalEncoding(TEXT_DIM)

        # import os
        # print(os.getcwd())

        self.kmeans = joblib.load(filename='E:\\github files\\CENew\\CENet-main\\CENet-main\\networks\\subnet\\kmeans_model_Acoustic.joblib')
        self.linear = nn.Linear(50, 100)

    def forward(self, text_embedding, embedding):
        b, _, _ = text_embedding.shape
        Q = self.Wq(text_embedding)
        # 位置编码
        Q = self.position(Q)
        K = self.Wk(embedding)
        # 位置编码
        K = self.position(K)

        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score)

        weight_prob = self.linear(weight_prob)


        # 使用Kmeans聚类，讲权值划分为0或1
        weight_prob = weight_prob.cpu().detach().numpy()
        weight_prob = weight_prob.reshape(-1, 2)
        adj_matrix = self.kmeans.fit_predict(weight_prob)

        # 还原尺度
        adj_matrix = torch.FloatTensor(adj_matrix.reshape(b, 50, 50)).to(device)

        return adj_matrix

class selfAttentionT(nn.Module):
    def __init__(self, hidden_size):
        super(selfAttentionT, self).__init__()

        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.position = PositionalEncoding(TEXT_DIM)

        # import os
        # print(os.getcwd())

        self.kmeans = joblib.load(filename='E:\\github files\\CENew\\CENet-main\\CENet-main\\networks\\subnet\\kmeans_model_Text.joblib')
        self.linear = nn.Linear(50, 100)

    def forward(self, text_embedding, embedding):
        b, _, _ = text_embedding.shape
        Q = self.Wq(text_embedding)
        # 位置编码
        Q = self.position(Q)
        K = self.Wk(embedding)
        # 位置编码
        K = self.position(K)

        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score)

        weight_prob = self.linear(weight_prob)


        # 使用Kmeans聚类，讲权值划分为0或1
        weight_prob = weight_prob.cpu().detach().numpy()
        weight_prob = weight_prob.reshape(-1, 2)
        adj_matrix = self.kmeans.fit_predict(weight_prob)

        # 还原尺度
        adj_matrix = torch.FloatTensor(adj_matrix.reshape(b, 50, 50)).to(device)

        return adj_matrix

# 前馈神经网络
class FFN(nn.Module):
    def __init__(self, ):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

if __name__ == "__main__":
    model = CE().to(device)
    text = torch.randn(32, 50, 768).cuda()
    visual_ids = abs(torch.randn(32, 50)).long().cuda()
    acoustic_ids = abs(torch.randn(32, 50)).long().cuda()
    output = model(text, visual_ids, acoustic_ids)
    flops, params = profile(model, inputs=(text, visual_ids, acoustic_ids,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print("Total params: %.2fM" % (params / 1e6))
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
    exit()
    print(output.shape)
