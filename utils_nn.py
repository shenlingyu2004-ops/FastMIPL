import torch
import numpy as np
import torch.nn as nn
import scipy.linalg as la
import ot

from dataloader import setup_scatter
from utils import args
from utils_classifier import SoftmaxRegression


class FeatureExtractor(nn.Module):
    def __init__(self, params=args):
        super(FeatureExtractor, self).__init__()
        self.args = params
        self.ds_name = self.args.ds 
        self.ds_suffix = self.args.ds_suffix
        self.nr_fea = self.args.nr_fea
        if 'mnist' in self.ds_name.lower(): # for MNIST, FMNIST dataset
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )
            [self.input_dim, self.H, self.L] = [50 * 4 * 4, 256, 128]

        elif 'sival' in self.ds_name.lower(): # for SIVAL_MIPL dataset
            [self.input_dim, self.H1, self.H2, self.L] = [self.nr_fea, 512, 1024, 256]

        elif 'crc' in self.ds_name.lower() and 'sift' in self.ds_suffix.lower(): # for CRC-MIPL-sift dataset
            [self.input_dim, self.H1, self.H2, self.L] = [self.nr_fea, 1024, 512, 64]
            
        else: # for Birdsong_MIPL, CRC-MIPL (row, sbn, kmeanSegs) dataset
            [self.input_dim, self.H, self.L] = [self.nr_fea, 512, 256]
        
        if 'sival' in self.ds_name.lower() or (
            'crc' in self.ds_name.lower() and 'sift' in self.ds_suffix.lower()):
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.input_dim, self.H1),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.H1, self.H2),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.H2, self.L),
                nn.ReLU(),
                nn.Dropout(),
            )
        else:
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.input_dim, self.H),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.H, self.L),
                nn.ReLU(),
                nn.Dropout(),
            )

    # 在 utils_nn.py 的 FeatureExtractor.forward 中
    def forward(self, xs):
        x = torch.cat(xs, dim=0)
        x, i, i_ptr = setup_scatter(xs)
        # 去掉 squeeze(0)，避免将2维张量压缩为1维
        x = x.float()  # 修正此处
        if 'mnist' in self.ds_name.lower():
            x = x.reshape(x.shape[0], 1, 28, 28)  # 保持实例维度
            h = self.feature_extractor_part1(x)
            h = h.view(-1, 50 * 4 * 4)  # 展平为 (实例数, 特征数)
            h = self.feature_extractor_part2(h)
        else:
            x = x.float()
            h = self.feature_extractor_part2(x)
        hs = []
        for start_idx, end_idx in zip(i_ptr[:-1], i_ptr[1:]):
            bag_fea = h[start_idx:end_idx]
            # 再次确保每个袋特征是2维（即使只有1个实例）
            if bag_fea.dim() == 1:
                bag_fea = bag_fea.unsqueeze(0)
            hs.append(bag_fea)
        return hs



def zb_regress(x, f):
    """
    regresses out x from f
    """
    f_inv = la.pinv(f) 
    b = f_inv.dot(x) 
    x_out = x - f.dot(b)
    out = [x_out, b, f_inv]
    return out

def init_sr(x, s, b, fiv):
    model = SoftmaxRegression(
        eta=0.1,
        epochs=1000,
        minibatches=1,
        random_seed=args.seed,
        print_progress=0,
        n_classes=args.nr_class
    )

    model.fit(x, s)

    alpha = model.b_[None]
    gamma = model.w_ 

    # Compute bag prediction u and reparametrize
    u = x.dot(gamma) 
    [um, us] = [u.mean(0)[None], u.std(0)[None]]
    alpha = alpha + um
    mu_gamma = us * gamma / np.sqrt((gamma**2).mean(0)[None])
    sd_gamma = np.sqrt(0.1 * (mu_gamma**2).mean()) * np.ones_like(mu_gamma)
    alpha = fiv.dot(np.ones((fiv.shape[1], 1))).dot(alpha) - b.dot(mu_gamma)

    # init prior
    var_z = (mu_gamma**2 + sd_gamma**2).mean(0).reshape(1, -1)

    return [torch.Tensor(el) for el in (mu_gamma, sd_gamma, var_z, alpha)]

def generate_init_params(x, fe, s, topology_list):
    eps = 1e-8
    z_b = np.concatenate([x.mean(0, keepdims=True) for x in x], axis=0) # generating features of each bag
    [fe_array, s_array] = [fe.numpy(), s.numpy()]

    zb, b, fe_inv = zb_regress(z_b, fe_array)
    nor_zb = (zb - zb.mean(0, keepdims=True)) / (zb.std(0, keepdims=True) * np.sqrt(zb.shape[-1]) + eps)

    mu_z, sd_z, var_z, alpha = init_sr(nor_zb, s_array, b, fe_inv)
    # 1. 实例分布与标签分布的OT映射
    x_dists = [feature_to_distribution(torch.from_numpy(xi)) for xi in x]  # 实例分布
    s_dists = [feature_to_distribution(si) for si in s]  # 标签分布
    # 计算初始运输计划（基于欧氏距离）
    init_pi = []
    for x_dist, s_dist in zip(x_dists, s_dists):
        M = ot.dist(x_dist.numpy().reshape(-1,1), s_dist.numpy().reshape(-1,1))  # 成本矩阵
        pi = ot.emd(x_dist.numpy(), s_dist.numpy(), M)  # 精确OT
        init_pi.append(torch.tensor(pi))
    # 2. 拓扑约束初始化（近邻平滑）
    k = 5  # K近邻数
    topo_sim = torch.cdist(torch.stack(topology_list), torch.stack(topology_list))  # 拓扑相似度矩阵
    knn_idx = torch.topk(topo_sim, k, largest=False).indices  # K近邻索引
    # 基于近邻的初始权重平滑
    init_weights = torch.ones(len(x))
    for i in range(len(x)):
        init_weights[i] = init_weights[knn_idx[i]].mean()  # 近邻平均权重
    # 3. 初始化模型参数（融入OT和拓扑）
    mu_z, sd_z, var_z, alpha = init_sr(nor_zb, s_array, b, fe_inv)
    alpha = alpha * init_weights.unsqueeze(1)  # 拓扑权重修正alpha
    return mu_z, sd_z, var_z, alpha, init_pi  # 返回OT运输计划

def feature_to_distribution(features, n_bins=100):
    """将实例特征转为离散概率分布（直方图）"""
    # 计算所有元素的全局最小值和最大值，并转为标量
    min_val = features.min().item()  # 取所有元素的最小值，转为Python数字
    max_val = features.max().item()  # 取所有元素的最大值，转为Python数字
    # 计算直方图（使用标量作为min和max参数）
    dist = torch.histc(features, bins=n_bins, min=min_val, max=max_val)
    return dist / dist.sum()  # 归一化
