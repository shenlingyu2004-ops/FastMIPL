#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
import random
import torch
import scipy.io as io
import torch.utils.data as data_utils
import gudhi as gd
 
def to_categorical(y, nr_class):
    '''
    generate one-hot label
    '''
    y_list = [0] * nr_class
    for i in y:
        y_list[i] = 1
    y_cate = np.array(y_list)

    return y_cate


def load_data_mat(mat_path, nr_fea, nr_class, normalize=False):
    '''
    load the dataset in mat format 
    '''
    data_mat = io.loadmat(mat_path)
    data = data_mat['data']
    all_ins_fea = []
    all_ins_fea_tmp = np.empty((0, nr_fea))
    ins_num, bag_idx_of_ins = [], []
    bag_lab, dummy_ins_lab = [], []
    partial_bag_lab = []
    partial_dummy_ins_lab = np.empty((0, nr_class))
    partial_dummy_ins_lab_processed = np.empty((0, nr_class))
    partial_bag_lab, partial_bag_lab_processed = np.empty((0, nr_class)), np.empty((0, nr_class))
    bag_cnt = 1
    for i in range(data.shape[0]):
        all_ins_fea = np.vstack((all_ins_fea_tmp, data[i, 0]))
        all_ins_fea_tmp = all_ins_fea
        ins_num_tmp = data[i, 0].shape[0]
        ins_num.append(ins_num_tmp)
        bag_idx_of_ins_tmp = [bag_cnt] * ins_num_tmp
        bag_idx_of_ins = bag_idx_of_ins + bag_idx_of_ins_tmp
        bag_cnt += 1
        # the ground-truth labels of bags
        bag_lab_tmp = list(data[i, 2].flatten() - 1)
        bag_lab = bag_lab + bag_lab_tmp
        dummy_ins_lab_tmp = [bag_lab_tmp] * ins_num_tmp
        dummy_ins_lab = dummy_ins_lab + dummy_ins_lab_tmp
        # the partial labels of bags
        partial_bag_lab_tmp = list(data[i, 1].flatten() - 1)
        partial_bag_lab_tmp = to_categorical(partial_bag_lab_tmp, nr_class)
        partial_bag_lab_tmp = np.expand_dims(partial_bag_lab_tmp, axis=0)
        partial_bag_lab = np.vstack((partial_bag_lab, partial_bag_lab_tmp))
        partial_dummy_ins_lab_tmp = partial_bag_lab_tmp.repeat(ins_num_tmp, axis=0)
        partial_dummy_ins_lab = np.vstack((partial_dummy_ins_lab, partial_dummy_ins_lab_tmp))

    bag_idx_of_ins = np.array(bag_idx_of_ins)
    bag_idx_of_ins = np.expand_dims(bag_idx_of_ins, axis=1)
    bag_lab = np.array(bag_lab)
    dummy_ins_lab = np.array(dummy_ins_lab)
    lab_inx_fea = np.hstack((dummy_ins_lab, bag_idx_of_ins, all_ins_fea))
    nr_partial_lab_per_ins = np.expand_dims(np.sum(partial_dummy_ins_lab, 1), axis=1)
    partial_dummy_ins_lab_processed = partial_dummy_ins_lab / nr_partial_lab_per_ins
    nr_partial_lab_per_bag = np.expand_dims(np.sum(partial_bag_lab, 1), axis=1)
    partial_bag_lab_processed = partial_bag_lab / nr_partial_lab_per_bag

    if normalize:
        data_mean, data_std = np.mean(all_ins_fea, 0), np.std(all_ins_fea, 0)
        data_min, data_max = np.min(all_ins_fea, 0), np.max(all_ins_fea, 0)
        all_ins_fea_norm = (all_ins_fea - data_mean) / data_std
        all_ins_fea = all_ins_fea_norm

    all_ins_fea = torch.from_numpy(all_ins_fea)
    bag_idx_of_ins = torch.from_numpy(bag_idx_of_ins)
    dummy_ins_lab = torch.from_numpy(dummy_ins_lab)
    bag_lab = torch.from_numpy(bag_lab)
    partial_bag_lab = torch.from_numpy(partial_bag_lab)
    partial_bag_lab_processed = torch.from_numpy(partial_bag_lab_processed)

    return all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed


def load_idx_mat(idx_file):
    '''
    load the index in mat format
    '''
    idx = io.loadmat(idx_file)
    idx_tr_np = idx['trainIndex']
    idx_te_np = idx['testIndex']
    idx_tr = list(np.array(idx_tr_np).flatten())
    idx_te = list(np.array(idx_te_np).flatten())
    random.shuffle(idx_tr)
    random.shuffle(idx_te)

    return idx_tr, idx_te

def compute_topological_features(bag, n_dim=2):
    """计算单个包的持久同调特征"""
    # 构建 Vietoris-Rips 复形
    rips_complex = gd.RipsComplex(points=bag.numpy(), max_edge_length=0.5)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=n_dim)
    # 计算持久同调（出生-死亡时间）
    persistence = simplex_tree.persistence()
    # 提取拓扑特征（如各维度贝蒂数、持久熵等）
    betti_numbers = simplex_tree.betti_numbers()  # 贝蒂数（0:连通分量, 1:孔洞数）
    persist_entropy = gd.persistent_entropy(persistence)  # 持久熵
    return np.concatenate([betti_numbers, [persist_entropy]])

class MIPLDataset(data_utils.Dataset):
    def __init__(self, bags_list, partial_bag_lab, true_bag_lab, F):
        self.bags_list = bags_list
        self.partial_bag_lab = partial_bag_lab
        self.true_bag_lab = true_bag_lab
        self.f = F
    
    def __len__(self):
        return len(self.true_bag_lab)
    
    def __getitem__(self, index):
        bag = self.bags_list[index]
        partial_bag_label = self.partial_bag_lab[index]
        true_bag_label = self.true_bag_lab[index]
        i_f = self.f[index]
        return bag, partial_bag_label, true_bag_label, i_f, index


def create_bags(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab,
                partial_bag_lab, idx_list):
    bags_list = []  # 存储每个袋的实例特征
    ins_lab = torch.empty(0, dtype=torch.uint8)  # 所有实例的标签
    y_bag_partial = torch.empty(0, dtype=torch.float64)  # 袋的部分标签
    true_bag_lab = torch.empty(0, dtype=torch.uint8)  # 袋的真实标签
    bag_features = []  # 存储每个袋的统计特征

    for i in idx_list:
        # 提取当前袋的实例索引和特征
        bag_idx_of_ins_a_bag = bag_idx_of_ins == i
        bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag)
        bag = all_ins_fea[bag_idx_of_ins_a_bag, :]  # 形状：[实例数, 特征维度]
        bags_list.append(bag)

        # 收集实例标签、袋的部分标签和真实标签（与原逻辑一致）
        ins_lab = torch.cat((ins_lab,
                             dummy_ins_lab[bag_idx_of_ins_a_bag].squeeze(1)), 0)
        y_bag_partial = torch.cat((y_bag_partial,
                                   partial_bag_lab[i - 1, :].unsqueeze(0)), 0)
        true_bag_lab = torch.cat((true_bag_lab,
                                  bag_lab[i - 1].view(1, 1)), 0)

        # 计算袋级统计特征
        bag_mean = bag.mean(dim=0)  # 均值
        bag_max, _ = bag.max(dim=0)  # 最大值
        bag_min, _ = bag.min(dim=0)  # 最小值
        bag_std = bag.std(dim=0)  # 标准差（反映离散程度）
        bag_median = bag.median(dim=0).values  # 中位数（抗极端值）
        bag_range = bag_max - bag_min  # 极差（分布范围）
        bag_mean_abs_dev = torch.abs(bag - bag_mean).mean(dim=0)  # 平均绝对偏差（抗极端值）

        # 拼接所有统计特征（按需求选择，此处保留全部以展示）
        current_bag_feat = torch.cat([
            bag_mean, bag_max, bag_min,
            bag_std, bag_median,
            bag_range, bag_mean_abs_dev
        ], dim=0)

        bag_features.append(current_bag_feat)

    return bags_list, ins_lab, y_bag_partial, true_bag_lab, bag_features


def mil_collate_fn(batch):
    X = [item[0] for item in batch]
    S = torch.stack([item[1] for item in batch])
    Y = torch.stack([item[2] for item in batch])
    F = torch.stack([item[3] for item in batch])
    index = [item[4] for item in batch]
    return X, S, Y, F, index

def setup_scatter(Xs):
    device = Xs[0].device
    x = torch.cat(Xs, dim=0)
    i = torch.cat([torch.full((x.shape[0],), idx) for idx, x in enumerate(Xs)]).to(device)
    i_ptr = torch.cat([torch.tensor([0], device=device), i.bincount().cumsum(0)]) # recording the start position of each bag
    return x, i, i_ptr