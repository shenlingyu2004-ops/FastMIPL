import torch
import numpy as np
import torch.optim as optim
import ot

from torch.distributions.kl import kl_divergence
from torch_scatter import scatter_softmax, segment_add_csr
from torch.distributions import LowRankMultivariateNormal

from utils import logging, device
from dataloader import setup_scatter, topolpgy_list
from utils_nn import generate_init_params
from utils_nn import FeatureExtractor
from utils_nn import feature_to_distribution
from utils_posterior import BayesianPosterior


class FastMIPL(torch.nn.Module):
    def __init__(self, cfg, params=[None, 1, 1]):
        super().__init__()
        self.args = cfg
        self.predict_flag = False
        self.nr_class = self.args.nr_class
        self.nr_samples = self.args.nr_samples
        self.fea_extractor = FeatureExtractor(self.args)
        [init_params, self.dim_xs, nr_fixed_effects] = params

        ln_sigma_u = torch.full((1, self.nr_class), 0.5 * np.log(0.5))
        if init_params is not None:
            *_, var_z, alpha = init_params
            ln_sigma_z = 0.5 * torch.log(var_z)
        else:
            alpha = torch.zeros((nr_fixed_effects, self.nr_class))
            ln_sigma_z = torch.full((1, self.nr_class), 0.5 * np.log(0.5))

        [self.alpha, self.ln_sigma_u, self.ln_sigma_z, self.posterior] = [
            torch.nn.Parameter(alpha), 
            torch.nn.Parameter(ln_sigma_u), 
            torch.nn.Parameter(ln_sigma_z), 
            BayesianPosterior(self.dim_xs, self.nr_class, init_params)
        ]

        self.topo_smooth = cfg.topo_smooth  # 拓扑平滑系数（新增配置）
        self.init_pi = params[-1]  # 接收OT初始运输计划

    def initialize_model(x, fe, s, cfg_params):
        fea_extractor = FeatureExtractor(cfg_params)
        xs = fea_extractor(x)
        # 确保每个元素都是张量（若为列表则拼接为张量），再转换为numpy
        xs_array = []
        for i_x in xs:
            # 若元素是列表，先拼接成一个张量；否则直接使用
            if isinstance(i_x, list):
                tensor_x = torch.cat(i_x)  # 假设列表内是同维度张量，可拼接
            else:
                tensor_x = i_x
            xs_array.append(tensor_x.detach().numpy())
        init_params = generate_init_params(xs_array, fe, s ,topology_list)
        model_params = [init_params, xs_array[0].shape[1], fe.shape[1]]
        return FastMIPL(cfg_params, model_params)

    def ot_denoise_loss(self, p_ij, x, knn_idx):
        """实例置信度的OT去噪+拓扑平滑损失"""
        # 1. OT运输成本（最小化实例-标签映射成本）
        M = ot.dist(x, self.label_prototypes)  # 实例到标签原型的成本矩阵
        ot_cost = torch.sum(p_ij * M)
        # 2. 拓扑平滑项（近邻实例置信度一致）
        topo_loss = 0.0
        for i in range(p_ij.shape[0]):
            topo_loss += torch.norm(p_ij[i] - p_ij[knn_idx[i]].mean(dim=0)) ** 2
        return ot_cost + self.topo_smooth * topo_loss

    @property
    def prior_dist(self):
        [scale_u, scale_z] = [
            self.ln_sigma_u.T * torch.ones([1, self.dim_xs], device=device),
            self.ln_sigma_z.T * torch.ones([1, self.dim_xs], device=device)
        ]
        cov_ldiag = torch.cat([scale_u, scale_z], 1)
        [cov_factor, mu] = [
            torch.zeros_like(cov_ldiag)[:, :, None],
            torch.zeros_like(cov_ldiag)
        ]
        prior_dist = LowRankMultivariateNormal(mu, cov_factor, torch.exp(2 * cov_ldiag))
        return prior_dist

    def forward(self, xs, x_dists, topology):
        beta_u, beta_z = self.posterior.get_beta(self.nr_samples, self.predict_flag)

        b = torch.sqrt((beta_z ** 2).mean(0, keepdim=True))
        eta = beta_z / b

        # 计算实例分布与包拓扑分布的瓦瑟斯坦距离（替换原有注意力权重计算）
        # 1. 转换拓扑结构为分布
        bag_topo_dist = feature_to_distribution(topology)  # 假设该函数已实现，返回与x_dists同维度的分布
        # 2. 计算每个实例分布与拓扑分布的瓦瑟斯坦距离
        wasserstein_dists = []
        for xd in x_dists:
            # 确保输入为numpy数组，计算1D瓦瑟斯坦距离
            dist = ot.wasserstein_1d(xd.cpu().numpy(), bag_topo_dist.cpu().numpy())
            wasserstein_dists.append(dist)
        # 3. 距离越小权重越大（通过softmax(-距离)实现）
        w_raw = -torch.tensor(wasserstein_dists, device=xs[0].device, dtype=torch.float32)
        # 4. 适配后续聚合的维度（与原有w的形状对齐：[i, p, s]，这里假设p=1, s=1简化处理，可根据实际调整）
        w = torch.softmax(w_raw, dim=0).unsqueeze(1).unsqueeze(2)  # 扩展维度以匹配后续操作

        # 特征转换与聚合（保留原有逻辑）
        x, i, i_ptr = setup_scatter(xs)
        t = torch.einsum("iq,qps->ips", x, eta)  # 实例特征转换
        z_bag = segment_add_csr(w * t, i_ptr)  # 基于新权重的袋级特征聚合

        # 标准化处理（保留原有逻辑）
        mean, std = z_bag.mean(0), z_bag.std(0)
        if std.isnan().any():
            std = 1.0
        z_bag = b * (z_bag - mean) / std
        return z_bag
    
    def calculate_loss(self, u, fe, s, kld_w):
        # Equation (3)
        logits = fe.mm(self.alpha).unsqueeze(2) + u
        logits_d = logits.permute(0, 2, 1)
        # calculate components of loss function
        link_func = torch.sum(
            torch.sum(
                logits_d * s.unsqueeze(1).expand(-1, self.nr_samples, -1), 
                dim=[1, 2]
                )
            ) / s.shape[0]
        # Equation (7)
        posterior_dist = self.posterior.distribution
        kld = kl_divergence(posterior_dist, self.prior_dist)
        kld_term = kld_w * kld.sum() / s.shape[0]
        # Equation (6)
        loss = - link_func + kld_term

        res_dict = {
            "loss": round(loss.item(), 4), 
            "ll": round(link_func.item(), 4), 
            "kld": round(kld_term.item(), 4)
        }
        return loss, res_dict
    
    def calculate_obj(self, data, fe, s_label, ratio_kld):
        xs = self.fea_extractor(data)
        z_b = self(xs) 
        loss, res = self.calculate_loss(z_b, fe, s_label, kld_w=ratio_kld)
        return loss, res, z_b

    def regenerate_s(self, s, zb, w_conf, x_dists, topology):
        # 1. OT优化后的实例置信度
        p_ij = self.ot_denoise(zb, x_dists)  # 调用OT去噪函数
        # 2. 包级聚合（融入拓扑匹配度）
        bag_topo_dist = feature_to_distribution(topology)
        w = torch.softmax(-torch.tensor([ot.wasserstein_1d(xd.numpy(), bag_topo_dist.numpy()) for xd in x_dists]),
                          dim=0)
        q_i = torch.sum(w.unsqueeze(1) * p_ij, dim=0)  # 包级置信度
        # 3. 标签消歧（结合原始标签和OT优化结果）
        new_s = w_conf * s + (1 - w_conf) * q_i
        return new_s
    
    def train(self, train_loader, num_bags, weight_list):

        [lr_value, n_epochs, reg_value] = [self.args.lr, self.args.epochs, self.args.reg]
        optimizer = optim.SGD(self.parameters(), lr=lr_value, momentum=0.9, 
                              nesterov=True, weight_decay=reg_value)

        for epoch in range(n_epochs):
            for step, (x, s, _, cov_b, idx) in enumerate(train_loader):
                i_epoch = epoch + 1
                ratio_kld = len(x) / num_bags

                loss, res, z_bag = self.calculate_obj(x, cov_b, s, ratio_kld)
                res["epoch"], res["step"] = i_epoch, step
                if epoch == 0 or (i_epoch) % 10 == 0:
                    logging.info("Loss Dict: {}".format(res))

                ## Dynamic Disambiguation
                alpha_value = weight_list[epoch]
                new_s = self.regenerate_s(s, z_bag, alpha_value)
                train_loader.dataset.partial_bag_lab[idx] = new_s

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.predict_flag = True
        return
    
    @torch.inference_mode()
    def predict(self, xs):
        self.nr_samples = None
        # Equation (9)
        xs = self.fea_extractor(xs)
        s_hat = self(xs).squeeze(2)
        s_logits = torch.softmax(s_hat, dim = 1)
        y_pred = torch.max(s_logits.data, 1)[1]
        return y_pred

def explain_key_instances(self, xs, x_dists, topology, y_true):
    """输出关键实例的多维度解释"""
    # 1. 贡献度权重（OT匹配度）
    w = self.forward(xs, x_dists, topology)[1]  # 获取实例权重
    # 2. 拓扑近邻一致性
    topo_sim = torch.cdist(torch.stack([feature_to_distribution(x) for x in xs]),
                          torch.stack([feature_to_distribution(x) for x in xs]))
    knn_idx = torch.topk(topo_sim, 5, largest=False).indices
    topo_consistency = [torch.norm(p_ij[i] - p_ij[knn_idx[i]].mean(dim=0))**2 for i in range(len(xs))]
    # 3. OT运输成本（到真实标签）
    true_label_dist = feature_to_distribution(y_true)
    ot_costs = [ot.wasserstein_1d(xd.numpy(), true_label_dist.numpy()) for xd in x_dists]
    # 生成解释报告
    explanations = []
    for i in range(len(xs)):
        explanations.append({
            "instance_idx": i,
            "weight": w[i].item(),
            "topo_consistency": topo_consistency[i].item(),
            "ot_cost_to_true_label": ot_costs[i],
            "is_key": (w[i] > 0.1) and (topo_consistency[i] < 0.05) and (ot_costs[i] < 0.1)
        })
    return explanations