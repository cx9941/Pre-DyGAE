import sys
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data.knowledge_graph import FB15k237Dataset, WN18Dataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import RelGraphConv
from mydataset import JDDataset
import json
from dgl.nn.pytorch import GraphConv, RelGraphConv
from utils import pearson_correlation_coefficient
from transformers.modeling_outputs import ModelOutput
from torchmetrics.regression import TweedieDevianceScore
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import scipy.stats as stats
from utils import TweedieLoss, info_nce_loss
from models.graphormer import Graphormer
from models.rgcn import RGCN
from models.vgae import VGAEModel
from models.biasd_mha import BiasedMHA
from models.time_emb import TemporalEmb
from sklearn.metrics import f1_score, classification_report
from dgl.nn.pytorch import GraphormerLayer
import copy

class LinkPredict(nn.Module):
    def __init__(self, num_nodes, pos_num_nodes, skill_num_nodes, num_rels, cross_attn="yes", embedding=None, h_dim=768, reg_param=0.01, rg_weight=1, lp_weight=1, rank_weight=1, con_weight=1, diff_weight = 0, rg_loss_fn=None, rg_activate_fn=None, gaussian="yes", bias="yes", initial_embedding="yes", time="no", num_heads=4, real_id_nodes=None, e_dim=10, adaptive='yes'):
        super().__init__()
        self.embedding = embedding
        self.emb = nn.Embedding(num_nodes, h_dim)

        if embedding is not None and initial_embedding == "yes":
            self.emb.weight = nn.Parameter(
                embedding.clone().detach().requires_grad_(True))

        self.bias_attn = BiasedMHA(768, num_heads)
        self.cross_attn = cross_attn

        self.rgcn = RGCN(h_dim, num_rels * 2)
        self.sigma_emb = RGCN(h_dim, num_rels * 2)
        self.mu_emb = RGCN(h_dim, num_rels * 2)

        self.temporal_emb_mu = RGCN(h_dim, num_rels * 2)
        self.temporal_emb_sigma = RGCN(h_dim, num_rels * 2)

        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(
            self.w_relation, gain=nn.init.calculate_gain("relu")
        )
        self.rg_weight = rg_weight
        self.lp_weight = lp_weight
        self.rank_weight = rank_weight
        self.con_weight = con_weight
        self.diff_weight = diff_weight
        if rg_loss_fn == None:
            self.rg_loss_fn = F.l1_loss
        else:
            self.rg_loss_fn = rg_loss_fn

        if rg_activate_fn is None:
            self.rg_activate_fn = nn.ReLU()
        else:
            self.rg_activate_fn = rg_activate_fn

        self.pos_num_nodes = pos_num_nodes
        self.skill_num_nodes = skill_num_nodes
        self.num_nodes = num_nodes

        self.fc_lp = nn.Linear(h_dim, h_dim)

        self.fc_rg = nn.Linear(h_dim, 1)
        self.h_dim = h_dim

        self.down_sampling = nn.Linear(h_dim, 5)
        self.num_heads = num_heads
        self.gaussian = gaussian
        self.bias = bias
        self.time = time

        if time == 'yes':
            self.temporal_emb = TemporalEmb(num_nodes, h_dim, e_dim=e_dim, real_id_nodes=real_id_nodes, adaptive=adaptive)
        self.load_node_embed = False
        self.load_node_temporal_embed = False

        self.fc_diff = nn.Linear(h_dim, 4)
    
    def load_node_embedding(self, mu_embedding, sigma_embedding):
        self.mu_embedding = nn.Embedding(self.num_nodes, self.h_dim)
        self.sigma_embedding = nn.Embedding(self.num_nodes, self.h_dim)
        self.mu_embedding.weight = nn.Parameter(mu_embedding)
        self.sigma_embedding.weight = nn.Parameter(sigma_embedding)
        self.load_node_embed = True
    
    def load_node_temporal_embedding(self, mu_time_embedding, sigma_time_embedding):
        self.mu_time_embedding = nn.Embedding(self.num_nodes, self.h_dim)
        self.sigma_time_embedding = nn.Embedding(self.num_nodes, self.h_dim)
        self.mu_time_embedding.weight = nn.Parameter(mu_time_embedding)
        self.sigma_time_embedding.weight = nn.Parameter(sigma_time_embedding)
        self.load_node_temporal_embed = True

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def calc_rg_score(self, input_embed, triplets):
        triplets = triplets.to(torch.long).to(input_embed.device)
        row_indices = triplets[:, 0]
        col_indices = triplets[:, 2]
        s = input_embed[row_indices]
        r = self.w_relation[triplets[:, 1]]
        o = input_embed[col_indices]
        rg_origin_score = self.fc_rg(s * r * o).squeeze(-1)
        rg_score = self.rg_activate_fn(rg_origin_score)
        return rg_score

    def calc_lp_score(self, input_embed, triplets):
        triplets = triplets.to(torch.long).to(input_embed.device)
        row_indices = triplets[:, 0]
        col_indices = triplets[:, 2]
        s = input_embed[row_indices]
        r = self.w_relation[triplets[:, 1]]
        o = input_embed[col_indices]
        lp_score = torch.sum(s * r * o, dim=1)
        return lp_score
    
    def encoder(self, g, nids, std):
        if not self.load_node_embed:
            x = self.emb(nids)

            if self.cross_attn == 'yes':
                # gain the bias
                if self.bias == "yes":
                    down_embedding = self.down_sampling(
                        self.embedding.to(x.device))
                    attn_bias = torch.cosine_similarity(down_embedding.unsqueeze(
                        1), down_embedding.unsqueeze(0), dim=-1).unsqueeze(-1).repeat(1, 1, self.num_heads)
                    attn_bias = attn_bias[nids[nids < self.pos_num_nodes]][:, nids[nids >= self.pos_num_nodes]].unsqueeze(0)
                    attn_bias = F.normalize(attn_bias)
                else:
                    attn_bias = None

                pos_embedding = x[nids < self.pos_num_nodes].unsqueeze(0)
                skill_embedding = x[nids >= self.pos_num_nodes].unsqueeze(0)
                pos_embedding = self.bias_attn(
                    pos_embedding, skill_embedding, skill_embedding, attn_bias=attn_bias).squeeze(0)
                x[nids < self.pos_num_nodes] = F.normalize(
                    pos_embedding) + x[nids < self.pos_num_nodes]
                # x[nids >= self.pos_num_nodes] = self.skill_attn(x[nids >= self.pos_num_nodes], x[nids >= self.pos_num_nodes], x[nids >= self.pos_num_nodes])[0] + x[nids >= self.pos_num_nodes]

            embedding = self.rgcn(g, x)

        # add gaussian noise
        if self.gaussian == "yes":
            if self.load_node_embed:
                mu_embedding = self.mu_embedding(nids)
                sigma_embedding = self.sigma_embedding(nids)
            else:
                mu_embedding = self.mu_emb(g, embedding)
                sigma_embedding = self.sigma_emb(g, embedding)

            # time_emb
            if self.time == 'yes':
                if self.load_node_temporal_embed:
                    mu_time_embdding, sigma_time_embdding = self.mu_time_embedding(nids), self.sigma_time_embedding(nids)
                else:
                    mu_time_embdding, sigma_time_embdding = self.temporal_emb(nids, g)
                    # mu_time_embdding, sigma_time_embdding = self.temporal_emb_mu(g, embedding), self.temporal_emb_sigma(g, embedding)

                mu = mu_embedding + mu_time_embdding
                sigma = sigma_embedding + sigma_time_embdding
            else:
                mu_time_embdding = sigma_time_embdding = None
                mu = mu_embedding
                sigma = sigma_embedding

            if std == "yes":
                gaussian_noise = torch.randn(self.pos_num_nodes + self.skill_num_nodes, self.h_dim).to(nids.device)[nids]
                embedding = mu + gaussian_noise * torch.exp(sigma).to(nids.device)
            else:
                embedding = mu

        return embedding, (mu_embedding, sigma_embedding), (mu_time_embdding, sigma_time_embdding), (mu, sigma)


    def forward(self, g, nids, triplets, labels=None, edge_labels=None, skill2cluster=None, std="yes", diff_labels=None):        
        embedding, (mu_embedding, sigma_embedding), (mu_time_embdding, sigma_time_embdding), (mu, sigma) = self.encoder(g, nids, std)
            
        # skill cluster constrastive learning
        if self.con_weight != 0:
            skill2labels = skill2cluster[nids[nids >= self.pos_num_nodes]]
            skill2embedding = embedding[nids >= self.pos_num_nodes]
            con_loss = info_nce_loss(skill2embedding, skill2labels)
        else:
            con_loss = 0

        # edge classification task
        triplets = triplets.to(torch.long).to(embedding.device)

        if self.lp_weight != 0:
            lp_score = self.calc_lp_score(embedding, triplets)

            if (labels == 0).sum() == 0:
                lp_loss = 0
            else:
                lp_loss = F.binary_cross_entropy_with_logits(lp_score, labels)
        else:
            lp_loss = 0

        # regression loss
        if self.rg_weight != 0:
            rg_score = self.calc_rg_score(embedding, triplets)
            rg_loss = self.rg_loss_fn(rg_score, edge_labels)
        else:
            rg_loss = 0

        # reg_loss
        reg_loss = self.regularization_loss(embedding)

        # ## extra test
        # from utils import time_metric
        # from sklearn.metrics import roc_auc_score
        # device = "cpu"
        # self = self.to(device)
        # gold_triplets = torch.load('gold_triplets.pt')
        # gold_labels = torch.load('gold_labels.pt')
        # embed = torch.randn(self.num_nodes, 768)
        # embed[nids] = embedding.to(device)
        # all_triplets_pos = torch.range(0, self.pos_num_nodes - 1).repeat_interleave(self.skill_num_nodes).unsqueeze(0)
        # all_triplets_skill = torch.range(self.pos_num_nodes, self.skill_num_nodes + self.pos_num_nodes - 1).repeat(self.pos_num_nodes).unsqueeze(0)
        # all_triplets_relation = torch.zeros(all_triplets_skill.shape[1]).unsqueeze(0)
        # all_triplets = torch.cat([all_triplets_pos, all_triplets_relation, all_triplets_skill], dim=0).T.long().to(device)
        # all_lp_score = self.calc_lp_score(embed, all_triplets)
        # all_rg_score = self.calc_rg_score(embed, all_triplets)
        # all_rg_matrix = torch.zeros(self.pos_num_nodes, self.skill_num_nodes).to(all_rg_score.device)
        # all_rg_matrix[all_triplets[:,0], all_triplets[:,2] - self.pos_num_nodes] = all_rg_score
        # all_lp_matrix = torch.zeros(self.pos_num_nodes, self.skill_num_nodes).to(all_lp_score.device)
        # all_lp_matrix[all_triplets[:,0], all_triplets[:,2] - self.pos_num_nodes] = all_lp_score
        # all_labels = torch.zeros(self.pos_num_nodes, self.skill_num_nodes).to(all_rg_score.device)
        # all_labels[gold_triplets[:,0], gold_triplets[:,2] - self.pos_num_nodes] = gold_labels.to(device)
        # metric = time_metric(all_labels, all_rg_matrix)
        # temp1_rg_score = all_rg_matrix[gold_triplets[:,0], gold_triplets[:,2] - self.pos_num_nodes]
        # temp2_rg_score = rg_score[edge_labels!=0]
        # temp1_auc = roc_auc_score(edge_labels.int().detach().cpu().numpy(), rg_score.detach().cpu().numpy())
        # temp2_auc = roc_auc_score((all_labels!=0).view(-1).int().detach().cpu().numpy(), all_rg_matrix.view(-1).detach().cpu().numpy())
        # ori_src, ori_dst = torch.load('ori_index.pt')
        # ori_src, ori_dst = ori_src.cpu(), ori_dst.cpu()
        # ori_src.unique()
        # import copy
        # real_triplets = copy.deepcopy(triplets)
        # real_triplets[:,0] = ori_src.unique()[triplets[:,0].cpu()]
        # real_triplets[:,2] = ori_dst.unique()[(triplets[:,2] - triplets[:,2].min()).cpu()]
        # ori_src.unique()[triplets[:,0].cpu()]

        # real_preds = all_rg_matrix[real_triplets[:,0].cpu(), real_triplets[:,2].cpu() - self.pos_num_nodes]
        # roc_auc_score(edge_labels.int().detach().cpu().numpy(), real_preds.detach().cpu().numpy())
        # ## extra test end

        # rankingloss
        if self.rank_weight != 0:
            criterion = nn.MarginRankingLoss(margin=1.0)
            score_level1 = lp_score[edge_labels > 0.1]
            score_level2 = lp_score[(0.1 > edge_labels) & (edge_labels > 0.01)]
            score_level3 = lp_score[edge_labels < 0.01]
            score_level2_3 = lp_score[0.1 > edge_labels]

            score_level1 = score_level1[torch.randperm(score_level1.shape[0])[
                :score_level1.shape[0] // 2]]
            score_level2 = score_level2[torch.randperm(score_level2.shape[0])[
                :score_level2.shape[0] // 2]]
            score_level3 = score_level3[torch.randperm(score_level3.shape[0])[
                :score_level3.shape[0] // 2]]
            score_level2_3 = score_level2_3[torch.randperm(score_level2_3.shape[0])[
                :score_level2_3.shape[0] // 2]]

            pos_sample1 = score_level1.repeat_interleave(
                score_level2_3.shape[0])
            neg_sample1 = score_level2_3.repeat(score_level1.shape[0])
            rank_loss1 = criterion(pos_sample1, neg_sample1, torch.ones(
                pos_sample1.shape).to(pos_sample1.device))

            pos_sample2 = score_level2.repeat_interleave(score_level3.shape[0])
            neg_sample2 = score_level3.repeat(score_level2.shape[0])
            rank_loss2 = criterion(pos_sample2, neg_sample2, torch.ones(
                pos_sample2.shape).to(pos_sample2.device))

            rank_loss = rank_loss1 + rank_loss2
        else:
            rank_loss = 0

        if self.gaussian == "yes":
            from utils import kl_div
            kl_loss = kl_div(sigma, mu, nids)
        else:
            kl_loss = 0

        diff_loss = 0
        if self.diff_weight != 0:
            s = mu_time_embdding[triplets[:, 0]]
            r = self.w_relation[triplets[:, 1]]
            o = mu_time_embdding[triplets[:, 2]]
            time_diff = self.fc_diff(s * r * o).squeeze(-1)
            diff_loss = nn.CrossEntropyLoss()(time_diff, diff_labels)

        loss = self.lp_weight * lp_loss + self.reg_param * reg_loss + rg_loss * self.rg_weight + rank_loss * self.rank_weight + con_loss * self.con_weight + kl_loss + self.diff_weight * diff_loss
    
        return ModelOutput(
            loss=loss,
            lp_loss=float(lp_loss),
            rg_loss=float(rg_loss),
            con_loss=float(con_loss),
            rank_loss=float(rank_loss),
            kl_loss=float(kl_loss),
            diff_loss=float(diff_loss),
            embedding=embedding,
        )

    def calc_metrics(self, edge_scores, edge_labels):
        from utils import error_geometric_mean
        metrics = {
            "MAE": F.l1_loss(edge_scores, edge_labels).item(),
            "MAPE": mean_absolute_percentage_error(edge_labels.detach().cpu().numpy(), edge_scores.detach().cpu().numpy()),
            "RMSE": torch.sqrt(F.mse_loss(edge_scores, edge_labels)).item(),
            "R2": r2_score(edge_labels.detach().cpu().numpy(), edge_scores.detach().cpu().numpy()),
            "KL": F.kl_div(edge_scores, edge_labels).item(),
            "EGM": error_geometric_mean(edge_scores, edge_labels).item(),
        }
        return metrics
