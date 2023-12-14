import dgl
import numpy as np
import torch
import torch.nn as nn
import tqdm
from dgl.nn.pytorch import RelGraphConv
from mydataset import JDDataset
from args import args
from utils import get_subset_g
from model import LinkPredict
from trainer import Trainer
from utils import calc_mrr
import logging
import torch.nn.functional as F
from torchmetrics.regression import TweedieDevianceScore
from sklearn.metrics import roc_auc_score


def compute_hits(predictions, labels, k):
    hits = 0
    for i in range(predictions.size(0)):
        _, sorted_indices = torch.sort(predictions[i], descending=True)
        hit_count = torch.sum(torch.gather(
            labels[i], 0, sorted_indices[:k]) == 1).item()
        hits += int(hit_count > 0)
    return hits / predictions.size(0)


def compute_mrr(predictions, labels):
    mrr_sum = 0.0

    for i in range(predictions.size(0)):
        _, sorted_indices = torch.sort(predictions[i], descending=True)
        rank = (sorted_indices == labels[i].nonzero()).nonzero()
        if len(rank) > 0:
            mrr_sum += 1.0 / (rank[0, 0].item() + 1)

    return mrr_sum / predictions.size(0)


def compute_auc(predictions, labels):
    # Flatten the predictions and labels matrices
    predictions_flat = predictions.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)

    # Calculate AUC using sklearn's roc_auc_score function
    auc = roc_auc_score(labels_flat.cpu().numpy(),
                        predictions_flat.cpu().numpy())

    return auc


if __name__ == "__main__":
    data = JDDataset(reverse=False, name=args.data_name,
                     raw_dir=f'../data/{args.data_name}/{args.task}', train_path=args.train_path, eval_path=args.eval_path, test_path=args.test_path)
    g = data[0]
    num_nodes = g.num_nodes()
    num_rels = data.num_rels
    pos_num_nodes = (g.edges()[0].max() + 1).item()
    skill_num_nodes = (g.edges()[1].max() - g.edges()[0].max()).item()

    if args.bias == 'yes':
        entity2embedding = torch.load(
            f'../data/{args.data_name}/{args.task}/entity2embedding.pt')
    else:
        entity2embedding = None

    if args.time == "yes":
        time_embedding = torch.load(
            f'../data/{args.data_name}/{args.task}/time_embedding.pt')
    else:
        time_embedding = None

    rg_loss_fn = {
        "l1": F.l1_loss,
        "mse": F.mse_loss,
        "tweedie": TweedieDevianceScore(1.5)
    }

    rg_activate_fn = {
        "elu": nn.ELU(),
        "softplus": nn.Softplus(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "leakyrelu": nn.LeakyReLU()
    }

    model = LinkPredict(num_nodes, pos_num_nodes, skill_num_nodes, num_rels, cross_attn=args.cross_attn, embedding=entity2embedding, time=args.time,
                        rg_weight=args.rg_weight, lp_weight=args.lp_weight, rank_weight=args.rank_weight, con_weight=args.con_weight,
                        gaussian=args.gaussian, bias=args.bias, initial_embedding=args.initial_embedding,
                        rg_loss_fn=rg_loss_fn[args.rg_loss_fn], rg_activate_fn=rg_activate_fn[args.rg_activate_fn]).to(args.device)
    for data in ['Dai']:
        pred = torch.load(
            f'/code/chenxi02/AAAI/data/{data}/dates/trans/output.pt').squeeze(0)
        label = torch.load(
            f'/code/chenxi02/AAAI/data/{data}/dates/7/matrix.pt')
        pred = pred.to(label.device)
        mae = model.calc_metrics(pred[label != 0], label[label != 0])

        label[label != 0] = 1
        mae['mrr'] = compute_mrr(pred, label) / 10
        mae['Hits1'] = compute_hits(pred, label, 1) / 10
        mae['Hits3'] = compute_hits(pred, label, 3) / 10
        mae['AUC'] = compute_auc(pred, label) / 10
        mae['MAPE'] = mae['MAPE'] / 10
        mae['Tweedie'] = TweedieDevianceScore(pred, label)
        print('STJGCN&' + '&'.join(str(round(mae[i] * 1000, 2)) for i in [
              'mrr', 'Hits1', 'Hits3', 'AUC', 'MAE', 'RMSE', 'MAPE', 'EGM']) + '\\\\')
