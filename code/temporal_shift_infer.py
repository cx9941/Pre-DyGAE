import dgl
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
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
import pandas as pd
from utils import compute_time_slice_similarity, attraction_loss, repulsion_loss
from models.timeautoencoder import TimeSeriesAutoEncoder
import torch.optim as optim
import os

if __name__ == "__main__":
    data = JDDataset(reverse=False, name=args.data_name, raw_dir=f'data/{args.data_name}', train_path=args.train_path, eval_path=args.eval_path, test_path=args.test_path)
    g = data[0]
    num_nodes = g.num_nodes()
    num_rels = data.num_rels
    entities = pd.read_csv(f'data/{args.data_name}/entities.dict', sep='\t', header=None)
    pos_num_nodes = (entities[1]<30000).sum()
    skill_num_nodes = (entities[1]>=30000).sum()
    real_id_nodes = np.unique((g.edges()[0].numpy(), g.edges()[1].numpy()))

    if args.bias == 'yes':
        entity2embedding = torch.load(
            f'data/{args.data_name}/entity2embedding.pt')
    else:
        entity2embedding = None

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

    if args.graph_inital_emb_path:
        graph_inital_emb = torch.load(args.graph_inital_emb_path)

    model = LinkPredict(num_nodes, pos_num_nodes, skill_num_nodes, num_rels, cross_attn=args.cross_attn, time=args.time, embedding=entity2embedding,
                        rg_weight=args.rg_weight, lp_weight=args.lp_weight, rank_weight=args.rank_weight, con_weight=args.con_weight, diff_weight=args.diff_weight,
                        gaussian=args.gaussian, bias=args.bias, initial_embedding=args.initial_embedding, e_dim=args.e_dim,
                        rg_loss_fn=rg_loss_fn[args.rg_loss_fn], rg_activate_fn=rg_activate_fn[args.rg_activate_fn], real_id_nodes=real_id_nodes).to(args.device)

    if args.load_state_path:
        model.load_state_dict(torch.load(args.load_state_path), strict=False)
    model.load_node_embedding(*torch.load(args.load_node_embedding_path))
    model.temporal_emb.load_emb_graph(graph_inital_emb)

    config = {
        "g": g,
        "data": data,
        "model": model,
        "device": args.device,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "eval_step": args.eval_step,
        "sample_size": args.sample_size,
        "checkpoint_path": args.checkpoint_path,
        "log_path": args.log_path,
        "results_path": args.results_path,
        "scores_path": args.scores_path,
        "k": args.k,
        "time_embedding_path": args.time_embedding_path,
        "node_embedding_path": args.node_embedding_path,
        "time": "yes"
    }
    trainer = Trainer(**config)

    ##### 调用temporal shift模块
    mu_embeddings = []
    sigma_embeddings = []
    for month in range(args.start_date, args.end_date):
        file_path = f'outputs/{args.owner_id}_{args.shift_type}_embedding/task2/{args.data_name}/train/{month}/{args.file_name}.pt'
        mu_embedding, sigma_embedding = torch.load(file_path)
        mu_embeddings.append(mu_embedding.unsqueeze(0))
        sigma_embeddings.append(sigma_embedding.unsqueeze(0))
    mu_embeddings = torch.cat(mu_embeddings, dim=0)
    sigma_embeddings = torch.cat(sigma_embeddings, dim=0)

    test_sigma_embedding = torch.mean(sigma_embeddings, dim=0)
    mu_time_embedding = mu_embeddings[-1]
    if args.start_date == 7:
        test_mu_embedding = torch.zeros(mu_embeddings[-1].shape).to(mu_embeddings.device)
    elif args.start_date == 6:
        test_mu_embedding = mu_embeddings[-1]
    elif args.strategy == 'mean':
        test_mu_embedding = torch.mean(mu_embeddings, dim=0)
    elif args.strategy == 'next':
        test_mu_embedding = mu_embeddings[-1]
    else:
        train_data = mu_embeddings
        dim = train_data.shape[-1]
        model = TimeSeriesAutoEncoder(dim).to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=args.time_lr)
        criterion = nn.MSELoss()
        best_auc = 0
        with tqdm(range(args.epochs)) as bar:
            for epoch in bar:
                if epoch % args.eval_step == 0:
                    with torch.no_grad():
                        model.eval()
                        _, test_predicted_next_step, _, _ = model(train_data)
                        trainer.model.load_node_temporal_embedding(mu_time_embedding, test_sigma_embedding)
                        metric = trainer.evaluate()
                        if metric['AUC'] > best_auc:
                            best_auc = metric['AUC']
                            test_mu_embedding  = mu_time_embedding
                        mu_time_embedding = test_predicted_next_step[-1]

                model.train()
                reconstructed, predicted_next_step, stable, trend = model(train_data)
                loss_reconstruction = criterion(reconstructed[train_data.sum(dim=-1)!=0], train_data[train_data.sum(dim=-1)!=0])
                loss_prediction = criterion(predicted_next_step[:-1], train_data[1:])
                loss_attract = attraction_loss(
                    stable, margin=1/args.temperature, temperature=args.temperature)
                loss_repel = repulsion_loss(
                    trend, margin=1/args.temperature, temperature=args.temperature)
                loss = loss_reconstruction + loss_prediction + (loss_attract + loss_repel)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_description(
                    f"Epoch [{epoch+1}/{args.epochs}], loss_reconstruction: {loss_reconstruction.item()}, loss_prediction: {loss_prediction.item()}, loss_attract: {loss_attract.item()}, loss_repel: {loss_repel.item()}")
    
    save_file_path = f'outputs/{args.owner_id}_{args.shift_type}_embedding/task2/{args.data_name}/test/{args.end_date}/{args.save_file_name}.pt'
    if not os.path.exists('/'.join(save_file_path.split('/')[:-1])):
        os.makedirs('/'.join(save_file_path.split('/')[:-1]))
    torch.save((test_mu_embedding, test_sigma_embedding), save_file_path)
