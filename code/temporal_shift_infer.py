# %%
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch
import argparse
import os
import numpy as np
import random

class TimeSeriesAutoEncoder(nn.Module):
    def __init__(self, dim):
        super(TimeSeriesAutoEncoder, self).__init__()
        self.stable_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.trend_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.next_step_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        stable = self.stable_mlp(x)
        trend = self.trend_mlp(x)
        reconstructed = stable + trend  # 重构原始embedding
        next_step = self.next_step_mlp(trend) + reconstructed  # 预测下一个时间片
        return reconstructed, next_step, stable, trend

def compute_time_slice_similarity(embeddings):
    T, num, dim = embeddings.shape
    embeddings_flat = embeddings.reshape(T, -1)  # 将每个时间片的embeddings展平
    sim = F.cosine_similarity(
        embeddings_flat[:, None, :], embeddings_flat[None, :, :], dim=2)
    return sim

def attraction_loss(embeddings, margin=2.0, temperature=2):
    sim = compute_time_slice_similarity(embeddings) / temperature
    mask = ~torch.eye(sim.size(0), dtype=torch.bool)  # 排除自相似
    return (margin - sim[mask]).mean()

def repulsion_loss(embeddings, margin=2.0, temperature=2):
    sim = compute_time_slice_similarity(embeddings) / temperature
    mask = ~torch.eye(sim.size(0), dtype=torch.bool)  # 排除自距离
    return (sim[mask] + margin).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='Man')
    parser.add_argument("--owner_id", type=str, default='new')
    parser.add_argument("--strategy", type=str, default='avg')
    parser.add_argument("--shift_type", type=str, default='time')
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--start_date", type=int, default=4)
    parser.add_argument("--end_date", type=int, default=8)
    parser.add_argument("--time_seed", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=10)
    parser.add_argument("--file_name", type=str, default='rglossfn_tweedie_activate_softplus_rgweight_100.0_lpweight_1.0_rankweight_0.0_conweight_0.0_diffweight_1.0_gaussian_yes_crossattn_yes_bias_yes_node')
    parser.add_argument("--save_file_name", type=str, default='rglossfn_tweedie_activate_softplus_rgweight_100.0_lpweight_1.0_rankweight_0.0_conweight_0.0_diffweight_1.0_gaussian_yes_crossattn_yes_bias_yes_node')
    time_args = parser.parse_args()
    print(time_args)

    random.seed(time_args.time_seed)
    os.environ['PYTHONHASHSEED'] = str(time_args.time_seed)
    np.random.seed(time_args.time_seed)
    torch.manual_seed(time_args.time_seed)
    torch.cuda.manual_seed(time_args.time_seed)

    mu_embeddings = []
    sigma_embeddings = []
    for month in range(time_args.start_date, time_args.end_date):
        file_path = f'outputs/{time_args.owner_id}_{time_args.shift_type}_embedding/task2/{time_args.data_name}/train/{month}/{time_args.file_name}.pt'
        mu_embedding, sigma_embedding = torch.load(file_path)
        mu_embeddings.append(mu_embedding.unsqueeze(0))
        sigma_embeddings.append(sigma_embedding.unsqueeze(0))
    mu_embeddings = torch.cat(mu_embeddings, dim=0)
    sigma_embeddings = torch.cat(sigma_embeddings, dim=0)


    if time_args.strategy == 'mean':
        test_mu_embedding = torch.mean(mu_embeddings, dim=0)
    elif time_args.strategy == 'next':
        test_mu_embedding = mu_embeddings[-1]
    else:
        train_data = mu_embeddings
        dim = train_data.shape[-1]
        model = TimeSeriesAutoEncoder(dim).to('cuda')
        optimizer = optim.Adam(model.parameters(), lr=time_args.lr)
        criterion = nn.MSELoss()
        with tqdm(range(time_args.epochs)) as bar:
            for epoch in bar:
                reconstructed, predicted_next_step, stable, trend = model(train_data)
                loss_reconstruction = criterion(reconstructed[train_data.sum(dim=-1)!=0], train_data[train_data.sum(dim=-1)!=0])
                loss_prediction = criterion(predicted_next_step[:-1], train_data[1:])
                loss_attract = attraction_loss(
                    stable, margin=1/time_args.temperature, temperature=time_args.temperature)
                loss_repel = repulsion_loss(
                    trend, margin=1/time_args.temperature, temperature=time_args.temperature)
                loss = loss_reconstruction + loss_prediction + loss_attract + loss_repel
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_description(
                    f"Epoch [{epoch+1}/{time_args.epochs}], loss_reconstruction: {loss_reconstruction.item()}, loss_prediction: {loss_prediction.item()}, loss_attract: {loss_attract.item()}, loss_repel: {loss_repel.item()}")
        _, test_predicted_next_step, _, _ = model(train_data)
        test_mu_embedding = test_predicted_next_step[-1]
    test_sigma_embedding = torch.mean(sigma_embeddings, dim=0)
    save_file_path = f'outputs/{time_args.owner_id}_{time_args.shift_type}_embedding/task2/{time_args.data_name}/test/{time_args.end_date}/{time_args.save_file_name}.pt'
    if not os.path.exists('/'.join(save_file_path.split('/')[:-1])):
        os.makedirs('/'.join(save_file_path.split('/')[:-1]))
    torch.save((test_mu_embedding, test_sigma_embedding), save_file_path)