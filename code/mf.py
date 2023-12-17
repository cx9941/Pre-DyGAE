import pandas as pd
import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='Dai')
parser.add_argument('--e_dim', type=int, default=10)
args = parser.parse_args()

triplet_percentage = pd.read_csv(f'data/{args.data_name}/task2/old_triplet_percentage.tsv', header=None, sep='\t')
entities = pd.read_csv(f'data/{args.data_name}/entities.dict', header=None, sep='\t')
user_num = (entities[1]<30000).sum()
item_num = (entities[1]>=30000).sum()
matrix = torch.zeros(user_num, item_num)
matrix[triplet_percentage[0] - 20000][triplet_percentage[0] - 30000] = 1

import torch
import torch.optim as optim

n, m = matrix.shape
k = 10  # 设定潜在特征的数量

# 初始化P和Q
P = torch.rand(n, k, requires_grad=True)
Q = torch.rand(k, m, requires_grad=True)

# 选择一个优化器
optimizer = optim.Adam([P, Q], lr=0.01)

# 迭代次数
num_iterations = 10000

# 训练循环
for i in range(num_iterations):
    optimizer.zero_grad()
    loss = ((torch.matmul(P, Q) - matrix) ** 2).mean()  # 计算MSE
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if i % 100 == 0:
        print(f"Iteration {i}: Loss {loss.item()}")

torch.save(torch.cat([P, Q.T], dim=0), f'data/{args.data_name}/task2/graph_initial_emb_dim_{args.e_dim}.pt')