import argparse
import os
import random
import numpy as np
import torch
import sys
import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", default="Dai", type=str)
parser.add_argument("--task", default="task1", type=str)
parser.add_argument("--rg_loss_fn", default="tweedie", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument(
    "--train_path", default="data/Dai/task1/train/triplet_percentage.tsv", type=str)
parser.add_argument(
    "--test_path", default="data/Dai/task1/test/triplet_percentage.tsv", type=str)
parser.add_argument(
    "--eval_path", default="data/Dai/task1/eval/triplet_percentage.tsv", type=str)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--time", default="no", type=str)
parser.add_argument("--fix_emb", default="no", type=str)
parser.add_argument("--fix_model", default="yes", type=str)
parser.add_argument("--cross_attn", default="yes", type=str)
parser.add_argument("--gaussian", default="yes", type=str)
parser.add_argument("--bias", default="yes", type=str)
parser.add_argument("--owner_id", default="new", type=str)
parser.add_argument("--rg_activate_fn", default="softplus", type=str)
parser.add_argument("--num_epochs", default=10000, type=int)
parser.add_argument("--eval_step", default=50, type=int)
parser.add_argument("--e_dim", default=10, type=int)
parser.add_argument("--sample_size", default=3000, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--rg_weight", default=100.0, type=float)
parser.add_argument("--lp_weight", default=1.0, type=float)
parser.add_argument("--rank_weight", default=0.1, type=float)
parser.add_argument("--con_weight", default=0.05, type=float)
parser.add_argument("--diff_weight", default=1.0, type=float)
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--time_seed", default=-1, type=int)
parser.add_argument("--k", default=10, type=int)
parser.add_argument("--initial_embedding", default="yes", type=str)
parser.add_argument("--date", default=1, type=int)
parser.add_argument("--load_state_path", default=None, type=str)
parser.add_argument("--load_node_embedding_path", default=None, type=str)
parser.add_argument("--load_time_embedding_path", default=None, type=str)
parser.add_argument("--start_date", default=None, type=int)
parser.add_argument("--task2_strategy", default="last", type=str)
parser.add_argument("--adaptive", default="yes", type=str)
parser.add_argument("--task2_abalation", default="no", type=str)
args = parser.parse_args()

args.root_dir = f"epoch_{args.num_epochs}_k_{args.k}_lr_{args.lr}_initalembed_{args.initial_embedding}_seed_{args.seed}/"

args.identity = f"rglossfn_{args.rg_loss_fn}_activate_{args.rg_activate_fn}_rgweight_{args.rg_weight}_lpweight_{args.lp_weight}_rankweight_{args.rank_weight}_conweight_{args.con_weight}_gaussian_{args.gaussian}_crossattn_{args.cross_attn}_bias_{args.bias}"
if args.time == "yes":
    assert args.task == 'task2'
    if args.task2_abalation == 'yes':
        args.identity += "_abalation"
    else:
        args.root_dir = args.load_state_path.split('/')[-2] + '/' + args.load_state_path.split('/')[-1][:-3]
        args.identity = f"/epoch_{args.num_epochs}_k_{args.k}_lr_{args.lr}_initalembed_{args.initial_embedding}_seed_{args.seed}/rglossfn_{args.rg_loss_fn}_activate_{args.rg_activate_fn}_rgweight_{args.rg_weight}_lpweight_{args.lp_weight}_rankweight_{args.rank_weight}_conweight_{args.con_weight}_gaussian_{args.gaussian}_crossattn_{args.cross_attn}_bias_{args.bias}/diffweight_{args.diff_weight}_adaptive_{args.adaptive}"

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.checkpoint_path = f"outputs/{args.owner_id}_checkpoints/{args.task}/{args.data_name}/"
args.log_path = f"outputs/{args.owner_id}_logs/{args.task}/{args.data_name}/{args.mode}/"
args.results_path = f"outputs/{args.owner_id}_results/{args.task}/{args.data_name}/{args.mode}/"
args.scores_path = f"outputs/{args.owner_id}_scores/{args.task}/{args.data_name}/{args.mode}/"
args.node_embedding_path = f"outputs/{args.owner_id}_node_embedding/{args.task}/{args.data_name}/{args.mode}/"

if args.time == 'yes':
    args.log_path += f"{args.date}/"
    args.results_path += f"{args.date}/"
    args.scores_path += f"{args.date}/"
    args.checkpoint_path += f"{args.date}/"
    args.node_embedding_path += f"{args.date}/"

    args.old_triplet_path = f'data/{args.data_name}/task2/old_triplet_percentage.tsv'

    if args.mode == 'test':
        if len(args.load_time_embedding_path.split('/')) > 8:
            time_identify = args.load_time_embedding_path.split('/')[-1][:-3]
            args.identity += f"_{args.task2_strategy}_{time_identify}"

    args.time_embedding_path = f"outputs/{args.owner_id}_time_embedding/{args.task}/{args.data_name}/{args.mode}/{args.date}/"
    args.time_embedding_path += args.root_dir
    args.time_embedding_path += args.identity + "_node.pt"
    args.graph_inital_emb_path = f"data/{args.data_name}/task2/graph_initial_emb_dim_{args.e_dim}.pt"
    if not os.path.exists('/'.join(args.time_embedding_path.split('/')[:-1])):
        os.makedirs('/'.join(args.time_embedding_path.split('/')[:-1]))


args.checkpoint_path += args.root_dir
args.log_path += args.root_dir
args.results_path += args.root_dir
args.scores_path += args.root_dir
args.node_embedding_path += args.root_dir


args.checkpoint_path += args.identity + ".pt"

if args.time_seed != -1:
    assert args.time == 'yes' and args.task == 'task2'
    args.log_path += args.identity + f"_time_seed_{args.time_seed}.log"
else:
    args.log_path += args.identity + ".log"

args.results_path += args.identity + ".pt"
args.scores_path += args.identity + ".pt"
args.node_embedding_path += args.identity + "_node.pt"


if not os.path.exists('/'.join(args.checkpoint_path.split('/')[:-1])):
    os.makedirs('/'.join(args.checkpoint_path.split('/')[:-1]))

if not os.path.exists('/'.join(args.log_path.split('/')[:-1])):
    os.makedirs('/'.join(args.log_path.split('/')[:-1]))

if not os.path.exists('/'.join(args.results_path.split('/')[:-1])):
    os.makedirs('/'.join(args.results_path.split('/')[:-1]))

if not os.path.exists('/'.join(args.scores_path.split('/')[:-1])):
    os.makedirs('/'.join(args.scores_path.split('/')[:-1]))

if not os.path.exists('/'.join(args.node_embedding_path.split('/')[:-1])):
    os.makedirs('/'.join(args.node_embedding_path.split('/')[:-1]))
    

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(
        sys.stdout), logging.FileHandler(args.log_path)],
    level=logging.INFO
)

for arg_name, arg_value in vars(args).items():
    logger.info(f"{arg_name}: {arg_value}")
