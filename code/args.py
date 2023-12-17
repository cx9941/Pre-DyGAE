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
parser.add_argument("--train_path", default="data/Dai/task1/train/triplet_percentage.tsv", type=str)
parser.add_argument("--test_path", default="data/Dai/task1/test/triplet_percentage.tsv", type=str)
parser.add_argument("--eval_path", default="data/Dai/task1/eval/triplet_percentage.tsv", type=str)
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
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--rg_weight", default=100.0, type=float)
parser.add_argument("--lp_weight", default=1.0, type=float)
parser.add_argument("--rank_weight", default=0.1, type=float)
parser.add_argument("--con_weight", default=0.05, type=float)
parser.add_argument("--diff_weight", default=1.0, type=float)
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--k", default=10, type=int)
parser.add_argument("--initial_embedding", default="yes", type=str)
parser.add_argument("--date", default=1, type=int)
parser.add_argument("--load_state_path", default=None, type=str)
parser.add_argument("--load_node_embedding_path", default=None, type=str)
parser.add_argument("--load_time_embedding_path", default=None, type=str)
parser.add_argument("--start_date", default=None, type=int)
parser.add_argument("--task2_strategy", default="last", type=str)
parser.add_argument("--adaptive", default="yes", type=str)
args = parser.parse_args()

args.root_dir = f"epoch_{args.num_epochs}_k_{args.k}_lr_{args.lr}_initalembed_{args.initial_embedding}_seed_{args.seed}/"

args.identity = f"rglossfn_{args.rg_loss_fn}_activate_{args.rg_activate_fn}_rgweight_{args.rg_weight}_lpweight_{args.lp_weight}_rankweight_{args.rank_weight}_conweight_{args.con_weight}_gaussian_{args.gaussian}_crossattn_{args.cross_attn}_bias_{args.bias}"
if args.time == "yes":
    args.identity = f"rglossfn_{args.rg_loss_fn}_activate_{args.rg_activate_fn}_rgweight_{args.rg_weight}_lpweight_{args.lp_weight}_rankweight_{args.rank_weight}_conweight_{args.con_weight}_diffweight_{args.diff_weight}_adaptive_{args.adaptive}_gaussian_{args.gaussian}_crossattn_{args.cross_attn}_bias_{args.bias}"
    # args.identity += "_time_shift"
    assert args.task == 'task2'


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
    if not os.path.exists(args.time_embedding_path):
        os.makedirs(args.time_embedding_path)
    args.time_embedding_path += args.identity + "_node.pt"
    args.graph_inital_emb_path = f"data/{args.data_name}/task2/graph_initial_emb_dim_{args.e_dim}.pt"
        


args.checkpoint_path += args.root_dir
args.log_path += args.root_dir
args.results_path += args.root_dir
args.scores_path += args.root_dir
args.node_embedding_path += args.root_dir




if not os.path.exists(args.checkpoint_path):
    os.makedirs(args.checkpoint_path)
if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)
if not os.path.exists(args.scores_path):
    os.makedirs(args.scores_path)
if not os.path.exists(args.node_embedding_path):
    os.makedirs(args.node_embedding_path)

    

args.checkpoint_path += args.identity + ".pt"
args.log_path += args.identity + ".log"
args.results_path += args.identity + ".pt"
args.scores_path += args.identity + ".pt"
args.node_embedding_path += args.identity + "_node.pt"



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