set -o errexit

data=$1
task=task2
e_dim=10
owner_id=new

task1_seed=10
task1_rg_weight=100.0
task1_lp_weight=1.0
task1_rank_weight=$2
task1_con_weight=$3
task1_rg_loss_fn=$4
task1_bias=$5
task1_k=5
task1_num_epochs=1000
task1_lr=0.0001

# python code/train_task1.py \
#     --mode train \
#     --data_name $data \
#     --rg_weight $task1_rg_weight \
#     --lp_weight $task1_lp_weight \
#     --rg_loss_fn $task1_rg_loss_fn \
#     --train_path data/$data/$task/old_triplet_percentage.tsv \
#     --eval_path data/$data/$task/old_triplet_percentage.tsv \
#     --test_path data/$data/$task/old_triplet_percentage.tsv \
#     --task $task \
#     --k $task1_k \
#     --num_epochs $task1_num_epochs \
#     --owner_id $owner_id \
#     --rg_activate_fn softplus \
#     --rank_weight $task1_rank_weight \
#     --gaussian yes \
#     --bias $task1_bias \
#     --cross_attn $task1_bias \
#     --con_weight $task1_con_weight \
#     --initial_embedding yes \
#     --seed $task1_seed \
#     --lr $task1_lr \
#     --time no

# python code/mf.py --data_name $data --e_dim $e_dim

lp_weight=1.0
rg_weight=100.0
rank_weight=0.1
diff_weight=$6
adaptive=$7
task2_lr=0.0001
num_epochs=1000
k=3

for seed in 0 1 2 3 4 
do
for month in 4 5 6
do
python code/train_task2.py \
    --data_name $data \
    --k $k \
    --mode train \
    --task task2 \
    --time yes \
    --num_epochs $num_epochs \
    --train_path data/$data/task2/$month/diff_triplet_percentage.tsv \
    --eval_path data/$data/task2/$month/diff_triplet_percentage.tsv \
    --test_path data/$data/task2/$month/diff_triplet_percentage.tsv \
    --date $month \
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/epoch_${task1_num_epochs}_k_${task1_k}_lr_${task1_lr}_initalembed_yes_seed_$task1_seed/rglossfn_${task1_rg_loss_fn}_activate_softplus_rgweight_${task1_rg_weight}_lpweight_${task1_lp_weight}_rankweight_${task1_rank_weight}_conweight_${task1_con_weight}_gaussian_yes_crossattn_${task1_bias}_bias_${task1_bias}.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/epoch_${task1_num_epochs}_k_${task1_k}_lr_${task1_lr}_initalembed_yes_seed_$task1_seed/rglossfn_${task1_rg_loss_fn}_activate_softplus_rgweight_${task1_rg_weight}_lpweight_${task1_lp_weight}_rankweight_${task1_rank_weight}_conweight_${task1_con_weight}_gaussian_yes_crossattn_${task1_bias}_bias_${task1_bias}_node.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight $rank_weight \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn tweedie \
    --rg_activate_fn softplus \
    --e_dim $e_dim \
    --adaptive $adaptive \
    --initial_embedding yes \
    --seed $seed \
    --lr $task2_lr

    python code/test_task2.py \
    --data_name $data \
    --k $k \
    --mode test \
    --task task2 \
    --time yes \
    --num_epochs $num_epochs \
    --train_path data/$data/task2/$month/diff_triplet_percentage.tsv \
    --eval_path data/$data/task2/$month/diff_triplet_percentage.tsv \
    --test_path data/$data/task2/$month/diff_triplet_percentage.tsv \
    --date $month \
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/epoch_${task1_num_epochs}_k_${task1_k}_lr_${task1_lr}_initalembed_yes_seed_$task1_seed/rglossfn_${task1_rg_loss_fn}_activate_softplus_rgweight_${task1_rg_weight}_lpweight_${task1_lp_weight}_rankweight_${task1_rank_weight}_conweight_${task1_con_weight}_gaussian_yes_crossattn_${task1_bias}_bias_${task1_bias}.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/epoch_${task1_num_epochs}_k_${task1_k}_lr_${task1_lr}_initalembed_yes_seed_$task1_seed/rglossfn_${task1_rg_loss_fn}_activate_softplus_rgweight_${task1_rg_weight}_lpweight_${task1_lp_weight}_rankweight_${task1_rank_weight}_conweight_${task1_con_weight}_gaussian_yes_crossattn_${task1_bias}_bias_${task1_bias}_node.pt \
    --load_time_embedding_path outputs/${owner_id}_time_embedding/task2/$data/train/$month/epoch_${num_epochs}_k_${k}_lr_${task2_lr}_initalembed_yes_seed_$seed/rglossfn_tweedie_activate_softplus_rgweight_${rg_weight}_lpweight_${lp_weight}_rankweight_${rank_weight}_conweight_0.0_diffweight_${diff_weight}_adaptive_${adaptive}_gaussian_yes_crossattn_yes_bias_yes_node.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight $rank_weight \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn tweedie \
    --rg_activate_fn softplus \
    --e_dim $e_dim \
    --adaptive $adaptive \
    --initial_embedding yes  \
    --seed $seed \
    --lr $task2_lr
done

month=7
epochs=1000
time_lr=1e-2
temperature=2.0
for time_seed in 0 1 2 3 4
do
for strategy in 'self' 'mean' 'next'
do
python code/temporal_shift_infer.py \
    --data_name $data \
    --owner_id $owner_id \
    --epochs $epochs \
    --start_date 4 \
    --end_date $month \
    --lr $time_lr \
    --temperature $temperature \
    --dir_name epoch_${num_epochs}_k_${k}_lr_${task2_lr}_initalembed_yes_seed_${seed} \
    --file_name rglossfn_tweedie_activate_softplus_rgweight_${rg_weight}_lpweight_${lp_weight}_rankweight_${rank_weight}_conweight_0.0_diffweight_${diff_weight}_adaptive_${adaptive}_gaussian_yes_crossattn_yes_bias_yes_node \
    --save_file_name lr_${time_lr}_seed_${time_seed}_epochs_${epochs}_temperature_${temperature} \
    --strategy $strategy \
    --time_seed $time_seed 

python code/test_task2.py \
    --data_name $data \
    --k $k \
    --mode test \
    --task task2 \
    --time yes \
    --num_epochs $num_epochs \
    --train_path data/$data/task2/$month/triplet_percentage.tsv \
    --eval_path data/$data/task2/$month/triplet_percentage.tsv \
    --test_path data/$data/task2/$month/triplet_percentage.tsv \
    --date $month \
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/epoch_${task1_num_epochs}_k_${task1_k}_lr_${task1_lr}_initalembed_yes_seed_$task1_seed/rglossfn_${task1_rg_loss_fn}_activate_softplus_rgweight_${task1_rg_weight}_lpweight_${task1_lp_weight}_rankweight_${task1_rank_weight}_conweight_${task1_con_weight}_gaussian_yes_crossattn_${task1_bias}_bias_${task1_bias}.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/epoch_${task1_num_epochs}_k_${task1_k}_lr_${task1_lr}_initalembed_yes_seed_$task1_seed/rglossfn_${task1_rg_loss_fn}_activate_softplus_rgweight_${task1_rg_weight}_lpweight_${task1_lp_weight}_rankweight_${task1_rank_weight}_conweight_${task1_con_weight}_gaussian_yes_crossattn_${task1_bias}_bias_${task1_bias}_node.pt \
    --load_time_embedding_path outputs/${owner_id}_time_embedding/task2/$data/test/$month/epoch_${num_epochs}_k_${k}_lr_${task2_lr}_initalembed_yes_seed_${seed}/rglossfn_tweedie_activate_softplus_rgweight_${rg_weight}_lpweight_${lp_weight}_rankweight_${rank_weight}_conweight_0.0_diffweight_${diff_weight}_adaptive_${adaptive}_gaussian_yes_crossattn_yes_bias_yes_node_${strategy}/lr_${time_lr}_seed_${time_seed}_epochs_${epochs}_temperature_${temperature}.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight $rank_weight \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn tweedie \
    --rg_activate_fn softplus \
    --e_dim $e_dim
done
done
done

