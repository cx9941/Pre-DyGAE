set -o errexit

data=$1
lp_weight=$2
rg_weight=$3
diff_weight=$4
adaptive=$5
e_dim=10
task=task2
num_epochs=1000
owner_id=new
k=3
task1_seed=10

# python code/train_task1.py \
#     --mode train \
#     --data_name $data \
#     --rg_weight 100.0 \
#     --lp_weight 1.0 \
#     --rg_loss_fn tweedie \
#     --train_path data/$data/$task/old_triplet_percentage.tsv \
#     --eval_path data/$data/$task/old_triplet_percentage.tsv \
#     --test_path data/$data/$task/old_triplet_percentage.tsv \
#     --task $task \
#     --k 5 \
#     --num_epochs 1000 \
#     --owner_id $owner_id \
#     --rg_activate_fn softplus \
#     --rank_weight 0.1 \
#     --gaussian yes \
#     --bias yes \
#     --cross_attn yes \
#     --con_weight 0.05 \
#     --initial_embedding yes \
#     --seed $task1_seed \
#     --time no

python code/mf.py --data_name $data --e_dim $e_dim

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
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/epoch_1000_k_5_lr_0.0001_initalembed_yes_seed_$task1_seed/rglossfn_tweedie_activate_softplus_rgweight_100.0_lpweight_1.0_rankweight_0.1_conweight_0.05_gaussian_yes_crossattn_yes_bias_yes.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/epoch_1000_k_5_lr_0.0001_initalembed_yes_seed_$task1_seed/rglossfn_tweedie_activate_softplus_rgweight_100.0_lpweight_1.0_rankweight_0.1_conweight_0.05_gaussian_yes_crossattn_yes_bias_yes_node.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight 0.1 \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn tweedie \
    --rg_activate_fn softplus \
    --e_dim $e_dim \
    --adaptive $adaptive \
    --initial_embedding yes \
    --seed $seed

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
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/epoch_1000_k_5_lr_0.0001_initalembed_yes_seed_$task1_seed/rglossfn_tweedie_activate_softplus_rgweight_100.0_lpweight_1.0_rankweight_0.1_conweight_0.05_gaussian_yes_crossattn_yes_bias_yes.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/epoch_1000_k_5_lr_0.0001_initalembed_yes_seed_$task1_seed/rglossfn_tweedie_activate_softplus_rgweight_100.0_lpweight_1.0_rankweight_0.1_conweight_0.05_gaussian_yes_crossattn_yes_bias_yes_node.pt \
    --load_time_embedding_path outputs/${owner_id}_time_embedding/task2/$data/train/$month/epoch_${num_epochs}_k_${k}_lr_0.0001_initalembed_yes_seed_$seed/rglossfn_tweedie_activate_softplus_rgweight_${rg_weight}_lpweight_${lp_weight}_rankweight_0.1_conweight_0.0_diffweight_${diff_weight}_adaptive_${adaptive}_gaussian_yes_crossattn_yes_bias_yes_node.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight 0.1 \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn tweedie \
    --rg_activate_fn softplus \
    --e_dim $e_dim \
    --adaptive $adaptive \
    --initial_embedding yes  \
    --seed $seed
done

month=7
epochs=1000
lr=1e-2
temperature=2.0
for time_seed in 0 1 2 3 4
do
for strategy in 'mean' 'next' 'self'
do
python code/temporal_shift_infer.py \
    --data_name $data \
    --owner_id $owner_id \
    --epochs $epochs \
    --start_date 4 \
    --end_date $month \
    --lr $lr \
    --temperature $temperature \
    --dir_name epoch_${num_epochs}_k_${k}_lr_0.0001_initalembed_yes_seed_${seed} \
    --file_name rglossfn_tweedie_activate_softplus_rgweight_${rg_weight}_lpweight_${lp_weight}_rankweight_0.1_conweight_0.0_diffweight_${diff_weight}_adaptive_${adaptive}_gaussian_yes_crossattn_yes_bias_yes_node \
    --save_file_name lr_${lr}_seed_${time_seed}_epochs_${epochs}_temperature_${temperature} \
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
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/epoch_1000_k_5_lr_0.0001_initalembed_yes_seed_$task1_seed/rglossfn_tweedie_activate_softplus_rgweight_100.0_lpweight_1.0_rankweight_0.1_conweight_0.05_gaussian_yes_crossattn_yes_bias_yes.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/epoch_1000_k_5_lr_0.0001_initalembed_yes_seed_$task1_seed/rglossfn_tweedie_activate_softplus_rgweight_100.0_lpweight_1.0_rankweight_0.1_conweight_0.05_gaussian_yes_crossattn_yes_bias_yes_node.pt \
    --load_time_embedding_path outputs/${owner_id}_time_embedding/task2/$data/test/8/epoch_${num_epochs}_k_${k}_lr_0.0001_initalembed_yes_seed_${seed}/rglossfn_tweedie_activate_softplus_rgweight_${rg_weight}_lpweight_${lp_weight}_rankweight_0.1_conweight_0.0_diffweight_${diff_weight}_adaptive_${adaptive}_gaussian_yes_crossattn_yes_bias_yes_node_${strategy}/lr_${lr}_seed_${time_seed}_epochs_${epochs}_temperature_${temperature}.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight 0.1 \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn tweedie \
    --rg_activate_fn softplus \
    --e_dim $e_dim
done
done
done

