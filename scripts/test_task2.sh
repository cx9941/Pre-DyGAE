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

lp_weight=1.0
rg_weight=100.0
rank_weight=0.1
diff_weight=$6
adaptive=$7
task2_lr=0.0001
num_epochs=1000
k=3

task1_identity=epoch_${task1_num_epochs}_k_${task1_k}_lr_${task1_lr}_initalembed_yes_seed_$task1_seed/rglossfn_${task1_rg_loss_fn}_activate_softplus_rgweight_${task1_rg_weight}_lpweight_${task1_lp_weight}_rankweight_${task1_rank_weight}_conweight_${task1_con_weight}_gaussian_yes_crossattn_${task1_bias}_bias_${task1_bias}

for seed in 0 1 2 3 4 
do

month=7
epochs=1000
time_lr=0.01
temperature=2.0
for time_seed in 0 1 2 3 4
do
for strategy in 'self' 'mean' 'next'
do

task2_identity=epoch_${num_epochs}_k_${k}_lr_${task2_lr}_initalembed_yes_seed_$seed/rglossfn_tweedie_activate_softplus_rgweight_${rg_weight}_lpweight_${lp_weight}_rankweight_${rank_weight}_conweight_0.0_gaussian_yes_crossattn_yes_bias_yes/diffweight_${diff_weight}_adaptive_${adaptive}

python code/temporal_shift_infer.py \
    --data_name $data \
    --owner_id $owner_id \
    --epochs $epochs \
    --start_date 4 \
    --end_date $month \
    --lr $time_lr \
    --temperature $temperature \
    --file_name ${task1_identity}/${task2_identity}_node \
    --save_file_name ${task1_identity}/${task2_identity}/lr_${time_lr}_seed_${time_seed}_epochs_${epochs}_temperature_${temperature}_${strategy}_node \
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
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/${task1_identity}.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/${task1_identity}_node.pt \
    --load_time_embedding_path outputs/${owner_id}_time_embedding/task2/$data/test/$month/${task1_identity}/${task2_identity}/lr_${time_lr}_seed_${time_seed}_epochs_${epochs}_temperature_${temperature}_${strategy}_node.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight $rank_weight \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn tweedie \
    --rg_activate_fn softplus \
    --e_dim $e_dim \
    --time_seed $time_seed \
    --task2_strategy $strategy \
    --adaptive $adaptive \
    --initial_embedding yes  \
    --seed $seed \
    --lr $task2_lr
done
done


done


