set -o errexit

data=$1
task=task1
lp_weight=1.0
num_epochs=1000
owner_id=new
time="no"

for seed in 9 10
do
for k in $3
do
for rg_loss_fn in $4
do
for rg_activate_fn in $5
do
for rg_weight in $6
do
for rank_weight in $7
do
for con_weight in $8
do
for gaussian in $9
do
for cross_attn in ${10}
do
for bias in ${11}
do
python code/train_task1.py \
    --mode train \
    --data_name $data \
    --rg_weight $rg_weight \
    --lp_weight $lp_weight \
    --rg_loss_fn $rg_loss_fn \
    --train_path data/$data/task1/train/triplet_percentage.tsv \
    --eval_path data/$data/task1/eval/triplet_percentage.tsv \
    --test_path data/$data/task1/test/triplet_percentage.tsv \
    --task $task \
    --k $k \
    --num_epochs $num_epochs \
    --owner_id $owner_id \
    --rg_activate_fn $rg_activate_fn \
    --rank_weight $rank_weight \
    --gaussian $gaussian \
    --bias $bias \
    --cross_attn $cross_attn \
    --con_weight $con_weight \
    --initial_embedding yes \
    --seed $seed \
    --time $time

python code/test_task1.py \
    --mode test \
    --data_name $data \
    --rg_weight $rg_weight \
    --lp_weight $lp_weight \
    --rg_loss_fn $rg_loss_fn \
    --train_path data/$data/task1/train/triplet_percentage.tsv \
    --test_path data/$data/task1/eval/filter_triplet_percentage.tsv \
    --eval_path data/$data/task1/test/filter_triplet_percentage.tsv \
    --task $task \
    --k $k \
    --num_epochs $num_epochs \
    --owner_id $owner_id \
    --rg_activate_fn $rg_activate_fn \
    --rank_weight $rank_weight \
    --gaussian $gaussian \
    --bias $bias \
    --cross_attn $cross_attn \
    --con_weight $con_weight \
    --initial_embedding yes \
    --seed $seed \
    --time $time

done
done
done
done
done
done
done
done
done
done