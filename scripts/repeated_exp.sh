set -o errexit
for data_name in $1
do
for epoch in 1000
do
for seed in 0 1 2 3 4 5 6 7 8 9
do
for temperature in 2.0 
do
for lr in 0.005
do
sh scripts/test_task2.sh $data_name $epoch $seed $lr $temperature
done
done
done
done
done