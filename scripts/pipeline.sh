# sh scripts/train_task1.sh Dai 0 5 tweedie softplus 100.0 0.1 0.05 yes yes yes
# sh scripts/train_task2.sh Dai 1.0 100.0
# sh scripts/test_task2.sh Dai 1.0 100.0


for data in Dai Fin IT Man
do
sh scripts/test_task2.sh $data 1.0 100.0 1.0
done