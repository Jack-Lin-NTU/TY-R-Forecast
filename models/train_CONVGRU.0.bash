clear
python train_GRUs.py --model CONVGRU --able-cuda --lr-scheduler --clip --clip-max-norm 0.01 --seed 1 \
--gpu 0 --lr 0.00005 --weight-decay 0 --max-epochs 200 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32
