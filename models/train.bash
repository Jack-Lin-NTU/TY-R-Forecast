
clear
python train_GRUs.py --model CONVGRU --able-cuda --target-RAD --denoise-RAD \
--gpu 0 --lr 0.00005 --clip --clip-max-norm 1 --weight-decay 0.1 --lr-scheduler \
--max-epochs 10 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32

clear
python train_GRUs.py --model TRAJGRU --able-cuda --target-RAD --denoise-RAD \
--gpu 0 --lr 0.00005 --clip --clip-max-norm 1 --weight-decay 0.1 --lr-scheduler \
--max-epochs 10 --batch-size 4 --train-num 11 --optimizer Adam --value-dtype float32

clear
python train_GRUs.py --model TRAJGRU_TEST --able-cuda --target-RAD --denoise-RAD \
--gpu 0 --lr 0.00005 --clip --clip-max-norm 1 --weight-decay 0.1 --lr-scheduler \
--max-epochs 10 --batch-size 6 --train-num 11 --optimizer Adam --value-dtype float32
