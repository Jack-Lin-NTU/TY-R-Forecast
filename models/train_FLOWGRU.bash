clear
python train_GRUs.py --model FLOWGRU --able-cuda --target-RAD --denoise-RAD \
--gpu 0 --lr 0.003 --clip --clip-max-norm 1 --weight-decay 0.01 --lr-scheduler \
--max-epochs 10 --batch-size 6 --train-num 11 --optimizer Adam --value-dtype float32