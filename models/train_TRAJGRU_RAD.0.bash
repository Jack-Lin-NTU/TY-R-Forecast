clear
python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --clip --clip-max-norm 0.01 --target-RAD --denoise-RAD \
--gpu 0 --lr 0.00005 --weight-decay 0 --max-epochs 100 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float32 \
