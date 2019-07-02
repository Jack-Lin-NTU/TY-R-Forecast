clear
python train_GRUs.py --model TRAJGRU --able-cuda --target-RAD --denoise-RAD \
--gpu 0 --lr 0.00005 --clip --clip-max-norm 1 --weight-decay 0.04 \
--max-epochs 30 --batch-size 4 --train-num 11 --optimizer Adam --value-dtype float32
