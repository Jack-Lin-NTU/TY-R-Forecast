clear
python train_GRUs.py --model TRAJGRU --able-cuda --target-RAD --normalize-input \
--gpu 1 --lr 0.0001 --lr-scheduler --clip --clip-max-norm 0.001 --weight-decay 0 \
--max-epochs 100 --batch-size 3 --train-num 10 --optimizer Adam --value-dtype float32
