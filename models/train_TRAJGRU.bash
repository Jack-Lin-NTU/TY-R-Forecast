clear
python train_GRUs.py --model TRAJGRU --able-cuda --target-RAD --normalize-input \
--gpu 0 --lr 0.0001 --lr-scheduler --clip --clip-max-norm 1 --weight-decay 0 \
--max-epochs 100 --batch-size 3 --train-num 10 --optimizer Adam --value-dtype float32
