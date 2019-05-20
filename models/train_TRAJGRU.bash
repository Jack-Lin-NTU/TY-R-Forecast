clear
python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --clip --clip-max-norm 5 --input-with-grid \
--gpu 0 --lr 0.001 --weight-decay 0 --max-epochs 50 --batch-size 3 --train-num 10 --optimizer Adam --value-dtype float32 \