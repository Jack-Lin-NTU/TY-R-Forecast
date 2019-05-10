clear
python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --clip --clip-max-norm 2 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 30 --batch-size 2 --train-num 10 --optimizer Adam --value-dtype float32 \