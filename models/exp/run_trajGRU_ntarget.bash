clear
python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --clip --clip-max-norm 0.005 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 50 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \