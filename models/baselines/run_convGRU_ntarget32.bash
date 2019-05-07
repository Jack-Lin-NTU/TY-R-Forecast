clear
python train_GRUs.py --model CONVGRU --able-cuda --lr-scheduler --clip --clip-max-norm 2 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 50 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float32 \
