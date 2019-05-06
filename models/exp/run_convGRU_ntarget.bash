clear
python train_GRUs.py --model CONVGRU  --able-cuda --lr-scheduler --clip --clip-max-norm 1 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 50 --batch-size 8 --train-num 10 --optimizer Adam --value-dtype float16 \