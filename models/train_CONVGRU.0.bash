clear
python train_GRUs.py --model CONVGRU --able-cuda --lr-scheduler --clip --clip-max-norm 5 --input-with-grid \
--gpu 0 --lr 0.0001 --weight-decay 0 --max-epochs 100 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32
