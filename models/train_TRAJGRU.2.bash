clear
python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --input-with-grid \
--gpu 2 --lr 0.00008 --weight-decay 0 --max-epochs 100 --batch-size 3 --train-num 10 --optimizer Adam --value-dtype float32 \
