clear
python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --input-with-grid \
--gpu 0 --lr 0.0003 --weight-decay 0 --max-epochs 200 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float32 \
