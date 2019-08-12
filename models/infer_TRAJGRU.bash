clear
python infer_GRUs.py --model TRAJGRU --able-cuda --target-RAD \
--gpu 0 --lr 0.00005 --clip --clip-max-norm 1 --weight-decay 0.001 --lr-scheduler \
--max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float32
