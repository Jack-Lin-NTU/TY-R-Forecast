clear
python detail_MYMODEL.py --model MYMODEL --able-cuda --lr-scheduler --clip --clip-max-norm 5 --target-RAD \
--gpu 0 --lr 0.0001 --weight-decay 0 --max-epochs 200 --batch-size 1 --train-num 10 --optimizer Adam --value-dtype float32 \