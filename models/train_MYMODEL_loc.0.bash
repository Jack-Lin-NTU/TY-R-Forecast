clear
python train_GRUs.py --model MYMODEL --able-cuda --lr-scheduler --clip --clip-max-norm 0.01 --catcher-location \
--gpu 0 --lr 0.0001 --weight-decay 0 --max-epochs 100 --batch-size 5 --train-num 10 --optimizer Adam --value-dtype float32 \
