clear
python train_GRUs.py --model MYMODEL --able-cuda --lr-scheduler --clip --clip-max-norm 0.05 --catcher-location --target-RAD \
--gpu 0 --lr 0.0001 --weight-decay 0 --max-epochs 200 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32 \