clear
python train_GRUs.py --model MYMODEL --multi-unit --able-cuda --target-RAD --normalize-input --catcher-location \
--gpu 2 --lr 0.0001 --lr-scheduler --clip --clip-max-norm 0.001 --weight-decay 0 \
--max-epochs 100 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32