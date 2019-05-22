clear
python train_my_model.py --model MYMULTIMODEL --able-cuda --lr-scheduler --clip --clip-max-norm 5 \
--gpu 0 --lr 0.001 --weight-decay 0 --max-epochs 50 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float32 \
