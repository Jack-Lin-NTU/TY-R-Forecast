clear
python train_my_model.py --model MYMULTIMODEL --able-cuda --lr-scheduler --clip --clip-max-norm 5 \
--gpu 1 --lr 0.0001 --weight-decay 0 --max-epochs 200 --batch-size 5 --train-num 10 --optimizer Adam --value-dtype float32 \
