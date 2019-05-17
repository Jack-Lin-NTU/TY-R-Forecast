clear
python train_my_model.py --model MYMULTIMODEL --able-cuda --lr-scheduler --clip --clip-max-norm 2 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.01 --weight-decay 0 --max-epochs 50 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32 \
