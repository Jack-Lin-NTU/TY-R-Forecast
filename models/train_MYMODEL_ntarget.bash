clear
python train_my_model.py --model MYMODEL --able-cuda --lr-scheduler --clip --clip-max-norm 2 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.05 --weight-decay 0 --max-epochs 50 --batch-size 12 --train-num 10 --optimizer Adam --value-dtype float32 \
--I-x-l 118 --I-x-h 123.5 --I-y-l 20 --I-y-h 27
