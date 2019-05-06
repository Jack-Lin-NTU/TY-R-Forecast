clear
python train_GRUs.py --model CONVGRU --able-cuda --lr-scheduler --clip --clip-max-norm 1 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 50 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32 \

# --I-x-l 118.9 --I-x-h 122.2675 --I-y-l 21.75 --I-y-h 25.1125 \
# --F-x-l 118.9 --F-x-h 122.2675 --F-y-l 21.75 --F-y-h 25.1125