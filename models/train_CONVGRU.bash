# clear
# python train_GRUs.py --model CONVGRU --able-cuda --target-RAD --normalize-input \
# --gpu 0 --lr 0.0001 --clip --clip-max-norm 1 --weight-decay 0.1 \
# --max-epochs 10 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32

# clear
# python train_GRUs.py --model CONVGRU --able-cuda --target-RAD --normalize-input \
# --gpu 0 --lr 0.0001 --clip --clip-max-norm 1 --weight-decay 0.01 \
# --max-epochs 10 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32

# clear
# python train_GRUs.py --model CONVGRU --able-cuda --target-RAD --normalize-input \
# --gpu 0 --lr 0.0001 --clip --clip-max-norm 1 --weight-decay 0.001 \
# --max-epochs 10 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32

clear
python train_GRUs.py --model CONVGRU --able-cuda --target-RAD --normalize-input \
--gpu 0 --lr 0.0001 --clip --clip-max-norm 1 --weight-decay 0.02 \
--max-epochs 10 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32

clear
python train_GRUs.py --model CONVGRU --able-cuda --target-RAD --normalize-input \
--gpu 0 --lr 0.0001 --clip --clip-max-norm 1 --weight-decay 0.04 \
--max-epochs 10 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32

clear
python train_GRUs.py --model CONVGRU --able-cuda --target-RAD --normalize-input \
--gpu 0 --lr 0.0001 --clip --clip-max-norm 1 --weight-decay 0.008 \
--max-epochs 10 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32

clear
python train_GRUs.py --model CONVGRU --able-cuda --target-RAD --normalize-input \
--gpu 0 --lr 0.0001 --clip --clip-max-norm 1 --weight-decay 0.006 \
--max-epochs 10 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32