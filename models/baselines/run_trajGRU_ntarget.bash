clear
python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --clip --clip-max-norm 0.09 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --clip --clip-max-norm 0.08 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --clip --clip-max-norm 0.07 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python train_GRUs.py --model TRAJGRU --able-cuda --lr-scheduler --clip --clip-max-norm 0.06 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \
