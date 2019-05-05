clear
# python run_convGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
# --gpu 0 --lr 0.5 --weight-decay 0 --max-epochs 10 --batch-size 8 --train-num 10 --optimizer Adam --value-dtype float16 \

# python run_convGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
# --gpu 0 --lr 0.1 --weight-decay 0 --max-epochs 10 --batch-size 8 --train-num 10 --optimizer Adam --value-dtype float16 \

# python run_convGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
# --gpu 0 --lr 0.05 --weight-decay 0 --max-epochs 10 --batch-size 8 --train-num 10 --optimizer Adam --value-dtype float16 \

# python run_convGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
# --gpu 0 --lr 0.01 --weight-decay 0 --max-epochs 10 --batch-size 8 --train-num 10 --optimizer Adam --value-dtype float16 \

# python run_convGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
# --gpu 0 --lr 0.005 --weight-decay 0 --max-epochs 10 --batch-size 8 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_convGRU.py --able-cuda --lr-scheduler --clip --clip-max-norm 100 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0005 --weight-decay 0 --max-epochs 10 --batch-size 8 --train-num 10 --optimizer Adam --value-dtype float16 \

# --I-x-l 118.9 --I-x-h 122.2675 --I-y-l 21.75 --I-y-h 25.1125 \
# --F-x-l 118.9 --F-x-h 122.2675 --F-y-l 21.75 --F-y-h 25.1125
