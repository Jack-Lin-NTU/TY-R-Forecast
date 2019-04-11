python train.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target\
                --I-x-l 121.3375 --I-x-h 121.7 --I-y-l 24.8125 --I-y-h 25.175\
                --gpu 0 --dtype float16 --lr 1e-3 --weight-decay 0.0001

python train.py --able-cuda --clip --input-with-grid --input-with-QPE --normalize-target\
                --I-x-l 121.3375 --I-x-h 121.7 --I-y-l 24.8125 --I-y-h 25.175\
                --gpu 0 --dtype float16 --lr 1e-3 --weight-decay 0.0001