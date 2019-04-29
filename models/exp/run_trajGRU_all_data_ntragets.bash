python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target --load-all-data\
                --I-x-l 121.3375 --I-x-h 121.7 --I-y-l 24.8125 --I-y-h 25.175\
                --gpu 0 --lr 100 --weight-decay 0.00001
                
# python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid\
#                         --I-x-l 121.3375 --I-x-h 121.7 --I-y-l 24.8125 --I-y-h 25.175\
#                         --gpu 0 --lr 1e-3
