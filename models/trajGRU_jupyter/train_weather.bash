python train.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target\
                --I-x-l 121.3375 --I-x-h 121.7 --I-y-l 24.8125 --I-y-h 25.175\
                --weather-list PP01 --weather-list PS01 --weather-list RH01\
                --weather-list WD01 --weather-list WD02 --weather-list TX01\
                --gpu 1 --dtype float16 --lr 1e-3 --weight-decay 0.0001
                
python train.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target\
                --I-x-l 121.3375 --I-x-h 121.7 --I-y-l 24.8125 --I-y-h 25.175\
                --weather-list PP01 --weather-list PS01 --weather-list RH01\
                --weather-list WD01 --weather-list WD02 --weather-list TX01\
                --gpu 1 --dtype float16 --lr 1e-3 --weight-decay 0.0005