# RAD_no_weather_grid_scheduler_Adam
# clear
# python train_GRUs.py --model CONVGRU --able-cuda --lr-scheduler --clip --clip-max-norm 5  --input-with-grid \
# --gpu 1 --lr 0.0005 --weight-decay 0 --max-epochs 100 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32 \

# RAD_no_weather_scheduler_Adam(X)
# clear
# python train_GRUs.py --model CONVGRU --able-cuda --lr-scheduler --clip --clip-max-norm 5 \
# --gpu 1 --lr 0.0005 --weight-decay 0 --max-epochs 100 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32 \

# RAD_no_weather_grid_scheduler_Adam
clear
python train_GRUs.py --model CONVGRU --able-cuda --lr-scheduler --clip --clip-max-norm 5 \
--gpu 1 --lr 0.0005 --weight-decay 0 --max-epochs 100 --batch-size 6 --train-num 10 --optimizer Adam --value-dtype float32