clear
python run_trajGRU.py --able-cuda --lr-scheduler --clip --clip-max-norm 0.005 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0001 --weight-decay 0 --max-epochs 30 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --clip-max-norm 0.005 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.00005 --weight-decay 0 --max-epochs 30 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

clear
python run_trajGRU.py --able-cuda --lr-scheduler --clip --clip-max-norm 0.005 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.000025 --weight-decay 0 --max-epochs 30 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \