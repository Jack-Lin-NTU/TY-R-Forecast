clear
python run_trajGRU.py --able-cuda --lr-scheduler --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 1 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.5 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.1 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.05 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.01 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.005 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \