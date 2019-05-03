clear
python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.001 --weight-decay 0 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.001 --weight-decay 0.1 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.001 --weight-decay 0.01 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.001 --weight-decay 0.001 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \

python run_trajGRU.py --able-cuda --lr-scheduler --clip --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.001 --weight-decay 0.0001 --max-epochs 10 --batch-size 4 --train-num 10 --optimizer Adam --value-dtype float16 \