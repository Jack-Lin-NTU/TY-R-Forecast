clear
python run_trajGRU.py --able-cuda --lr-scheduler --clip --clip-max-norm 0.001 --input-with-grid --input-with-QPE --normalize-target \
--gpu 0 --lr 0.0002 --weight-decay 0 --max-epochs 30 --batch-size 2 --train-num 10 --optimizer Adam --value-dtype float32 \