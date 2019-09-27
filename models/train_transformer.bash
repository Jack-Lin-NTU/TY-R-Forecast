clear
python train_transformer.py --model TRANSFORMER --able-cuda \
--gpu 0 --lr 0.0001 --weight-decay 0.5 --I-size 150 --F-size 150 \
--max-epochs 30 --batch-size 3 --train-num 10 --optimizer Adam --value-dtype float32