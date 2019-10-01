clear
python train_trajGRU_CIKM.py --model TRAJGRU --able-cuda --dataset CIKM \
--gpu 0 --lr 0.0001 --weight-decay 0.1 --I-size 101 --F-size 101 \
--max-epochs 30 --batch-size 16 --train-num 10 --optimizer Adam --value-dtype float32 \
--I-nframes 5 --F-nframes 10