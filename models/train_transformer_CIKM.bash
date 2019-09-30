clear
python train_transformer_CIKM.py --model TRANSFORMER --able-cuda --dataset CIKM \
--gpu 0 --lr 0.0005 --weight-decay 0.5 --I-size 101 --F-size 101 \
--max-epochs 30 --batch-size 12 --optimizer Adam --value-dtype float32 \
--I-nframes 5 --F-nframes 10