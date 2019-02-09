python cnn2D_run.py \
    --root-dir 01_Radar_data/02_numpy_files \
    --ty-list-file ty_list.xlsx \
    --result-dir 04_results/server \
    --I-lat-l 24.6625 --I-lat-h 25.4 --I-lon-l 121.15 --I-lon-h 121.8875 \
    --F-lat-l 24.6625 --F-lat-h 25.4 --F-lon-l 121.15 --F-lon-h 121.8875 \
    --weight-decay 0.1  --gpu 2
