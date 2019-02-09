# input frames 10, output frames 18
# I size = 60, F size = 60
python convGRU_run.py \
    --root-dir 01_Radar_data/02_numpy_files \
    --ty-list-file ty_list.xlsx \
    --result-dir 04_results/server \
    --I-lat-l 24.6625 --I-lat-h 25.4 --I-lon-l 121.15 --I-lon-h 121.8875 \
    --F-lat-l 24.6625 --F-lat-h 25.4 --F-lon-l 121.15 --F-lon-h 121.8875 \
    --weight-decay 0.1 --gpu 2 --input-with-grid

# input frames 9, output frames 18
# I size = 60, F size = 60
python convGRU_run.py \
    --input-frames 9 \
    --root-dir 01_Radar_data/02_numpy_files \
    --ty-list-file ty_list.xlsx \
    --result-dir 04_results/server \
    --I-lat-l 24.6625 --I-lat-h 25.4 --I-lon-l 121.15 --I-lon-h 121.8875 \
    --F-lat-l 24.6625 --F-lat-h 25.4 --F-lon-l 121.15 --F-lon-h 121.8875 \
    --weight-decay 0.1 --gpu 2

# input frames 7, output frames 18
# I size = 60, F size = 60
python convGRU_run.py \
    --input-frames 7 \
    --root-dir 01_Radar_data/02_numpy_files \
    --ty-list-file ty_list.xlsx \
    --result-dir 04_results/server \
    --I-lat-l 24.6625 --I-lat-h 25.4 --I-lon-l 121.15 --I-lon-h 121.8875 \
    --F-lat-l 24.6625 --F-lat-h 25.4 --F-lon-l 121.15 --F-lon-h 121.8875 \
    --weight-decay 0.1  --gpu 2


# input frames 5, output frames 18
# I size = 60, F size = 60
python convGRU_run.py \
    --input-frames 5 \
    --root-dir 01_Radar_data/02_numpy_files \
    --ty-list-file ty_list.xlsx \
    --result-dir 04_results/server \
    --I-lat-l 24.6625 --I-lat-h 25.4 --I-lon-l 121.15 --I-lon-h 121.8875 \
    --F-lat-l 24.6625 --F-lat-h 25.4 --F-lon-l 121.15 --F-lon-h 121.8875 \
    --weight-decay 0.1  --gpu 2
