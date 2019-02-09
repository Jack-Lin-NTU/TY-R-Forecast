# input frames 10, output frames 18
# I size = 180, F size = 60
python convGRU_run_180_180.py \
    --root-dir 01_Radar_data/02_numpy_files \
    --ty-list-file ty_list.xlsx \
    --result-dir 04_results/server \
    --I-lat-l 23.9125 --I-lat-h 26.15 --I-lon-l 120.4 --I-lon-h 122.6375 \
    --F-lat-l 23.9125 --F-lat-h 26.15 --F-lon-l 120.4 --F-lon-h 122.6375 \
    --weight-decay 0.1 --gpu 0 --input-with-grid

python convGRU_run_180_180.py \
    --root-dir 01_Radar_data/02_numpy_files \
    --ty-list-file ty_list.xlsx \
    --result-dir 04_results/server \
    --I-lat-l 23.9125 --I-lat-h 26.15 --I-lon-l 120.4 --I-lon-h 122.6375 \
    --F-lat-l 23.9125 --F-lat-h 26.15 --F-lon-l 120.4 --F-lon-h 122.6375 \
    --weight-decay 0.1 --gpu 0
