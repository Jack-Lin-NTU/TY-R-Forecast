# extraxt data from the original tar.gz files and output the readable files
if [ ! -d "../01_readable_files" ]; then
  python 01_extract_data_to_readable_files.py
fi

# run the data wrangler
python 02_data_wrangler.py
