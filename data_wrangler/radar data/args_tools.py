import argparse
import numpy as np
import pandas as pd
import math
import datetime as dt
import os


def createfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def make_path(path, workfolder):
    if path[0] == '~':
        p = os.path.expanduser(path)
    else:
        p = path

    if not os.path.isabs(p):
        return os.path.join(os.path.expanduser(workfolder), p)
    else:
        return p

parser = argparse.ArgumentParser()

working_folder = os.path.expanduser('~/OneDrive/01_IIS/04_TY_research/01_Radar_data')

parser.add_argument("--study-area", default="Taipei", metavar='', type=str)

parser.add_argument("--radar-folder", default=working_folder,
                    metavar='', type=str, help="The folder path of the radar data.")

parser.add_argument("--ty-list", default='ty_list.xlsx',
                    metavar='', type=str, help="The file path of the typhoon list file.")
parser.add_argument("--sta-list", default='sta_list_all.xlsx',
                    metavar='', type=str, help="The file path of the station list file.")
parser.add_argument("--TW-map-file", default='TW_shapefile/gadm36_TWN_2',
                    metavar='', type=str, help="The file path of the TW-map file.")


parser.add_argument("--fortran-code-folder", default="fortran_codes/", metavar='', type=str, help="The path of the fortran-code folder")

parser.add_argument("--origin-files-folder", default="/ssd/research/origianal_radar_data_2012-2018", metavar='', type=str,
                    help="The path of the original files folder")

parser.add_argument("--compressed-files-folder", default='01_compressed_files',
                    metavar='', type=str, help="The folder path that the compressed files in")
parser.add_argument("--numpy-files-folder", default='02_numpy_files',
                    metavar='', type=str, help="The folder path that the numpy files in")
parser.add_argument("--figures-folder", default='03_figures',
                    metavar='', type=str, help="The folder path that the figures in")


parser.add_argument("--I-lat-l", default=23.9125, type=float, metavar='',
                    help='The lowest latitude of the input frames')
parser.add_argument("--I-lat-h", default=26.15, type=float, metavar='',
                    help='The highest latitude of the input frames')
parser.add_argument("--I-lon-l", default=120.4, type=float, metavar='',
                    help='The lowest longitude of the input frames')
parser.add_argument("--I-lon-h", default=122.6375, type=float, metavar='',
                    help='The highest longitude of the input frames')

parser.add_argument("--F-lat-l", default=24.6625, type=float, metavar='',
                    help='The lowest latitude of the forecast frames')
parser.add_argument("--F-lat-h", default=25.4, type=float, metavar='',
                    help='The highest latitude of the forecast frames')
parser.add_argument("--F-lon-l", default=121.15, type=float, metavar='',
                    help='The lowest longitude of the forecast frames')
parser.add_argument("--F-lon-h", default=121.8875, type=float, metavar='',
                    help='The highest longitude of the forecast frames')


parser.add_argument("--res-degree", default=0.0125, type=float, metavar='',
                    help='The res_degree degree of the data')

args = parser.parse_args()


# trans path string to Path
args.ty_list = make_path(args.ty_list, args.radar_folder)
args.sta_list = make_path(args.sta_list, args.radar_folder)
args.TW_map_file = make_path(args.TW_map_file, args.radar_folder)

args.compressed_files_folder = make_path(args.compressed_files_folder, args.radar_folder)
args.numpy_files_folder = make_path(args.numpy_files_folder, args.radar_folder)
args.figures_folder = make_path(args.figures_folder, args.radar_folder)

args.origin_lat_l = 20
args.origin_lat_h = 27
args.origin_lon_l = 118
args.origin_lon_h = 123.5

args.input_size=(math.ceil((args.I_lon_h-args.I_lon_l)/args.res_degree)+1,math.ceil((args.I_lat_h-args.I_lat_l)/args.res_degree)+1)
args.forecast_size=(math.ceil((args.F_lon_h-args.F_lon_l)/args.res_degree)+1,math.ceil((args.F_lat_h-args.F_lat_l)/args.res_degree)+1)
args.origin_lat_size=math.ceil((args.origin_lat_h-args.origin_lat_l)/args.res_degree)+1
args.origin_lon_size=math.ceil((args.origin_lon_h-args.origin_lon_l)/args.res_degree)+1

args.I_x_left = int((args.I_lon_l-args.origin_lon_l)/args.res_degree + 1)
args.I_x_right = int(args.I_x_left + (args.I_lon_h-args.I_lon_l)/args.res_degree + 1)
args.I_y_low = int((args.I_lat_l-args.origin_lat_l)/args.res_degree + 1)
args.I_y_high = int(args.I_y_low + (args.I_lat_h-args.I_lat_l)/args.res_degree + 1)

if __name__ == "__main__":
    print(args.ty_list)
    print(args.sta_list)
    print(args.TW_map_file)
    print(args.compressed_files_folder)
    print(args.numpy_files_folder)
    print(args.figures_folder)
