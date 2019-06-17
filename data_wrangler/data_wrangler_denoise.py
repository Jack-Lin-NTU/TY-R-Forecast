import os
import sys
import gzip
import numpy as np
import pandas as pd
import datetime as dt
from utils.parser import get_args
from utils.tools import createfolder, checkpath, print_dict, make_path
from data_wrangler import output_files, check_data_and_create_miss_data

if __name__ == '__main__':
    args = get_args()
    args.denoise = 20
    checkpath(args.ty_list)
    checkpath(args.TW_map_file+'.prj')
    checkpath(args.radar_wrangled_data_folder)
    
    args.radar_wrangled_data_folder += '_denoise'

    _ = output_files(args)
    # check data and wrangle missing data
    _ = check_data_and_create_miss_data(args)

    ## deal with the specific data
    # 2012.SAOLA
    data1 = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2012.SAOLA.201208021530.pkl'),compression=args.compression)
    data2 = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2012.SAOLA.201208021620.pkl'),compression=args.compression)

    data = data1 + (data2 - data1)/5*1
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2012.SAOLA.201208021540.pkl'),compression=args.compression)
    data = data1 + (data2 - data1)/5*2
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2012.SAOLA.201208021550.pkl'),compression=args.compression)
    data = data1 + (data2 - data1)/5*3
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2012.SAOLA.201208021600.pkl'),compression=args.compression)
    data = data1 + (data2 - data1)/5*4
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2012.SAOLA.201208021610.pkl'),compression=args.compression)

    # 2015.SOUDELOR
    l1 = ['080700','080800','081000','081200','081750','081850','082050','090150','090700','091000']
    l2 = ['080710','080810','081010','081210','081800','081900','082100','090200','090710','091010']
    ll = ['080720','080820','081020','081220','081810','081910','082110','090210','090720','091020']
    for i in range(len(ll)):
        data1 = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2015.SOUDELOR.201508'+l1[i]+'.pkl'),compression=args.compression)
        data2 = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2015.SOUDELOR.201508'+l2[i]+'.pkl'),compression=args.compression)

        data = data1 + (data2 - data1)/2
        data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2015.SOUDELOR.201508'+ll[i]+'.pkl'),compression=args.compression)
        
    data1 = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2015.SOUDELOR.201508081100.pkl'),compression=args.compression)
    data2 = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2015.SOUDELOR.201508081130.pkl'),compression=args.compression)

    data = data1 + (data2 - data1)/3*1
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2015.SOUDELOR.201508081110.pkl'),compression=args.compression)
    data = data1 + (data2 - data1)/3*2
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2015.SOUDELOR.201508081120.pkl'),compression=args.compression)

    # 2016.MEGI
    data1 = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2016.MEGI.201609271830.pkl'),compression=args.compression)
    data2 = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2016.MEGI.201609271740.pkl'),compression=args.compression)

    data = data1 + (data2 - data1)/7*1
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2016.MEGI.201609271840.pkl'),compression=args.compression)
    data = data1 + (data2 - data1)/7*2
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2016.MEGI.201609271850.pkl'),compression=args.compression)
    data = data1 + (data2 - data1)/7*3
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2016.MEGI.201609271900.pkl'),compression=args.compression)
    data = data1 + (data2 - data1)/7*4
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2016.MEGI.201609271910.pkl'),compression=args.compression)
    data = data1 + (data2 - data1)/7*5
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2016.MEGI.201609271920.pkl'),compression=args.compression)
    data = data1 + (data2 - data1)/7*6
    data.to_pickle(os.path.join(args.radar_wrangled_data_folder, 'RAD', '2016.MEGI.201609271930.pkl'),compression=args.compression)