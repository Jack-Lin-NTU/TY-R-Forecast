import os
import sys
import gzip
import pandas as pd
import numpy as np
import datetime as dt
from args_tools_jupyter import args, createfolder

pd.set_option('precision', 4)

def extract_original_data():
    '''
    Arguments:
        This function is to extract the selected event data form original data.
    '''
    #load typhoon list file
    ty_list = pd.read_csv(args.ty_list)
    ty_list.loc[:, 'Time of issuing'] = pd.to_datetime(ty_list.loc[:, 'Time of issuing'])
    ty_list.loc[:, 'Time of canceling'] = pd.to_datetime(ty_list.loc[:, 'Time of canceling'])
    
    radar_original_data_folder = args.radar_original_data_folder
    radar_compressed_data_folder = args.radar_compressed_data_folder

    for i in range(len(ty_list)):
        # get the start and end time of the typhoon
        year = ty_list.loc[i, 'Time of issuing'].year
        time_start = ty_list.loc[i, 'Time of issuing'] - dt.timedelta(hours=8)
        time_end =ty_list.loc[i, 'Time of canceling'] - dt.timedelta(hours=8)
        ty_name = ty_list.loc[i, 'En name']
        info = '|{:8s}| start time: {} | end time: {} |'.format(ty_name, time_start, time_end)
        print(info)

        tmp_path1 = os.path.join(radar_original_data_folder, str(year))
        
        for j in sorted(os.listdir(tmp_path1)):
            if time_start.date() <= dt.datetime.strptime(j, '%Y%m%d').date() <= time_end.date():
                tmp_path2 = os.path.join(tmp_path1, j)
                output_folder = os.path.join(radar_compressed_data_folder, str(year)+'.'+ty_name)
                createfolder(output_folder)
                for k in os.listdir(tmp_path2):
                    tmp_path3 = os.path.join(tmp_path2, k)
                    for o in os.listdir(tmp_path3):
                        if time_start <= dt.datetime.strptime(o[-16:-3], '%Y%m%d.%H%M') <= time_end:
                            tmp_path4 = os.path.join(tmp_path3, o)
                            output_path = os.path.join(output_folder, o)

                            command = 'cp {:s} {:s}'.format(tmp_path4, output_path)
                            os.system(command)
                        else:
                            pass
            else:
                pass
        print('|'+'-'*(len(info)-2)+'|')

def output_files():
    '''
    Arguments:
        This function is to uncompress the extracted files and output the wrangled files.
    '''
    # load typhoon list file
    ty_list = pd.read_csv(args.ty_list)
    ty_list.loc[:, 'Time of issuing'] = pd.to_datetime(ty_list.loc[:, 'Time of issuing'])
    ty_list.loc[:, 'Time of canceling'] = pd.to_datetime(ty_list.loc[:, 'Time of canceling'])

    radar_compressed_data_folder = args.radar_compressed_data_folder
    tmp_uncompressed_folder = os.path.join(args.radar_folder, 'tmp')
    createfolder(tmp_uncompressed_folder)
    
    count_qpe = {}
    count_qpf = {}
    count_rad = {}
    # uncompress the file and output the readable file
    for i in sorted(os.listdir(radar_compressed_data_folder)):
        print('-' * 40)
        print(i)
        radar_compressed_data_folder = os.path.join(args.radar_compressed_data_folder, i)
        count_qpe[i] = len([x for x in os.listdir(radar_compressed_data_folder) if 'C' == x[0]])
        count_qpf[i] = len([x for x in os.listdir(radar_compressed_data_folder) if 'M' == x[0]])
        count_rad[i] = len([x for x in os.listdir(radar_compressed_data_folder) if 'q' == x[0]])
        
        for j in sorted(os.listdir(radar_compressed_data_folder)):
            compressed_file = os.path.join(radar_compressed_data_folder, j)
            outputtime = j[-16:-8]+j[-7:-3]
            outputtime = dt.datetime.strftime(dt.datetime.strptime(outputtime, '%Y%m%d%H%M')+dt.timedelta(hours=8), '%Y%m%d%H%M')

            if j[0] == 'C':
                name = 'QPE'
                output_folder = os.path.join(args.radar_wrangled_data_folder, name)
                createfolder(output_folder)
            elif j[0] == 'M':
                name = 'RAD'
                output_folder = os.path.join(args.radar_wrangled_data_folder, name)
                createfolder(output_folder)
            elif j[0] == 'q':
                name = 'QPF'
                output_folder = os.path.join(args.radar_wrangled_data_folder, name)
                createfolder(output_folder)
            
            tmp_uncompressed_file = os.path.join(tmp_uncompressed_folder, name+'_'+outputtime)

            # define the object of gzip
            g_file = gzip.GzipFile(compressed_file)
            # use read() to open gzip and write into the open file。
            open(tmp_uncompressed_file, 'wb').write(g_file.read())
            # close the object of gzip
            g_file.close()

            tmp_file_out = os.path.join(tmp_uncompressed_folder, name+'_'+outputtime+'.txt')
            bashcommand = os.path.join('.', args.fortran_code_folder, '{:s}.out {:s} {:s}'.format(name, tmp_uncompressed_file, tmp_file_out))
            os.system(bashcommand)

            data = pd.read_csv(tmp_file_out, sep='\s+', header=None)
            output_path = os.path.join(output_folder, i+'.'+outputtime)
            
            data.columns = pd.Index(np.linspace(args.O_x[0], args.O_x[1], args.O_shape[0]), name='longitude')
            data.index = pd.Index(np.linspace(args.O_y[1], args.O_y[0], args.O_shape[1]), name='latitude')
            data.to_pickle(output_path+'.pkl', compression=args.compression)
            
            os.remove(tmp_uncompressed_file)
            os.remove(tmp_file_out)

    return count_qpe, count_qpf, count_rad

def check_data_and_create_miss_data():
    '''
    Arguments:
        This function is to check whether the wrangled files are continuous in each typhoon events and address missing files.
    '''
    # Set path
    radar_wrangled_data_folder = args.radar_wrangled_data_folder

    ty_list = pd.read_csv(args.ty_list)
    ty_list.loc[:, 'Time of issuing'] = pd.to_datetime(ty_list.loc[:, 'Time of issuing'])
    ty_list.loc[:, 'Time of canceling'] = pd.to_datetime(ty_list.loc[:, 'Time of canceling'])
    count_qpe = {}
    count_qpf = {}
    count_rad = {}
    for i in sorted(os.listdir(radar_wrangled_data_folder)):
        for j in ty_list.loc[:, 'En name']:
            if i == 'QPE':
                count_qpe[j] = len([x for x in os.listdir(os.path.join(radar_wrangled_data_folder, 'QPE')) if j in x])
            elif i == 'QPF':
                pass
                # count_qpf[j] = len([x for x in os.listdir(os.path.join(radar_wrangled_data_folder, 'QPF')) if j in x])
            else:
                count_rad[j] = len([x for x in os.listdir(os.path.join(radar_wrangled_data_folder, 'RAD')) if j in x])

    qpe_list = [x[-16:-4] for x in os.listdir(os.path.join(radar_wrangled_data_folder, 'QPE'))]
    # qpf_list = [x[-16:-4] for x in os.listdir(os.path.join(radar_wrangled_data_folder, 'QPF'))]
    rad_list = [x[-16:-4] for x in os.listdir(os.path.join(radar_wrangled_data_folder, 'RAD'))]

    qpe_list_miss = []
    # qpf_list_miss = []
    rad_list_miss = []
    
    
    file_end = '.pkl'

    for i in np.arange(len(ty_list)):
        for j in np.arange(1000):
            time = ty_list.loc[i, 'Time of issuing'] + pd.Timedelta(minutes=10*j)
            if time > ty_list.loc[i, 'Time of canceling']:
                break
            time = time.strftime('%Y%m%d%H%M')
            if time not in qpe_list:
                qpe_list_miss.append(time[:4]+'.'+ty_list.loc[i, 'En name']+'.'+time+file_end)
            # if time not in qpf_list:
            #     qpf_list_miss.append(time[:4]+'.'+ty_list.loc[i, 'En name']+'.'+time+file_end)
            if time not in rad_list:
                rad_list_miss.append(time[:4]+'.'+ty_list.loc[i, 'En name']+'.'+time+file_end)
    missfiles = np.concatenate([np.array(qpe_list_miss), np.array(rad_list_miss)])
    missfiles_index = []
    for i in range(len(qpe_list_miss)):
        missfiles_index.append('QPE')
    for i in range(len(rad_list_miss)):
        missfiles_index.append('RAD')
    # for i in range(len(qpf_list_miss)):
    #     missfiles_index.append('QPF')

    missfiles = pd.DataFrame(missfiles, index=missfiles_index, columns=['File_name'])
    missfiles.to_csv(os.path.join(args.radar_folder, 'Missing_files.csv'))
    
    for i in range(len(missfiles)):
        missdatatime = dt.datetime.strptime(missfiles.iloc[i, 0][-16:-4], '%Y%m%d%H%M')
        forwardtime = dt.datetime.strftime((missdatatime - dt.timedelta(minutes=10)), '%Y%m%d%H%M')
        backwardtime = dt.datetime.strftime((missdatatime + dt.timedelta(minutes=10)), '%Y%m%d%H%M')

        forwardfile = missfiles.iloc[i, 0][:-16] + forwardtime + missfiles.iloc[i, 0][-4:]
        backwardfile  = missfiles.iloc[i, 0][:-16] + backwardtime + missfiles.iloc[i, 0][-4:]
        
        forwarddata = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, missfiles.index[i], forwardfile), compression=args.compression)
        backwarddata = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, missfiles.index[i], backwardfile), compression=args.compression)
        data = (forwarddata+backwarddata)/2
        data.to_pickle(os.path.join(args.radar_wrangled_data_folder, missfiles.index[i], missfiles.iloc[i, 0]), compression=args.compression)
        
            
    return count_qpe, count_qpf, count_rad

def overall_of_data():
    '''
    Arguments:
        This function is to summarize the overall property of the wrangled data.
    '''
    # Set path
    measures = pd.DataFrame(np.ones((4,3))*10, index=['max','min','mean','std'], columns=sorted(os.listdir(args.radar_wrangled_data_folder)))
    measures.index.name = 'Measures'
    tmp_mean = 0
    for i in sorted(os.listdir(args.radar_wrangled_data_folder)):
        for j in sorted(os.listdir(os.path.join(args.radar_wrangled_data_folder, i))):
            tmp_data = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, i , j), compression=args.compression).values
            if measures.loc['max', i] < np.max(tmp_data):
                measures.loc['max', i] = np.max(tmp_data)
            if measures.loc['min', i] > np.min(tmp_data):
                measures.loc['min', i] = np.min(tmp_data)
            tmp_mean += np.mean(tmp_data)/100
        measures.loc['mean', i] = tmp_mean/len(os.listdir(os.path.join(args.radar_wrangled_data_folder, i)))*100
    tmp = 0
    for i in sorted(os.listdir(args.radar_wrangled_data_folder)):
        for j in sorted(os.listdir(os.path.join(args.radar_wrangled_data_folder, i))):
            tmp_data = pd.read_pickle(os.path.join(args.radar_wrangled_data_folder, i , j), compression=args.compression).values
            tmp += np.mean((tmp_data - measures.loc['mean', i])**2)
        measures.loc['std', i] = np.sqrt(tmp/len(os.listdir(os.path.join(args.radar_wrangled_data_folder, i))))
    output_path = os.path.join(args.radar_folder,'overall.csv')
    measures.to_csv(output_path)
    
    return measures