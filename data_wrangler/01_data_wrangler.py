import sys
import gzip
import pandas as pd
import datetime as dt
import os
from args_tools import args, createfolder

def extract_original_data():
    #load typhoon list file
    ty_list = pd.read_excel(args.ty_list)

    origin_files_folder = args.origin_files_folder
    compressed_files_folder = args.compressed_files_folder

    for i in range(len(ty_list)):
        # get the start and end time of the typhoon
        year = ty_list.loc[i,"Time of issuing"].year
        date_s = dt.datetime.strftime(ty_list["Time of issuing"][i] - dt.timedelta(hours=8),format="%Y%m%d")
        date_e = dt.datetime.strftime(ty_list["Time of canceling"][i] - dt.timedelta(hours=8),format="%Y%m%d")
        time_s = dt.datetime.strftime(ty_list["Time of issuing"][i] - dt.timedelta(hours=8),format="%H%M")
        time_e = dt.datetime.strftime(ty_list["Time of canceling"][i] - dt.timedelta(hours=8),format="%H%M")
        ty_name = ty_list.loc[i,"En name"]

        print("|{:8s}| start time: {:s} | end time: {:s} |".format(ty_name+"",date_s,date_e))
        print("|{:8s}|             {:8s} |           {:8s} |".format(" ",time_s,time_e))

        tmp_path1 = os.path.join(origin_files_folder,str(year))
        for j in os.listdir(tmp_path1):
            if dt.datetime.strptime(date_s,"%Y%m%d") <= dt.datetime.strptime(j,"%Y%m%d") <= dt.datetime.strptime(date_e,"%Y%m%d"):
                tmp_path2 = os.path.join(tmp_path1,j)
                output_folder = os.path.join(compressed_files_folder, str(year)+'.'+ty_name)
                createfolder(output_folder)
                for k in os.listdir(tmp_path2):
                    tmp_path3 = os.path.join(tmp_path2,k)
                    for o in os.listdir(tmp_path3):
                        if dt.datetime.strptime(date_s+"."+time_s,"%Y%m%d.%H%M") <= dt.datetime.strptime(o[-16:-3],"%Y%m%d.%H%M") <= dt.datetime.strptime(date_e+"."+time_e,"%Y%m%d.%H%M"):
                            tmp_path4 = os.path.join(tmp_path3,o)
                            output_path = os.path.join(output_folder,o)

                            command = "cp {:s} {:s}".format(tmp_path4, output_path)
                            os.system(command)
                        else:
                            pass
            else:
                pass
        print("|----------------------------------------------------|")

def uncompress_and_output_numpy_files():
    # load typhoon list file
    ty_list = pd.read_excel(args.ty_list)

    compressed_files_folder = args.compressed_files_folder
    tmp_uncompressed_folder = 'tmp'
    createfolder(tmp_uncompressed_folder)
    count_qpe = {}
    count_qpf = {}
    count_rad = {}
    # uncompress the file and output the readable file
    for i in sorted(os.listdir(compressed_files_folder)):
        print("-" * 40)
        print(i)
        compressed_files_folder = os.path.join(args.compressed_files_folder,i)
        count_qpe[i] = len([x for x in os.listdir(compressed_files_folder) if 'C' == x[0]])
        count_qpf[i] = len([x for x in os.listdir(compressed_files_folder) if 'M' == x[0]])
        count_rad[i] = len([x for x in os.listdir(compressed_files_folder) if 'q' == x[0]])
        for j in sorted(os.listdir(compressed_files_folder)):
            compressed_file = os.path.join(compressed_files_folder,j)
            outputtime = j[-16:-8]+j[-7:-3]
            outputtime = dt.datetime.strftime(dt.datetime.strptime(outputtime,"%Y%m%d%H%M")+dt.timedelta(hours=8),"%Y%m%d%H%M")

            if j[0] == "C":
                name = 'QPE'
                output_folder = os.path.join(args.numpy_files_folder,name)
                createfolder(output_folder)
            elif j[0] == "M":
                name = 'RAD'
                output_folder = os.path.join(args.numpy_files_folder,name)
                createfolder(output_folder)
            elif j[0] == "q":
                name = 'QPF'
                output_folder = os.path.join(args.numpy_files_folder,name)
                createfolder(output_folder)
            tmp_uncompressed_file = os.path.join(tmp_uncompressed_folder,name+'_'+outputtime)

            g_file = gzip.GzipFile(compressed_file)
            # 创建gzip对象
            open(tmp_uncompressed_file, "wb").write(g_file.read())
            # gzip对象用read()打开后，写入open()建立的文件中。
            g_file.close()

            tmp_file_out = os.path.join(tmp_uncompressed_folder, name+"_"+outputtime+".txt")
            bashcommand = os.path.join('.',args.fortran_code_folder,'{:s}.out {:s} {:s}'.format(name, tmp_uncompressed_file, tmp_file_out))

            os.system(bashcommand)

            data = pd.read_table(tmp_file_out,delim_whitespace=True,header=None)
            output_path = os.path.join(output_folder,i+"."+outputtime)

            np.savez_compressed(output_path, data=data)

            os.remove(tmp_uncompressed_file)
            os.remove(tmp_file_out)

    return count_qpe, count_qpf, count_rad

def check_data_and_create_miss_data():
    # Set path
    numpy_files_folder = args.numpy_files_folder

    ty_list = pd.read_excel(args.ty_list)

    count_qpe = {}
    count_qpf = {}
    count_rad = {}
    for i in sorted(os.listdir(numpy_files_folder)):
        for j in ty_list.loc[:,"En name"]:
            if i == "QPE":
                count_qpe[j] = len([x for x in os.listdir(os.path.join(numpy_files_folder,"QPE")) if j in x])
            elif i == "QPF":
                pass
                # count_qpf[j] = len([x for x in os.listdir(os.path.join(numpy_files_folder,"QPF")) if j in x])
            else:
                count_rad[j] = len([x for x in os.listdir(os.path.join(numpy_files_folder,"RAD")) if j in x])

    qpe_list = [x[-16:-4] for x in os.listdir(os.path.join(numpy_files_folder,"QPE"))]
    # qpf_list = [x[-16:-4] for x in os.listdir(os.path.join(numpy_files_folder,"QPF"))]
    rad_list = [x[-16:-4] for x in os.listdir(os.path.join(numpy_files_folder,"RAD"))]

    qpe_list_miss = []
    # qpf_list_miss = []
    rad_list_miss = []

    for i in np.arange(len(ty_list)):
        for j in np.arange(1000):
            time = ty_list.loc[i,"Time of issuing"] + pd.Timedelta(minutes=10*j)
            if time > ty_list.loc[i,"Time of canceling"]:
                break
            time = time.strftime("%Y%m%d%H%M")
            if time not in qpe_list:
                qpe_list_miss.append(time[:4]+"."+ty_list.loc[i,"En name"]+"."+time+".npz")
            # if time not in qpf_list:
            #     qpf_list_miss.append(time[:4]+"."+ty_list.loc[i,"En name"]+"."+time+".npz")
            if time not in rad_list:
                rad_list_miss.append(time[:4]+"."+ty_list.loc[i,"En name"]+"."+time+".npz")
    missfiles = np.concatenate([np.array(qpe_list_miss),np.array(rad_list_miss)])
    missfiles_index = []
    for i in range(len(qpe_list_miss)):
        missfiles_index.append("QPE")
    for i in range(len(rad_list_miss)):
        missfiles_index.append("RAD")
    # for i in range(len(qpf_list_miss)):
    #     missfiles_index.append("QPF")

    missfiles = pd.DataFrame(missfiles,index=missfiles_index,columns=["File_name"])
    missfiles.to_excel(os.path.join(args.radar_folder,"Missing_files.xlsx"))
    for i in range(len(missfiles)):
        missdatatime = dt.datetime.strptime(missfiles.iloc[i,0][-16:-4],"%Y%m%d%H%M")
        forwardtime = dt.datetime.strftime((missdatatime - dt.timedelta(minutes=10)),"%Y%m%d%H%M")
        backwardtime = dt.datetime.strftime((missdatatime + dt.timedelta(minutes=10)),"%Y%m%d%H%M")

        forwardfile = missfiles.iloc[i,0][:-16] + forwardtime + missfiles.iloc[i,0][-4:]
        backwardfile  = missfiles.iloc[i,0][:-16] + backwardtime + missfiles.iloc[i,0][-4:]

        forwarddata = np.load(os.path.join(args.numpy_files_folder,missfiles.index[i],forwardfile))['data']
        backwarddata = np.load(os.path.join(args.numpy_files_folder,missfiles.index[i],backwardfile))['data']
        data = (forwarddata+backwarddata)/2
        np.savez_compressed(os.path.join(args.numpy_files_folder,missfiles.index[i],missfiles.iloc[i,0]), data=data)
    return count_qpe, count_qpf, count_rad


if __name__ == "__main__":
    info = "*{:^58s}*".format('Data extracter')
    print("*" * len(info))
    print(info)
    print("*" * len(info))
    print("-" * len(info))

    if os.path.exists(args.compressed_files_folder):
        print('Already extract oringinal data')
        print("-" * len(info))
    else:
        print(extract_original_data())
        print("-" * len(info))

    if args.numpy_files_folder.exists():
        print('Already output numpy files')
        print("-" * len(info))
    else:
        print(uncompress_and_output_numpy_files())
        print("-" * len(info))

    count_qpe, _, count_rad = check_data_and_create_miss_data()
    print(count_qpe)
    print(count_rad)
