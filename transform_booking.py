import torch
import tarfile
import pickle
import pandas
import json
import argparse
from pathlib import Path
import numpy as np
import shutil
from shutil import copyfile
import os
import re
import pandas as pd
import sys
from numpy import asarray
from numpy import savetxt
import numpy as np
sys.path.append("..")
def parse_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_dir', type=str, required=True, help='dir holding sequences as separate files')
    parser.add_argument('--save_dir', type=str, default = './', help='path to save results')
    args = parser.parse_args()
    return args
def tranform_data(args):
    """
    Loads the sequences saved in the given directory.
    Args:
        data_dir    (str, Path) - directory containing sequences
        save_dir - directory for saving transform data
        maxsize     (int)       - maximum number of sequences to load
        maxlen      (int)       - maximum length of sequence, the sequences longer than maxlen will be truncated
        ext         (str)       - extension of files in data_dir directory
        datetime    (bool)      - variable meaning if time values in files are represented in datetime format
             
    """
    save_dir = args.save_dir
    os.makedirs(save_dir)
    gt_ids = None
    df = pd.read_csv('data/booking_challenge_tpp_labeled.csv')
    ids1 = df['user_id'].unique()
    ids = [] #[]
    for i in range(len(ids1)):
        a =  len(df.loc[df['user_id'].isin([ids1[i]])]['user_id'].to_numpy())
        print (a, '\n')
        if a > 10:
            ids.append(ids1[i])
    gt_data = []
    #print(test)
    #print("test", df.loc[df['user_id'].isin(['773518', '4569712'])]['diff_checkout'])
    for i in range(len(ids)):
        print("id", ids[i], "\n")
        data = {'time': df.loc[df['user_id'].isin([ids[i]])]['checkin'][:-1], 'event0': df.loc[df['user_id'].isin([ids[i]])]['device_class'][:-1], 'event1': df.loc[df['user_id'].isin([ids[i]])]['nr_trips'][:-1],  'event2': df.loc[df['user_id'].isin([ids[i]])]['checkout'][:-1],  'event3': df.loc[df['user_id'].isin([ids[i]])]['diff_checkin'][:-1],  'event4': df.loc[df['user_id'].isin([ids[i]])]['diff_checkout'][:-1],  'event5': df.loc[df['user_id'].isin([ids[i]])]['diff_inout'][:-1]}
        print(f'Reading {i}')
        df_data = pd.DataFrame(data=data)
        df_data = df_data.reset_index(drop=True)
        print("dfdf", df_data['time'])
        df_data['time'] = pd.to_datetime(df_data['time'])

        df_data['time'] = (df_data['time'] - df_data['time'][0]) / np.timedelta64(1,'D')
        df_data['event2'] = pd.to_datetime(df_data['event2'])
        df_data['event2'] = (df_data['event2'] - df_data['event2'][0]) / np.timedelta64(1,'D')
        os.mknod(Path(save_dir,f'{i+1}.csv'))
        df_data.to_csv(Path(save_dir,f'{i+1}.csv'))
        gt_data.append(df.loc[df['user_id'].isin([ids[i]])]['label'].iloc[0])
        print("gt_data", gt_data[i])
    
    gt = {'cluster_id': gt_data}
    cl_data = pd.DataFrame(data=gt)
    cl_data.to_csv(Path(save_dir, 'clusters.csv'))
    return "Yes"



args = parse_arguments()
print(args)
tranform_data(args)