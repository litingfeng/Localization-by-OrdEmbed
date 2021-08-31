"""
https://github.com/wyharveychen/CloserLookFewShot/blob/master/filelists/CUB/write_CUB_filelist.py
Created on 5/11/2021 9:38 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = '/research/cbim/vast/tl601/Dataset/'
data_path = join(cwd,'CUB_200_2011/images')
savedir = os.path.join(cwd, 'CUB_200_2011', 'fewshot')
#dataset_list = ['base','val','novel']
dataset_list = ['val']

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    if 59 <= int(folder.split('.')[0]) <= 64:
        classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
        random.shuffle(classfile_list_all[-1])

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        file_list = file_list + classfile_list
        label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        # if 'base' in dataset:
        #     if (i%2 == 0):
        #         file_list = file_list + classfile_list
        #         label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        # if 'val' in dataset:
        #     if (i%4 == 1):
        #         file_list = file_list + classfile_list
        #         label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        # if 'novel' in dataset:
        #     if (i%4 == 3):
        #         file_list = file_list + classfile_list
        #         label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    print('label_list ', len(label_list), ' file_list ', len(file_list))
    selected_label_list = list(set(label_list))
    print('sel ', len(selected_label_list))
    fo = open(savedir + '/gull_59_64.json', "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for i, item in enumerate(file_list)
                   if label_list[i] in selected_label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list
                   if item  in selected_label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)