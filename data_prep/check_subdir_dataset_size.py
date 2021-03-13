import os, json, glob
from Datapair import get_dplist_from_dir


data_dir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/split_trainval_from_coverage_split/210313_171723/train'

subdir_list = glob.glob(os.path.join(data_dir, '*'))

subdir_list = [d for d in subdir_list if os.path.isdir(d)]

assert len(subdir_list)> 0

info = {}
total = 0

for d in subdir_list:

    imgdir = os.path.join(d, 'images')
    annotdir = os.path.join(d, 'annots')
    dplist = get_dplist_from_dir(imgdir, annotdir)
    name = os.path.basename(d)

    info[name] = len(dplist)

    total += len(dplist)

print(info)
