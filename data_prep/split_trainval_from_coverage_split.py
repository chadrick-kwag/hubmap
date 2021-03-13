import os, json, shutil, random, glob, datetime
from Datapair import get_dplist_from_dir
from tqdm import tqdm

random.seed()


coverage_split_datadir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/split_by_coverage/210313_165246_train_split'


subdir_list = glob.glob(os.path.join(coverage_split_datadir, '*'))

for d in subdir_list:
    assert os.path.isdir(d), f'{d} is not dir'


valid_count = 100

subdir_count = len(subdir_list)

per_dir_sample_size = valid_count // subdir_count

assert per_dir_sample_size > 0



timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/split_trainval_from_coverage_split/{timestamp}'

os.makedirs(outputdir)


dirname_list = []

for d in subdir_list:
    bn = os.path.basename(d)
    dirname_list.append(bn)


save_train_dir = os.path.join(outputdir, 'train')
save_valid_dir = os.path.join(outputdir, 'valid')

os.makedirs(save_train_dir)
os.makedirs(save_valid_dir)


save_train_subdir_list =[]
for d in dirname_list:
    path = os.path.join(save_train_dir, d)
    os.makedirs(path)

    imgdir = os.path.join(path, 'images')
    os.makedirs(imgdir)

    annotdir = os.path.join(path, 'annots')
    os.makedirs(annotdir)

    save_train_subdir_list.append(path)

# setup valid dir

imgdir = os.path.join(save_valid_dir, 'images')
os.makedirs(imgdir)

annotdir = os.path.join(save_valid_dir, 'annots')
os.makedirs(annotdir)

def copy_dp_to_savedir(dp, savedir):

    imgdir = os.path.join(savedir, 'images')
    annotdir = os.path.join(savedir, 'annots')

    bn = os.path.basename(dp.imgpath)
    savepath = os.path.join(imgdir, bn)

    shutil.copy2(dp.imgpath, savepath)

    bn = os.path.basename(dp.annotpath)
    savepath = os.path.join(annotdir, bn)

    shutil.copy2(dp.annotpath, savepath)

for subdir_index, d in enumerate(subdir_list):

    print(f'splitting {d}, {subdir_index+1}/{len(subdir_list)}')

    imgdir = os.path.join(d, 'images')
    annotdir = os.path.join(d, 'annots')
    dplist = get_dplist_from_dir(imgdir, annotdir)

    sampled_index = random.sample(range(len(dplist)), k=per_dir_sample_size)


    in_valid_bool_list = [False] * len(dplist)
    for i in sampled_index:
        in_valid_bool_list[i] = True

    progress = tqdm(total=len(dplist))
    for in_valid, dp in tqdm(zip(in_valid_bool_list, dplist)):

        if in_valid:
            savedir = save_valid_dir
        else:
            savedir = save_train_subdir_list[subdir_index]
        
        copy_dp_to_savedir(dp, savedir)

        progress.update(1)


savejson = {
    'coverage_split_datadir': coverage_split_datadir,
    'valid_count': valid_count,
    'per_dir_sample_size': per_dir_sample_size

}