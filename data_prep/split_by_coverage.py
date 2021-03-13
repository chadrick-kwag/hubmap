import os, sys, cv2, shutil, datetime, numpy as np, json
from Datapair import get_dplist_from_dir
from tqdm import tqdm


timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/split_by_coverage/{timestamp}'

os.makedirs(outputdir)


data_dir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/slice_dataset_v2/210313_153126_train_all'

imgdir = os.path.join(data_dir, 'images')
annotdir = os.path.join(data_dir, 'annots')

dplist = get_dplist_from_dir(imgdir, annotdir)

print(len(dplist))

bins = [0.         ,0.05169578, 0.10339156, 0.15508734 ,0.20678312 ,0.25847891, 1]

bin_name_list = []
bin_range_list = []
bin_count_list = [0] * (len(bins) - 1)


for start, end in zip(bins[:-1], bins[1:]):

    name = f'{start}-{end}'

    bin_name_list.append(name)
    bin_range_list.append((start, end))


bin_savedir_list = []

for name in bin_name_list:
    savedir = os.path.join(outputdir, name)
    os.makedirs(savedir)

    imgdir = os.path.join(savedir, 'images')
    annotdir = os.path.join(savedir, 'annots')

    os.makedirs(imgdir)
    os.makedirs(annotdir)

    bin_savedir_list.append(savedir)


def find_coverage_bin_index(coverage, bin_range_list):

    for i, (s,e) in enumerate(bin_range_list):
        if coverage >= s and coverage <=e:

            return i

    return None


def copy_dp_to_savedir(dp, savedir):

    imgdir = os.path.join(savedir, 'images')
    annotdir = os.path.join(savedir, 'annots')

    bn = os.path.basename(dp.imgpath)
    savepath = os.path.join(imgdir, bn)

    shutil.copy2(dp.imgpath, savepath)

    bn = os.path.basename(dp.annotpath)
    savepath = os.path.join(annotdir, bn)

    shutil.copy2(dp.annotpath, savepath)


for dp in tqdm(dplist):

    mask = cv2.imread(dp.annotpath)

    img_h, img_w, _ = mask.shape


    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = mask > 127

    mask_area = np.sum(mask)

    total = img_h * img_w

    coverage = mask_area / total


    index = find_coverage_bin_index(coverage, bin_range_list)

    if index is None:
        raise Exception(f'no bin for coverage={coverage}, dp={dp.imgpath}')

    bin_count_list[index] +=1

    savedir = bin_savedir_list[index]

    copy_dp_to_savedir(dp, savedir)



savejson = {
    'data_dir': data_dir,
    'bins': bins,
    'dplist_size': len(dplist),
    'bin_count_list': bin_count_list
}


savepath = os.path.join(outputdir, 'info.json')

with open(savepath, 'w') as fd:
    json.dump(savejson, fd, indent=4, ensure_ascii=False)