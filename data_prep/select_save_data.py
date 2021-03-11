import os, cv2, numpy as np, random,datetime, shutil
from tqdm import tqdm

random.seed()

from Datapair import get_dplist_from_dir


data_dir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/slice_dataset/210310_210941'

imgdir = os.path.join(data_dir, 'images')
annotdir = os.path.join(data_dir, 'annots')

dplist = get_dplist_from_dir(imgdir, annotdir)

print(len(dplist))

no_mask_dp_list =[]

mask_coveraged_satisfied_dp_list =[]


mask_coverage_min = 0.05


for dp in tqdm(dplist):


    img = cv2.imread(dp.imgpath)
    mask = cv2.imread(dp.annotpath)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask > 0

    mask_sum = np.sum(mask)

    if mask_sum == 0:
        no_mask_dp_list.append(dp)
        continue

    img_h, img_w = mask.shape

    total_area = img_w * img_h

    mask_coverage = mask_sum / total_area

    print(f'mask_coverage: {mask_coverage}')

    if mask_coverage > mask_coverage_min:
        mask_coveraged_satisfied_dp_list.append(dp)



print(f'no mask dp list size: {len(no_mask_dp_list)}')
print(f'mask_coveraged_satisfied_dp_list size: {len(mask_coveraged_satisfied_dp_list)}')




timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/select_save_data/{timestamp}'

os.makedirs(outputdir)

save_no_mask = os.path.join(outputdir, 'no_mask')
save_mcs = os.path.join(outputdir, 'mask_coverage_satisfied')

os.makedirs(save_no_mask)
os.makedirs(save_mcs)


def save_dp_list(savedir, dp_list):

    imgdir = os.path.join(savedir, 'images')
    annotdir = os.path.join(savedir, 'annots')

    os.makedirs(imgdir)
    os.makedirs(annotdir)

    for dp in dp_list:

        bn = os.path.basename(dp.imgpath)
        savepath = os.path.join(imgdir, bn)

        shutil.copy2(dp.imgpath, savepath)

        bn = os.path.basename(dp.annotpath)
        savepath = os.path.join(annotdir, bn)

        shutil.copy2(dp.annotpath, savepath)
    

save_dp_list(save_no_mask, no_mask_dp_list)
save_dp_list(save_mcs, mask_coveraged_satisfied_dp_list)

