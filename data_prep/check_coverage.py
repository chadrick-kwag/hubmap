import os, cv2, numpy as np, random,datetime, shutil, json
from tqdm import tqdm

random.seed()

from Datapair import get_dplist_from_dir


data_dir = '/home/chadrick/prj/kaggle/hubmap/data_prep/testoutput/slice_dataset_v2/210313_153126_train_all'

imgdir = os.path.join(data_dir, 'images')
annotdir = os.path.join(data_dir, 'annots')

dplist = get_dplist_from_dir(imgdir, annotdir)

coverage_list = []




timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/check_coverage/{timestamp}'

os.makedirs(outputdir)


for dp in tqdm(dplist):

    img = cv2.imread(dp.imgpath)

    img_h, img_w, _ = img.shape

    mask = cv2.imread(dp.annotpath)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = mask > 127

    mask_count = np.sum(mask)

    total_count = img_h * img_w

    coverage = mask_count / total_count

    coverage_list.append(coverage)


total_count = len(dplist)


hist, bins = np.histogram(np.array(coverage_list), bins=10)

print(hist)
print(bins)

hist_ratio = hist / total_count

print(hist_ratio)


savejson = {
    'data_dir': data_dir,
    'total_count': total_count,
    'hist': hist.tolist(),
    'bins': bins.tolist(),
    'hist_ratio': hist_ratio.tolist()
}

savepath = os.path.join(outputdir, 'info.json')

with open(savepath, 'w') as fd:
    json.dump(savejson, fd, indent=4, ensure_ascii=False)