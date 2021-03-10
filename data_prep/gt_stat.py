import os, sys, cv2, json, matplotlib.pyplot as plt, numpy as np, datetime

from Datapair import get_dplist_from_dir

datadir = '/home/chadrick/prj/kaggle/hubmap/teststuff/testoutput/crop_targets/210310_000942'


imgdir = os.path.join(datadir, 'images')
annotdir = os.path.join(datadir, 'annots')
dplist = get_dplist_from_dir(imgdir, annotdir)


print(len(dplist))


imgshape_list = []

for dp in dplist:
    img = cv2.imread(dp.imgpath)

    img_h, img_w, _ = img.shape

    imgshape_list.append([img_w, img_h])


figure = plt.figure()

img_w_arr = np.array([a[0] for a in imgshape_list])
img_h_arr = np.array([a[1] for a in imgshape_list])


min_w = np.min(img_w_arr)
max_w = np.max(img_w_arr)

min_h = np.min(img_h_arr)
max_h = np.max(img_h_arr)


print(f'min_w={min_w}, max_w={max_w}, min_h={min_h}, max_h={max_h}')

w_bin = np.arange(min_w - 10, max_w+20, 10)

print(w_bin)

h_bin = np.arange(min_h - 10, max_h+20, 10)


plt.hist2d(img_w_arr, img_h_arr)
plt.colorbar()

timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
outputdir = f'testoutput/gt_stat/{timestamp}'

os.makedirs(outputdir)

print(f'outputdir: {outputdir}')


savepath = os.path.join(outputdir, 'test.png')

figure.savefig(savepath)


