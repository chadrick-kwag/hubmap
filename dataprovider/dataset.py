import torch, os, cv2, numpy as np
from data_prep.Datapair import get_dplist_from_dir


class Dataset(torch.utils.data.Dataset):

    def __init__(self, datadir):

        self.datadir = datadir

        imgdir = os.path.join(datadir, 'images')
        annotdir = os.path.join(datadir, 'annots')

        assert os.path.exists(imgdir)
        assert os.path.exists(annotdir)

        self.dplist = get_dplist_from_dir(imgdir, annotdir)

        assert len(self.dplist) > 0

    def __len__(self):
        return len(self.dplist)

    def __getitem__(self, idx):

        dp = self.dplist[idx]

        img = cv2.imread(dp.imgpath)
        mask = cv2.imread(dp.annotpath)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.expand_dims(mask, -1)

        mask = mask / 255
        mask = mask > 0.5
        mask = mask.astype(float)

        # normalize img by /255

        img = img/255

        return img, mask