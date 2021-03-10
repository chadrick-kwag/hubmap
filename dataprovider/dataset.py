import torch, os, cv2
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

        return img, mask