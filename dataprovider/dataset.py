import torch, os, cv2, numpy as np, random
from data_prep.Datapair import get_dplist_from_dir

random.seed()


class Dataset(torch.utils.data.Dataset):

    def __init__(self, datadir, resize):

        self.datadir = datadir
        resize_w, resize_h = resize
        self.resize_w = resize_w
        self.resize_h = resize_h

        imgdir = os.path.join(datadir, 'images')
        annotdir = os.path.join(datadir, 'annots')

        assert os.path.exists(imgdir), f'{imgdir}'
        assert os.path.exists(annotdir), f'{annotdir}'

        self.dplist = get_dplist_from_dir(imgdir, annotdir)

        assert len(self.dplist) > 0



    def __len__(self):
        return len(self.dplist)

    def __iter__(self):
        self.index_list = list(range(len(self.dplist)))
        random.shuffle(self.index_list)
        self.num = 0

        return self

    def __next__(self):

        if self.num >= len(self.dplist):

            self.index_list = list(range(len(self.dplist)))
            random.shuffle(self.index_list)
            self.num=0
            
        

        index = self.index_list[self.num]

        self.num+=1

        # result = self[index]

        dp = self.dplist[index]

        return self.getdata(dp)
        
    def getdata(self, dp):


        img = cv2.imread(dp.imgpath)
        mask = cv2.imread(dp.annotpath)

        img = cv2.resize(img, (self.resize_w, self.resize_h))
        mask = cv2.resize(mask, (self.resize_w, self.resize_h))

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.expand_dims(mask, -1)

        mask = mask.astype(float) / 255
        mask = mask > 0.5
        

        # normalize img by /255
        img = img.astype(float)
        img = img/255

        return img, mask


    def __getitem__(self, idx):

        dp = self.dplist[idx]

        return self.getdata(dp)


class PredictDataset(torch.utils.data.Dataset):


    def __init__(self, imglist, resize):

        self.imglist = imglist
        resize_w, resize_h = resize
        self.resize_w = resize_w
        self.resize_h = resize_h


        assert len(self.imglist) > 0


    def __len__(self):
        return len(self.imglist)

    
    def getdata(self, imgpath):


        img = cv2.imread(imgpath)

        img = cv2.resize(img, (self.resize_w, self.resize_h))

        img = img.astype(float)
        img = img/255

        return img


    def __getitem__(self, idx):

        imgpath = self.imglist[idx]

        return self.getdata(imgpath), imgpath