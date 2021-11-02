import numpy as np
import cv2
import os
import glob

directory = './data/'

def readSignleImg(file_path):
    img = cv2.imread(filename=file_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def readDataset(train=True,test=False):
    # train_img = [[],[],[]]
    # test_img = [[],[],[],[]]
    train_img = []
    test_img = []
    if train:
        paths = [file for file in glob.glob("./data/*/1/*.bmp")]
        # for indx in range(3):
        #     for path in paths[indx::3]:
        #         train_img[indx].append(readSignleImg(path))
        for path in paths:
            train_img.append(readSignleImg(path))
    
    if test:
        paths = [file for file in glob.glob("./data/*/2/*.bmp")]
        # for indx in range(4):
        #     for path in paths[indx::3]:
        #         test_img[indx].append(readSignleImg(path))
        for path in paths:
            test_img.append(readSignleImg(path))

    return (train_img, test_img) if (train and test) else (train_img if train else test_img)



if __name__=="__main__":
    train,test = readDataset(True,True)
    print(len(train),len(test))