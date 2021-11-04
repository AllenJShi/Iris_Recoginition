import glob
import cv2
import numpy as np
from IrisLocalization import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *


directory = './data/'

def readSignleImg(file_path):
    """take file path and read the image in the path

    Args:
        file_path (string): input image location

    Returns:
        image matrix: the image matrix stored in the path
    """
    img = cv2.imread(filename=file_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def readDataset(train=True,test=False):
    """read all images in the dataset

    Args:
        train (bool, optional): true if ask to extract training set. Defaults to True.
        test (bool, optional): true if ask to extract testing set. Defaults to False.

    Returns:
        tuple or array: return the train and/or test data based on request
    """
    train_img = []
    test_img = []
    if train:
        paths = [file for file in glob.glob("./data/*/1/*.bmp")]
        # loop through the path list and read all images into the container
        for path in paths:
            train_img.append(readSignleImg(path))
    
    if test:
        paths = [file for file in glob.glob("./data/*/2/*.bmp")]
        # loop through the path list and read all images into the container
        for path in paths:
            test_img.append(readSignleImg(path))

    return (train_img, test_img) if (train and test) else (train_img if train else test_img)


def rotateImg(img,offset):
    """rotate the image by the offset degree

    Args:
        img (array): input image to be rotated
        offset (int): degree by which to rotate the image

    Returns:
        array: rotated image
    """
    pixels = abs(int(512*offset/360))
    if offset > 0:
        return np.hstack([img[:,pixels:],img[:,:pixels]] )
    else:
        return np.hstack([img[:,(512 - pixels):],img[:,:(512 - pixels)]])



def saveNormalizedImage():
    """help to read all train images and normalize and enhance the images
        save all images into npy file
    """
    train = np.load("train.npy")
    norm = []
    # preprocess each image in the train dataset
    for i,img in enumerate(train):
        inner_circle, outer_circle = irisLocalization(img)
        # img = denoising(img)
        img_norm = irisNormalization(img,inner_circle,outer_circle)
        img_enhance = imageEnhancement(img_norm)
        norm.append(img_enhance)
        print(f"process class {int(i/3)}: {i}/{len(train)}")
    np.save("train_norm",norm)


def rotateAll():
    """rotate all train images by offsets, save to npy file
    """
    norm = np.load("train_norm.npy")
    offsets = [-9,-6,-3,0,3,6,9]
    res = []
    # rotate each image by each offset angle
    for i, img in enumerate(norm):
        for offset in offsets:
            p = rotateImg(img,offset)
            res.append(p)
        print(f"process class {int(i/3)}:{i}/{len(norm)}")
    np.save("train_norm_rotate.npy",res)
    
    
def extractFeatureFromRotatedImg():
    """extract features from rotated images and save to npy
    """
    imgs = np.load("train_norm_rotate.npy")
    V = []
    for i,img in enumerate(imgs):
        feature_vector = featureExtraction(img)
        V.append([feature_vector])
        print(f"process class {int(i/21)}:{i}/{len(imgs)}")
    np.save("X_train_rotate",V)

# ===============================================================
# denoising decreased the accuracy 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def denoisingAll():
#     imgs = np.load("train.npy")
#     denoise = []
#     for i,img in enumerate(imgs):
#         (_, B) = cv2.threshold(img ,180 ,255 ,cv2.THRESH_BINARY)
#         (_, C) = cv2.threshold(img ,100 ,255 ,cv2.THRESH_BINARY)
#         img = img & ~B & C
#         denoise.append(img)
#         print(f"process class {int(i/21)}:{i}/{len(imgs)}")
#     np.save("train_denoise",denoise)
    
# def denoising(img):
#     (_, B) = cv2.threshold(img ,180 ,255 ,cv2.THRESH_BINARY)
#     (_, C) = cv2.threshold(img ,100 ,255 ,cv2.THRESH_BINARY)
#     img = img & ~B & C
#     return img

if __name__=="__main__":
    
    saveNormalizedImage()
    rotateAll()
    extractFeatureFromRotatedImg()
