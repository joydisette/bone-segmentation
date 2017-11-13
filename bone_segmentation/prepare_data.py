#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import cv2
import h5py
import numpy as np
from PIL import Image


def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./xrays/train/images/"
groundTruth_imgs_train = "./xrays/train/mask/"
#test
original_imgs_test = "./xrays/test/images/"
groundTruth_imgs_test = "./xrays/test/mask/"
#---------------------------------------------------------------------------------------------

Nimgs = 11
channels = 3
height = 584
width = 565
dataset_path = "./xrays_datasets_training_testing/"


def get_datasets(imgs_dir,groundTruth_dir,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    #imgs = np.empty((Nimgs,height,width))
    groundTruth = np.empty((Nimgs,height,width,channels))
    #print(groundTruth.dtype)
    #groundTruth=groundTruth.astype(np.float32)
    #groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print "original image: " +files[i]
            img = cv2.imread(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i][0:4] + "_mask.jpg"
            print "ground truth name: " + groundTruth_name
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            print(groundTruth.shape)

    print "imgs max: " +str(np.max(imgs))
    print "imgs min: " +str(np.min(imgs))
    print(np.max(groundTruth))
    assert(np.max(groundTruth)==255)
    assert(np.min(groundTruth)==0)
    print "ground truth is correctly withih pixel value range 0-255 (black-white)"
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,3,height,width))
    def rgb2gray(rgb):
        assert (len(rgb.shape)==4)  #4D arrays
        assert (rgb.shape[1]==3)
        bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
        bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
        return bn_imgs
    groundTruth = rgb2gray(groundTruth)
    print(groundTruth.shape)
    assert(groundTruth.shape == (Nimgs,1,height,width))
    return imgs, groundTruth


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
imgs_train, groundTruth_train = get_datasets(original_imgs_train,groundTruth_imgs_train,"train")
print "saving train datasets"
write_hdf5(imgs_train, dataset_path + "xrays_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "xrays_dataset_groundTruth_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test = get_datasets(original_imgs_test,groundTruth_imgs_test,"test")
print "saving test datasets"
write_hdf5(imgs_test,dataset_path + "xrays_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "xrays_dataset_groundTruth_test.hdf5")
