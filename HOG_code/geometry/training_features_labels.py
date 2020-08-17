import math
import numpy as np
from new_size import new_size
import glob
from skimage import feature
import cv2
from new_size import new_size
from var_in_img_name import var_in_img_name



def training_features_labels(focal_length, default_win_size, HOG_c, HOG_o, HOG_b, win_buffer, img_height, img_width):

    
    # focal length in milimeter
    f = focal_length # (mm)

    # defualt windwo size
    win_size = default_win_size

    # parameters of HOG
    c=HOG_c
    o=HOG_o
    b=HOG_b

    # for negative image patches
    center_point = img_height/2

    # window size buffer
    c_win = win_buffer
    
    

    # to read files
    filenames=[]
    filenames_n=[]

    # training positive images, change it to your own address
    for filename in glob.glob('/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/train/positive/*.jpg'):

        filenames.append(filename)
    

    #using just one channel
    img = cv2.imread(filenames[0])[:,:,1]
    img_crop = cv2.resize(img,(win_size,win_size), interpolation = cv2.INTER_AREA)


    (H, hogImage) = feature.hog(img_crop, orientations=o, pixels_per_cell=(c, c),
        cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
        visualise=True)

    positive_feature_data = np.zeros((len(filenames),len(H))) 
    positive_labels=np.zeros((len(filenames)))


    for i in range(len(filenames)):
    #for i in range(100):
        img = cv2.imread(filenames[i])[:,:,1]

        img_name = filenames[i]

        [i_img,j_img,slant_r,b1,b2,H_trap,
         dstlat1,dstlon1,dstlat2,dstlon2,
         dstlat3,dstlon3,dstlat4,dstlon4,gamma,x1,y1,img_label] = var_in_img_name (img_name,10)

        resize_dim = new_size(x1,y1,gamma,slant_r,f,c_win,img_height,img_width,0.02)

        yy2 = int( y1-resize_dim/2) 
        yy1 = int( y1+resize_dim/2)

        xx2 = int(x1 -resize_dim/2)
        xx1 = int(x1 +resize_dim/2)

        if yy2>0 and  yy1<img_height and xx2>0 and xx1<img_width:

            img = img[ yy2 : yy1 , xx2 : xx1]

            img_crop = cv2.resize(img,(win_size,win_size), interpolation = cv2.INTER_AREA)

            (H, hogImage) = feature.hog(img_crop, orientations=o, pixels_per_cell=(c, c),
            cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
            visualise=True)
            positive_feature_data[i,:]= H
            positive_labels[i]= True


    # negative Images
    for filename in glob.glob('/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/train/negative/*.jpg'):    

        filenames_n.append(filename)

    img = cv2.imread(filenames_n[0])[:,:,1]
    img_crop = cv2.resize(img,(win_size,win_size), interpolation = cv2.INTER_AREA)


    (H, hogImage) = feature.hog(img_crop, orientations=o, pixels_per_cell=(c, c),
        cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
        visualise=True)


    negative_feature_data = np.zeros((len(filenames_n),len(H))) 
    negative_labels=np.zeros((len(filenames_n)))


    for i in range(len(filenames_n)):
    #for i in range(100):
        img = cv2.imread(filenames_n[i])[:,:,1]

        img_name = filenames_n[i]

        [i_img,j_img,slant_r,b1,b2,H_trap,
         dstlat1,dstlon1,dstlat2,dstlon2,
         dstlat3,dstlon3,dstlat4,dstlon4,gamma,x1,y1,img_label] = var_in_img_name (img_name,10)

        resize_dim = c_win/slant_r

        yy2 = int(center_point-resize_dim/2) 
        yy1 =  int(center_point+resize_dim/2)

        xx2 = int(center_point-resize_dim/2)
        xx1 = int(center_point+resize_dim/2)

        if yy2>0 and  yy1<img_height and xx2>0 and xx1<img_width:

            img = img[ yy2 : yy1 , xx2 : xx1]

            img_crop = cv2.resize(img,(win_size,win_size), interpolation = cv2.INTER_AREA)

            (H, hogImage) = feature.hog(img_crop, orientations=o, pixels_per_cell=(c, c),
            cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
            visualise=True)
            negative_feature_data[i,:]= H
            negative_labels[i]= False
            
    return positive_feature_data, negative_feature_data, positive_labels, negative_labels

