from __future__ import division
import math
import numpy as np
import glob
import cv2
from new_size import new_size
from var_in_img_name import var_in_img_name


def img_avg (addrs, win_size, focal_length, c_win, img_height, img_width):
    
  
    filenames = []

    for filename in glob.glob(addrs):

        filenames.append(filename)

    img = cv2.imread(filenames[0]) 

    img_crop = cv2.resize(img,(win_size,win_size), interpolation = cv2.INTER_AREA)

    avg =  np.zeros((img_crop.shape[0],img_crop.shape[0]))  


    for i in range(len(filenames)):
    #for i in range(100):

        img = cv2.imread(filenames[i])[:,:,1]

        img_name = filenames[i]

        [i_img,j_img,slant_r,b1,b2,H_trap,
         dstlat1,dstlon1,dstlat2,dstlon2,
         dstlat3,dstlon3,dstlat4,dstlon4,gamma,x1,y1,img_label] = var_in_img_name (img_name,10)
        
        
        if img_label == 'p':
            
            resize_dim = new_size(x1,y1,gamma,slant_r,focal_length, c_win,img_height,img_width,0.02)

        else:
            
            resize_dim = c_win/slant_r
            y1 = img_height /2
            x1 = img_height /2

        
        yy2 = int( y1-resize_dim/2) 
        yy1 = int( y1+resize_dim/2)

        xx2 = int(x1 -resize_dim/2)
        xx1 = int(x1 +resize_dim/2)

        if yy2>0 and  yy1<img_height and xx2>0 and xx1<img_width:

            img_crp = img[ yy2 : yy1 , xx2 : xx1]

            img_crop = cv2.resize(img_crp,(win_size,win_size), interpolation = cv2.INTER_AREA)

            avg = avg + img_crop * 1/len(filenames)

    return avg
