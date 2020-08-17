from __future__ import division
import math
import numpy as np
from new_size import new_size
import glob
import cv2
from var_in_img_name import var_in_img_name


def match_template (addrs, win_size, img_height, img_width, template,decision_val_true_lable):
    
    
    filenames=[]
    score = []


    for filename in glob.glob(addrs):
        filenames.append(filename)


    for i in range(len(filenames)):
    #for i in range(100):
        img = cv2.imread(filenames[i])[:,:,1]
        img_name = filenames[i]

        [i_img,j_img,slant_r,b1,b2,H_trap,
         dstlat1,dstlon1,dstlat2,dstlon2,
         dstlat3,dstlon3,dstlat4,dstlon4,gamma,x1,y1,img_label] = var_in_img_name (img_name,10)



        if img_label != 'p':
            
            y1 = img_height/2
            x1 = img_height/2



        yy2 = int( y1-win_size/2) 
        yy1 = int( y1+win_size/2)

        xx2 = int(x1 -win_size/2)
        xx1 = int(x1 +win_size/2)

        if yy2>0 and  yy1<img_height and xx2>0 and xx1<img_width:

            img_crp = img[ yy2 : yy1 , xx2 : xx1]

            img_crop = cv2.resize(img_crp,(win_size,win_size), interpolation = cv2.INTER_AREA)

            res = cv2.matchTemplate(img_crop.astype(np.float32),template.astype(np.float32),cv2.TM_CCOEFF)
            
            if img_label == 'p':
                
                decision_val_true_lable.append([res[0][0],1])
            
            else:
                 
                decision_val_true_lable.append([res[0][0],0])
            
            score.append(res[0][0])
    
    return decision_val_true_lable, score
