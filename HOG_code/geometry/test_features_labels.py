import math
import numpy as np
from new_size import new_size
import glob
from skimage import feature
import cv2
from new_size import new_size
from var_in_img_name import var_in_img_name



def test_features_labels(trained_SVM, focal_length, default_win_size, HOG_c, HOG_o, HOG_b, win_buffer, img_height, img_width):
    
    
   

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
    

    # for pr curve 
    decision_patch_val_true_lable_patch =[]
    

    # to read files
    filenames=[]
    filenames_n=[]


    # test images
    for filename in glob.glob('/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/test/positive/*.jpg'):

        filenames.append(filename)

    img = cv2.imread(filenames[0])[:,:,1]


    img_crop = cv2.resize(img,(win_size,win_size), interpolation = cv2.INTER_AREA)


    (H, hogImage) = feature.hog(img_crop, orientations=o, pixels_per_cell=(c, c),
        cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
        visualise=True)

    Test_Data_Air = np.zeros((len(filenames),len(H))) 
    Test_labels_Air = np.zeros((len(filenames)))


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
            Test_Data_Air[i,:]= H
            Test_labels_Air[i]= True
            prob = trained_SVM.predict_proba([H])[0]
            decision_value =trained_SVM.decision_function([H])
            response = trained_SVM.predict([H])

            decision_patch_val_true_lable_patch.append([decision_value[0],1])


    for filename in glob.glob('/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/test/negative/*.jpg'):

        filenames_n.append(filename)

    img = cv2.imread(filenames_n[0])[:,:,1]

    img_crop = cv2.resize(img,(win_size,win_size), interpolation = cv2.INTER_AREA)

    (H, hogImage) = feature.hog(img_crop, orientations=o, pixels_per_cell=(c, c),
        cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
        visualise=True)

    Test_Data_noAir = np.zeros((len(filenames_n),len(H))) 
    Test_labels_noAir = np.zeros((len(filenames_n)))


    for i in range(len(filenames_n)):
    #for i in range(100): 
        
        img = cv2.imread(filenames_n[i])[:,:,1]

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
            Test_Data_noAir[i,:] = H
            Test_labels_noAir[i] = False
            prob = trained_SVM.predict_proba([H])[0]
            decision_value =trained_SVM.decision_function([H])
            response = trained_SVM.predict([H])
            #text_file.write(str([prob[0],prob[1],decision_value[0],response[0]])+'\n')

            decision_patch_val_true_lable_patch.append([decision_value[0],0])


    return Test_Data_Air, Test_Data_noAir, Test_labels_Air, Test_labels_noAir,decision_patch_val_true_lable_patch

