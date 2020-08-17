
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from __future__ import division
import math
from skimage import exposure
from skimage import feature
from skimage.feature import hog
from sklearn.svm import SVC
import glob
from non_maximum_suppression import nms
from new_size import new_size
from var_in_img_name import var_in_img_name
from training_features_labels import training_features_labels
from test_features_labels import test_features_labels
from rec_prec_fp_rate import rec_prec_fp_rate
from save_TP import save_TP
from save_FN import save_FN
from dec_val_tru_labl import dec_val_tru_labl
from save_results import save_results
from save_txt_patch import save_txt_patch
from save_txt import save_txt
import time




# init
filenames_val=[]
decision_val_true_lable=[]


step_size = 5
avg_size=20
img_height = 480
img_width = 640
focal_length = 40 #(mm)
win_size = 40 #(pixels)
c = 10
o = 7
b = 2
c_win = 34000



addrs_result = '/media/maryam/Data/training_data_NN_HOG/April21st/my_way/HOG/'+str(c_win)+'_all/'

start_time = time.time()
# train
[positive_feature_data, negative_feature_data,
 positive_labels, negative_labels] = training_features_labels(focal_length, win_size, c, o,
                                                              b, c_win, 480, 640)


#SVM
clf = SVC(C=1.0, kernel='linear', degree=3, coef0=0.0, shrinking=True,
           probability=True,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
           decision_function_shape='ovo', random_state=None)


X_train=np.concatenate((positive_feature_data, negative_feature_data))
y_train = np.concatenate((positive_labels,negative_labels))


trained_SVM = clf.fit(X_train, y_train) 

# test
[Test_Data_Air, Test_Data_noAir, Test_labels_Air,
 Test_labels_noAir, decision_patch_val_true_lable_patch] = test_features_labels(trained_SVM,
                                                        focal_length, win_size, c, o, b,
                                                                               c_win, 480, 640)

print("--- %s seconds ---" % (time.time() - start_time))
# accuracy

diff = 0

test_data_all=np.concatenate((Test_Data_Air, Test_Data_noAir))    
test_labels_all= np.concatenate((Test_labels_Air,Test_labels_noAir))    


for k in range(test_data_all.shape[0]):
#     H = np.reshape(test_data_all[k],(1,H.size))
    response = trained_SVM.predict([test_data_all[k]])
    #print('response:  ',response,'    label:  ',test_labels_all[k])
    diff= abs(response[0]-test_labels_all[k])+diff

accuracy= 100-(diff/len(test_data_all))*100    
print accuracy



# # PR for patch________________________________________________________________________________________

# [rec_patch, prec_patch, fp_rate_patch, rec_reduced_patch, prec_reduced_patch, fp_rate_reduced_patch] = rec_prec_fp_rate (decision_patch_val_true_lable_patch)


# addrs_txt = '/media/maryam/Data/training_data_NN_HOG/April21st/my_way/HOG/'+str(c_win)+'_3/'
# save_txt_patch (addrs_txt, rec_patch, rec_reduced_patch, prec_patch, prec_reduced_patch,
#                 fp_rate_patch, fp_rate_reduced_patch)




# import matplotlib.pyplot as plt
# figure = plt.gcf()
# figure.set_size_inches(20, 20)

# plt.plot(rec_patch,prec_patch)
# plt.plot(fp_rate_patch ,rec_patch )


start_time = time.time()

# validation_____________________________________________________________________________________


for filename in glob.glob('/media/maryam/Data/training_data_NN_HOG/April21st/dataset/validation_1/*.jpg'):
    filenames_val.append(filename)



for i in range(len(filenames_val)):
#for i in range(43,44):

    print('i',i)

    boxes =[]
    
    img = cv2.imread(filenames_val[i])[:,:,1]
    img_name = filenames_val[i]
    [i_img,j_img,slant_r,b1,b2,H_trap,
     dstlat1,dstlon1,dstlat2,dstlon2,
     dstlat3,dstlon3,dstlat4,dstlon4,gamma,x1,y1,img_label] = var_in_img_name (img_name,8)
    
    # if the image is a positive image
    
    if img_label == 'p':
        
        print('positive')

        for j_h in range(int((img_height-win_size)/step_size)):
            # (10/step_size) is for the 10 black pixels on the sides
            for j_v in range(int(10/step_size),int((img_width-win_size)/step_size)-int(10/step_size)):
                
                
                # check the sliding window position
                #img = cv2.circle(img,(win_size+(j_v)*step_size,win_size+(j_h)*step_size), 3, (0,255,0), -1)
                #cv2.imwrite('/media/maryam/Data/training_data_NN_HOG/HOG_2019/HoG_cropped/crop_j_h='+str(j_h)+'j_v='+str(j_v)+'.jpg',img_crop )

                
                av_x = j_v*step_size + win_size/2

                av_y = j_h*step_size + win_size/2

                #rescale and eliminate the the different sizing factor
                resize_dim = new_size(x1, y1, gamma, slant_r, focal_length, c_win, img_height, img_width,0.02)

                yy2 = int( av_y-resize_dim/2) 
                yy1 = int( av_y+resize_dim/2)

                xx2 = int(av_x -resize_dim/2)
                xx1 = int(av_x +resize_dim/2)

                if yy2>0 and  yy1<img_height and xx2>0 and xx1<img_width and resize_dim > 0:

                    img_crp = img[ yy2 : yy1 , xx2 : xx1]
                    img_crop = cv2.resize(img_crp,(win_size,win_size), interpolation = cv2.INTER_AREA)
                   
                    (H, hogImage) = feature.hog(img_crop, orientations=o, pixels_per_cell=(c, c),
                    cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
                    visualise=True)

                    # out put of SVM
                    response = trained_SVM.predict([H])
                    decision_value =trained_SVM.decision_function([H])
                    prob = trained_SVM.predict_proba([H])[0]
                    
                
                    # find the distace from the center of the sliding window to the annotation
                    dis = math.sqrt((float(x1) - float(avg_xg))**2 +(float(y1) - float(avg_yg))**2)


                    if dis <= int(avg_size/2):
                    
                        # collecting the TPs and their decision values
                        boxes.append((float(j_h*step_size),float(j_v*step_size),float(win_size+(j_h)*step_size),
                                  float(win_size+(j_v)*step_size),decision_value[0],1))
                        
                    
#                         # saving TP and FN
#                         addrs_TP = addrs_result + 'results/tp/'
#                         save_TP (addrs_TP,win_size, response[0], decision_value[0],resize_dim,j_h,j_v,
#                                  img, img_crop ,step_size, c,o,b)

#                         addrs_FN = addrs_result+ 'results/fn/'
#                         save_FN (addrs_FN,win_size, response[0], decision_value[0],resize_dim,j_h,j_v,
#                                  img, img_crop ,step_size, c,o,b)


#                     if dis > int(avg_size/2) and dis < int(win_size/2 * np.sqrt(2)-avg_size/2):
                    
#                         boxes.append((float(j_h*step_size),float(j_v*step_size),float(win_size+(j_h)*step_size),
#                                   float(win_size+(j_v)*step_size),decision_value[0],1))
                        
                    #if dis > int(win_size/2 * np.sqrt(2)-avg_size/2):# and  response == 1:
                    if dis > int(avg_size/2):
                
                        boxes.append((float(j_h*step_size),float(j_v*step_size),float(win_size+(j_h)*step_size),
                                  float(win_size+(j_v)*step_size),decision_value[0],0))
                        
#                         addrs_FP = addrs_result + 'results/fp/'
#                         cv2.imwrite( addrs_FP + str(decision_value[0])+'j_h='+str(j_h)+'j_v='
#                                      +str(j_v)+'resize_dim='+str(resize_dim)+'.jpg',hogImage)


                        
                        
                        
                    #cv2.rectangle(img,(j_h*step_size,j_v*step_size),(win_size+(j_h)*step_size,win_size+(j_v)*step_size),(0,255,0),3)
                    
        
        # creating the decision value - true label array for PR curve           
        [dets,decision_val_true_lable] = dec_val_tru_labl (boxes,0,decision_val_true_lable)
        
#         img_result = cv2.imread(filenames_val[i])
#         results_addrs = addrs_result +'results/results_'
#         save_results (img_result,dets, results_addrs ,i)       
                    

    else:
        
        print('negative')
        
        for j_h in range(int((480-win_size)/step_size)):

            for j_v in range(int(10/step_size),int((640-win_size)/step_size)-int(10/step_size)):

                av_x = j_v*step_size+win_size/2

                av_y = j_h*step_size+win_size/2

                #rescale and eliminate the the different sizing factor
                resize_dim = new_size(av_x, av_y , gamma, slant_r, focal_length, c_win, img_height, img_width,0.02)

                yy2 = int( av_y-resize_dim/2) 
                yy1 = int( av_y+resize_dim/2)

                xx2 = int(av_x -resize_dim/2)
                xx1 = int(av_x +resize_dim/2)

                if yy2>0 and  yy1<img_height and xx2>0 and xx1<img_width and resize_dim > 0:

                    img_crp = img[ yy2 : yy1 , xx2 : xx1]
                    img_crop = cv2.resize(img_crp,(win_size,win_size), interpolation = cv2.INTER_AREA)
                   
                    (H, hogImage) = feature.hog(img_crop, orientations=o, pixels_per_cell=(c, c),
                    cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
                    visualise=True)

                    # out put of SVM
                    response = trained_SVM.predict([H])
                    decision_value =trained_SVM.decision_function([H])
                    prob = trained_SVM.predict_proba([H])[0]
                    
                    
                    boxes.append((float(j_h*step_size),float(j_v*step_size),float(win_size+(j_h)*step_size),
                              float(win_size+(j_v)*step_size),decision_value[0],0))

#                     addrs_FP = '/media/maryam/Data/training_data_NN_HOG/April21st/my_way/HOG/'+ str(c_win)+'_3/results/fp/'
#                     cv2.imwrite( addrs_FP + str(decision_value)+'j_h='+str(j_h)+'j_v='
#                                  +str(j_v)+'resize_dim='+str(resize_dim)+'.jpg',hogImage)

#                     addrs_TN = '/media/maryam/Data/training_data_NN_HOG/April21st/my_way/HOG/'+ str(c_win)+'_3/results/tn/'
#                     cv2.imwrite( addrs_TN + str(decision_value)+'j_h='+str(j_h)+'j_v='
#                                  +str(j_v)+'resize_dim='+str(resize_dim)+'.jpg',hogImage)
                        
        # creating the (decision value - true label) array for PR curve           
        [dets,decision_val_true_lable] = dec_val_tru_labl (boxes,0,decision_val_true_lable) 

            


# PR for images __________________________________________________________________________                

[rec, prec, fp_rate, rec_reduced, prec_reduced,fp_rate_reduced] = rec_prec_fp_rate (decision_val_true_lable)

print("--- %s seconds ---" % (time.time() - start_time))
addrs_txt = addrs_result
save_txt (addrs_txt, rec, rec_reduced, prec, prec_reduced, fp_rate, fp_rate_reduced)

import matplotlib.pyplot as plt
figure = plt.gcf()
figure.set_size_inches(20, 20)


plt.plot(rec_reduced,prec_reduced)
plt.plot(fp_rate,rec)
                                                                                                                                    

