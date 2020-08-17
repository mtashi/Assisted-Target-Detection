import cv2
import numpy as np
import matplotlib.pyplot as plt
from __future__ import division
import math
from skimage import exposure
import glob
from non_maximum_suppression import nms
from new_size import new_size
from var_in_img_name import var_in_img_name
from img_avg_diff_win import img_avg
from match_template_diff_win import match_template  
from rec_prec_fp_rate import rec_prec_fp_rate
from dec_val_tru_labl import dec_val_tru_labl
from save_results import save_results
from save_txt_patch import save_txt_patch
from save_txt import save_txt




# init
filenames_val=[]
decision_val_true_lable=[]
neg_score_all =[]
pos_score_all =[]

step_size = 5
avg_size=20
img_height = 480
img_width = 640
win_size = 10 #(pixels)


addrs_result = '/media/maryam/Data/training_data_NN_HOG/April21st/diff_win_size/MF/'+str(win_size)+'_0/'



# avg of positive images
addrs = '/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/train/positive/*.jpg'
avg_pos = img_avg (addrs, win_size, img_height, img_width)
    
        
# avg of negative images
addrs = '/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/train/negative/*.jpg'
avg_neg = img_avg (addrs, win_size, img_height, img_width)
    

template = avg_pos- avg_neg


# # inner product of the training set with the template

# decision_val_true_lable_train_patch=[]
# addrs = '/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/train/positive/*.jpg'

# [decision_val_true_lable_train_patch, score_train_pos] = match_template (addrs, win_size, img_height, img_width, template,decision_val_true_lable_train_patch)

# addrs = '/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/train/negative/*.jpg'

# [decision_val_true_lable_train_patch, score_train_neg] = match_template (addrs, win_size, img_height, img_width, template,decision_val_true_lable_train_patch)
 
    
# # inner product of the test set with the template 

# decision_val_true_lable_test_patch=[]
# addrs = '/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/test/positive/*.jpg'

# [decision_val_true_lable_test_patch, score_test_pos] = match_template (addrs, win_size, img_height, img_width, template,decision_val_true_lable_test_patch)

# addrs = '/media/maryam/Data/training_data_NN_HOG/April21st/dataset/whole/test/negative/*.jpg'

# [decision_val_true_lable_test_patch, score_test_neg] = match_template (addrs, win_size,  img_height, img_width, template,decision_val_true_lable_test_patch)


# # PR for patch________________________________________________________________________________________

# [rec_patch, prec_patch, fp_rate_patch, rec_reduced_patch, prec_reduced_patch, fp_rate_reduced_patch] = rec_prec_fp_rate (decision_val_true_lable_test_patch)


# addrs_txt = addrs_result
# save_txt_patch (addrs_txt, rec_patch, rec_reduced_patch, prec_patch, prec_reduced_patch,
#                 fp_rate_patch, fp_rate_reduced_patch)




# import matplotlib.pyplot as plt
# figure = plt.gcf()
# figure.set_size_inches(20, 20)

# plt.plot(rec_patch,prec_patch)
# plt.plot(fp_rate_patch ,rec_patch )



# validation_____________________________________________________________________________________


for filename in glob.glob('/media/maryam/Data/training_data_NN_HOG/April21st/dataset/validation/*.jpg'):
    filenames_val.append(filename)



for i in range(len(filenames_val)):
#for i in range(42,44):

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

                img_crop = img[j_h*step_size:win_size+(j_h)*step_size,j_v*step_size:win_size+(j_v)*step_size]
                
                res = cv2.matchTemplate(img_crop.astype(np.float32),template.astype(np.float32),cv2.TM_CCORR)
                
                # find the distace from the center of the sliding window to the annotation
                dis = math.sqrt((float(x1) - float(av_x))**2 +(float(y1) - float(av_y))**2)

                if dis <= int(avg_size/2):

                    # collecting the TPs and their decision values
                    boxes.append((float(j_h*step_size),float(j_v*step_size),float(win_size+(j_h)*step_size),
                              float(win_size+(j_v)*step_size),res[0][0],1))
                    
                    pos_score_all.append(res[0][0])
                    
#                     if res[0][0]> -760000:
#                         addrs_TP = '/media/maryam/Data/training_data_NN_HOG/April21st/my_way/MF/'+ str(win_size)+'/results/tp/'
#                         cv2.imwrite( addrs_TP + str(res[0][0])+'j_h='+str(j_h)+'j_v='
#                                      +str(j_v)+'.jpg',img_crop)
#                     if res[0][0]<= -760000:
#                         addrs_FN = '/media/maryam/Data/training_data_NN_HOG/April21st/my_way/MF/'+ str(win_size)+'/results/fn/'
#                         cv2.imwrite( addrs_FN + str(res[0][0])+'j_h='+str(j_h)+'j_v='
#                                      +str(j_v)+'.jpg',img_crop)

#                 if dis > int(avg_size/2) and dis < int(win_size/2 * np.sqrt(2)-avg_size/2):

#                     boxes.append((float(j_h*step_size),float(j_v*step_size),float(win_size+(j_h)*step_size),
#                               float(win_size+(j_v)*step_size),res[0][0],1))

                #if dis > int(win_size/2 * np.sqrt(2)-avg_size/2) and  res[0][0]> -760000:
                if dis > int(avg_size/2):
                    
                    boxes.append((float(j_h*step_size),float(j_v*step_size),float(win_size+(j_h)*step_size),
                              float(win_size+(j_v)*step_size),res[0][0],0))
                    
                    neg_score_all.append(res[0][0])
#                     addrs_FP = '/media/maryam/Data/training_data_NN_HOG/April21st/my_way/MF/'+ str(win_size)+'/results/fp/'
#                     cv2.imwrite( addrs_FP + str(res[0][0])+'j_h='+str(j_h)+'j_v='
#                                  +str(j_v)+'.jpg',img_crop)



        # creating the decision value - true label array for PR curve           
        [dets,decision_val_true_lable] = dec_val_tru_labl (boxes,0,decision_val_true_lable)
        
        img_result = cv2.imread(filenames_val[i])
        results_addrs = addrs_result + 'results/results_'
        save_results (img_result,dets, results_addrs ,i)       
                    
   
        
    else:
       
        print('negative')
        
        for j_h in range(int((480-win_size)/step_size)):

            for j_v in range(int(10/step_size),int((640-win_size)/step_size)-int(10/step_size)):

                av_x = j_v*step_size+win_size/2

                av_y = j_h*step_size+win_size/2

                
                img_crop = img[j_h*step_size:win_size+(j_h)*step_size,j_v*step_size:win_size+(j_v)*step_size]
                    
                
                res = cv2.matchTemplate(img_crop.astype(np.float32),template.astype(np.float32),cv2.TM_CCORR)

                #if res[0][0]> -760000:
                neg_score_all.append(res[0][0])
                boxes.append((float(j_h*step_size),float(j_v*step_size),float(win_size+(j_h)*step_size),
                              float(win_size+(j_v)*step_size),res[0][0],0))

# #                         addrs_FP = '/media/maryam/Data/training_data_NN_HOG/April21st/my_way/MF/'+ str(c_win)+'/results/fp/'
# #                         cv2.imwrite( addrs_FP + str(res[0][0])+'j_h='+str(j_h)+'j_v='
# #                                      +str(j_v)+'.jpg',img_crop)

#                 else:

#                     addrs_TN = '/media/maryam/Data/training_data_NN_HOG/April21st/my_way/MF/'+ str(win_size)+'/results/tn/'
#                     cv2.imwrite( addrs_TN + str(res[0][0])+'j_h='+str(j_h)+'j_v='
#                                  +str(j_v)+'.jpg',img_crop)

        # creating the (decision value - true label) array for PR curve           
        [dets,decision_val_true_lable] = dec_val_tru_labl (boxes,0,decision_val_true_lable) 


# PR for images __________________________________________________________________________                

[rec, prec, fp_rate, rec_air, prec_air,fp_rate_air] = rec_prec_fp_rate (decision_val_true_lable)


addrs_txt = addrs_result
save_txt (addrs_txt, rec, rec_air, prec, prec_air, fp_rate, fp_rate_air)


# with open(addrs_txt + 'positive_scores_whole.txt', "a") as text_file:
        
#         for i in range(len(pos_score_all)):

#             text_file.write(str(pos_score_all[i])+'\n')
    
# with open(addrs_txt + 'negaive_scores_whole.txt', "a") as text_file:

#         for i in range(len(neg_score_all)):
            
#             if neg_score_all[i]!='nan':

#                 text_file.write(str(neg_score_all[i])+'\n')


import matplotlib.pyplot as plt
figure = plt.gcf()
figure.set_size_inches(20, 20)


plt.plot(rec_air,prec_air)
plt.plot(fp_rate,rec)
                        

