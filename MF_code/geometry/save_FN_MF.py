import cv2
import numpy as np
from skimage import exposure
from skimage import feature
from skimage.feature import hog


def save_FN (addrs,win_size, const, res, resize_dim,j_h,j_v, img_crop ,step_size):

    if res  > -760000:

        res = cv2.matchTemplate(img_crop.astype(np.float32),template.astype(np.float32),cv2.TM_CCORR)

        cv2.imwrite(addrs  + str(decision_value)+'j_h='+str(j_h)+'j_v='
                    +str(j_v)+'resize_dim='+str(resize_dim)+'_orig.jpg',hogImage_orig)
