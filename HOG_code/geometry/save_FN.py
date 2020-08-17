import cv2
import numpy as np
from skimage import exposure
from skimage import feature
from skimage.feature import hog


def save_FN (addrs,win_size, response, decision_value,resize_dim,j_h,j_v, img, img_crop ,step_size, c,o,b):

    if response == 0:

        im_orig = img[j_h*step_size:win_size+(j_h)*step_size,j_v*step_size:win_size+(j_v)*step_size]
        cv2.imwrite(addrs +str(decision_value)+'j_h='+str(j_h)+'j_v='
                    +str(j_v)+'resize_dim='+str(resize_dim)+'_orig.jpg',im_orig)


        cv2.imwrite( addrs + str(decision_value)+'j_h='+str(j_h)+'j_v='
                    +str(j_v)+'resize_dim='+str(resize_dim)+'.jpg',img_crop)
        

        (H, hogImage) = feature.hog(img_crop, orientations=o, pixels_per_cell=(c, c),
                    cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
                    visualise=True)

        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")

        #cv2.imwrite(addrs + str(decision_value)+'j_h='+str(j_h)+'j_v='
        #            +str(j_v)+'resize_dim='+str(resize_dim)+'.jpg',hogImage)


        (H_orig, hogImage_orig) = feature.hog(im_orig, orientations=o, pixels_per_cell=(c, c),
            cells_per_block=(b, b), transform_sqrt=True, block_norm="L1",
            visualise=True)

        hogImage_orig = exposure.rescale_intensity(hogImage_orig, out_range=(0, 255))
        hogImage_orig = hogImage_orig.astype("uint8")

        #cv2.imwrite(addrs  + str(decision_value)+'j_h='+str(j_h)+'j_v='
        #            +str(j_v)+'resize_dim='+str(resize_dim)+'_orig.jpg',hogImage_orig)
