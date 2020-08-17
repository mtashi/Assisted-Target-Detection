import cv2
import numpy as np

def save_results (img,dets,addrs,i):
    
    if len(dets) < 5 and len(dets) > 0 :
        for  pic in range(len(dets)):

            if pic ==0:

                cv2.rectangle(img, (int(dets[pic][1]),int(dets[pic][0])),
                                      (int(dets[pic][3]),int(dets[pic][2])),(0,255,0),3)

            else:

                cv2.rectangle(img, (int(dets[pic][1]),int(dets[pic][0])),
                                      (int(dets[pic][3]),int(dets[pic][2])),(255,0,255),3)




        cv2.imwrite(addrs + str(i) + '.jpg', img)

    if len(dets) >= 5 :
        for  pic in range(5):

            if pic ==0:

                cv2.rectangle(img, (int(dets[pic][1]),int(dets[pic][0])),
                                      (int(dets[pic][3]),int(dets[pic][2])),(0,255,0),3)

            else:

                cv2.rectangle(img, (int(dets[pic][1]),int(dets[pic][0])),
                                      (int(dets[pic][3]),int(dets[pic][2])),(255,0,255),3)




        cv2.imwrite(addrs + str(i) + '.jpg', img)
