import numpy as np
from non_maximum_suppression import nms

# sort based on the second to the last
def sortLast(val): 
    return val[len(val)-2] 

def sortFirst(val): 
    return val[0] 

def dec_val_tru_labl (boxes,ovrlp,decision_val_true_lable):

    boxes.sort(key = sortLast,reverse = True)

    dets = np.array(boxes)

    if dets !=[]:

        keep_indx = nms(dets, ovrlp)

        dets = list(dets[keep_indx])
        
        dets.sort(key = sortLast,reverse = True)

        for detection in range (len(dets)): 

            decision_val_true_lable.append([dets[detection][4],dets[detection][5]])
            
    return dets,decision_val_true_lable
