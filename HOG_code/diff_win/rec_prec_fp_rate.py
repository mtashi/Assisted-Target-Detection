from __future__ import division
import numpy as np

def rec_prec_fp_rate (decision_val_true_lable):

    
    # Sorting based on the decision value, from the highest to the lowest
    
    decision_val_true_lable.sort(reverse = True)
    
    my_array = np.asarray(decision_val_true_lable)
    
    # to and fp
    my_tp = np.cumsum( my_array [:,1])
    my_fp = np.cumsum(1 - my_array [:,1])
    
    # rec and prec
    rec = my_tp /sum(my_array[:,1])
    prec = my_tp /(my_tp + my_fp)
    
    # finding the tn by revesring the sorting and flipping the true label
    decision_val_true_lable.sort(reverse = False)
    
    my_array_tn = np.asarray(decision_val_true_lable)
    
    # tn
    my_tn =  np.cumsum(1 - my_array_tn [:,1])
    my_tn = list(my_tn)
    my_tn.sort(reverse = True)
    my_tn = np.asarray(my_tn)

    # false positive rate 
    fp_rate = my_fp/(my_fp + my_tn)

    # eliminating repetitive ones and reducing the size of the array
    index_airplne = np.where(my_array [:,1] == 1)
    
    #reduced- size rec, prec and false positive rate
    rec_reduced = rec [index_airplne]
    prec_reduced = prec [index_airplne]
    fp_rate_reduced = fp_rate [index_airplne]
    
    
    return rec, prec, fp_rate, rec_reduced, prec_reduced, fp_rate_reduced
