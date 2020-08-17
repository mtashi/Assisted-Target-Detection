import numpy as np


def save_txt (addrs, rec, rec_reduced, prec, prec_reduced, fp_rate, fp_rate_reduced):
    
    with open(addrs + 'rec.txt', "a") as text_file:

            for i in range(len(rec)):
                text_file.write(str(rec[i])+'\n')   
                
    with open(addrs + 'rec_reduced.txt', "a") as text_file:

            for i in range(len(rec_reduced)):
                text_file.write(str(rec_reduced[i])+'\n')


    with open(addrs + 'prec.txt', "a") as text_file:

            for i in range(len(prec)):
                text_file.write(str(prec[i])+'\n')                 

    with open(addrs + 'prec_reduced.txt', "a") as text_file:

            for i in range(len(prec_reduced)):
                text_file.write(str(prec_reduced[i])+'\n')
                
    with open(addrs + 'fp_rate.txt', "a") as text_file:

            for i in range(len(fp_rate)):
                text_file.write(str(fp_rate[i])+'\n') 
                
    with open(addrs + 'fp_rate_reduced.txt', "a") as text_file:

            for i in range(len(fp_rate_reduced)):
                text_file.write(str(fp_rate_reduced[i])+'\n') 
