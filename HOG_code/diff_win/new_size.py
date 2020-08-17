import math
import numpy as np


def new_size(annotation_x,annotation_y,gamma,slant_range,focal_length,win_buffer, img_height, img_width, pixel_size):


    # pixel_size is the pixel's size in the camera


    theta_h = math.atan((abs(img_width/2-annotation_x)*pixel_size)/focal_length) 
    theta_v = math.atan((abs(img_height/2-annotation_y)*pixel_size)/focal_length) 

    # img_height/2 = 240
    # img_width/2 = 320

    if annotation_y <= img_height/2: 

        phi = gamma - theta_v 
    else:

        phi = gamma + theta_v 

    AB = slant_range * math.sin (gamma)
    AD = AB / math.sin(phi)
    AE = AD / math.cos(theta_h)
    resize_dim = win_buffer/AE

    return resize_dim

