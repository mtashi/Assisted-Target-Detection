def var_in_img_name (img_name,num):

    
    i_img = int(img_name.split('/')[num].split('=')[1].split('_')[0])
    j_img = int(img_name.split('/')[num].split('=')[2].split('_')[0])
    slant_r = float(img_name.split('/')[num].split('=')[3].split('_')[0])
    b1 = float(img_name.split('/')[num].split('=')[4].split('_')[0])
    b2 = float(img_name.split('/')[num].split('=')[5].split('_')[0])
    H_trap = float(img_name.split('/')[num].split('=')[6].split('_')[0])
    dstlat1 = float(img_name.split('/')[num].split('=')[7].split('_')[0])
    dstlon1 = float(img_name.split('/')[num].split('=')[8].split('_')[0])
    dstlat2 = float(img_name.split('/')[num].split('=')[9].split('_')[0])
    dstlon2 = float(img_name.split('/')[num].split('=')[10].split('_')[0])
    dstlat3 = float(img_name.split('/')[num].split('=')[11].split('_')[0])
    dstlon3 = float(img_name.split('/')[num].split('=')[12].split('_')[0])
    dstlat4 = float(img_name.split('/')[num].split('=')[13].split('_')[0])    
    dstlon4 = float(img_name.split('/')[num].split('=')[14].split('_')[0]) 
    
    if img_name.split('/')[num].split('_')[1][0] == 'p':
        
        img_label = 'p'
        gamma = float(img_name.split('/')[num].split('=')[15].split('_')[0])   
        x1 = int((img_name.split('/')[num].split('=')[16].split('y1')[0]))
        y1 = int((img_name.split('/')[num].split('=')[17].split('.jpg')[0]))
                                
    if img_name.split('/')[num].split('_')[1][0] == 'n':
        
        img_label = 'n'
        gamma = float(img_name.split('/')[num].split('=')[15].split('.jpg')[0])
        x1 ='NAN'
        y1 ='NAN' 
        
    return [i_img,j_img,slant_r,b1,b2,H_trap,
            dstlat1,dstlon1,dstlat2,dstlon2,
            dstlat3,dstlon3,dstlat4,dstlon4,gamma,x1,y1,img_label]

