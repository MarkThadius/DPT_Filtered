"""
Created on Fri Jul 19 16:40:10 2019
@author: Mark
"""
import rmpavage.process_image
import rmpavage.df_to_excel
import numpy as np
import glob, os
import pandas as pd

import importlib
importlib.reload(rmpavage.process_image)

#pick and arbitrary image of the same shape and size of the test images:
url = r"Test Images/BSDS500/BSDS500/data/images/train/Just30" #set the URL to the testing folder
img = url + r"\8049.jpg"#import the image,slash is for url to work

#init the pi object with an image of the shape and size we want:
pi = rmpavage.process_image.ProcessImage(img)
#LARGEST EIGENVALUE & CENTREPOINT:
pi.defineGraphAttributes(refmat = 'NL',Lmax = 1.9999999998,Lmin = 0)
pi.createDualFilters(w=1,m=30)
pi.filterHighLow(show=False)
pi.createSubGraphs()
tile_size = 7

#create datafram to store results:
#add lots of CPU times:
columns = ['Image Label','Ht index','Ht index Low', 'Ht index High','CPU-Time DPT','CPU-Time DPT Low','CPU-Time DPT High',
           'CPU-Time Recon F1','CPU-Time Recon F2','CPU-Time Recon FF' ,'CPU-Time Recon Low F1','CPU-Time Recon Low F2','CPU-Time Recon Low FF',
           'CPU-Time Recon High F1','CPU-Time Recon High F2','CPU-Time Recon High FF', 'Fractal 1 SSIM','Fractal 2 SSIM','Fractal Final SSIM']
df_results = pd.DataFrame(columns=columns)
count = 0

os.chdir(url)
for img in glob.glob("*.jpg"):
    temp_dict = dict.fromkeys(columns)
#    ssim_dict = {'fractal 1':0.0,'fractal 2':0.0,'fractal 3+':0.0}
    
    #Must implement image reloading for this to work efficiently
    count = count + 1
    print("Current Image count: ",count,"Current Image name: ", img)
    temp_dict['Image Label'] = img
    #loads the current image onto it's graphs:
    pi.loadNewImage(img)
    print("New Image Loaded")
    pi.plotOriginalImage()
    
    pi.triple_DPT(feedback=0)
    temp_dict['CPU-Time DPT'] = pi.RP.time_dpt + pi.RP.time_feat_table
    temp_dict['CPU-Time DPT Low'] = pi.RPl_sample.time_dpt + pi.RP.time_feat_table
    temp_dict['CPU-Time DPT High'] = pi.RPh_sample.time_dpt + pi.RP.time_feat_table
    #Get distribution of scales:
    pi.getHt_structure()
    temp_dict['Ht index'] = pi.Ht_orig
    temp_dict['Ht index Low'] = pi.Ht_low
    temp_dict['Ht index High'] = pi.Ht_high
    print("Ht structure:")
    print(pi.Ht_means_orig)
    print(pi.Ht_means_low)
    print(pi.Ht_means_high)
    
    try:#attempt the reconstruction
        ###################################################################################################
        #FRACTAL 1
        pi.triple_recon_partial(1,pi.Ht_means_orig[0],1,pi.Ht_means_low[0],1,pi.Ht_means_high[0])
        print("Recon Orig for fractal 1:")
        pi.plotDPTRecon()
        print("Recon Dual for fractal 1:")
        pi.plotDualDPTRecon()
        x_pil_O = Image.fromarray(pi.RP.signal_to_image(),'L')
        x_pil_D = Image.fromarray(pi.signal_dual.reshape(pi.imgShape),'L')
        temp_dict['Fractal 1 SSIM'] = compare_ssim(x_pil_O,x_pil_D,tile_size=tile_size,GPU=False)
        print("SSIM for fractal 1: ", temp_dict['Fractal 1 SSIM'] )
        
        temp_dict['CPU-Time Recon F1'] = pi.RP.time_recon
        temp_dict['CPU-Time Recon Low F1'] = pi.RPl_sample.time_recon
        temp_dict['CPU-Time Recon High F1'] = pi.RPh_sample.time_recon
        
        
        ###################################################################################################
        #FRACTAL 2    
        pi.triple_recon_partial(pi.Ht_means_orig[0],pi.Ht_means_orig[1],pi.Ht_means_low[0],pi.Ht_means_low[1],pi.Ht_means_high[0],pi.Ht_means_high[1])
        print("Recon Orig for fractal 2:")
        pi.plotDPTRecon()
        print("Recon Dual for fractal 2:")
        pi.plotDualDPTRecon()
        #compare ssim using suiatble object convertion to PIL:
        x_pil_O = Image.fromarray(pi.RP.signal_to_image(),'L')
        x_pil_D = Image.fromarray(pi.signal_dual.reshape(pi.imgShape),'L')
        temp_dict['Fractal 2 SSIM'] = compare_ssim(x_pil_O,x_pil_D,tile_size=tile_size,GPU=False)
        print("SSIM for fractal 2: ", temp_dict['Fractal 2 SSIM'])
        
        temp_dict['CPU-Time Recon F2'] = pi.RP.time_recon
        temp_dict['CPU-Time Recon Low F2'] = pi.RPl_sample.time_recon
        temp_dict['CPU-Time Recon High F2'] = pi.RPh_sample.time_recon
        
        ###################################################################################################
        #FRACTAL FINAL    
        pi.triple_recon_partial(pi.Ht_means_orig[1],np.inf,pi.Ht_means_low[1],np.inf,pi.Ht_means_high[1],np.inf)
        print("Recon Orig for final fractal:")
        pi.plotDPTRecon()
        print("Recon Dual for final fractal :")
        pi.plotDualDPTRecon()
        #compare ssim using suiatble object convertion to PIL:
        x_pil_O = Image.fromarray(pi.RP.signal_to_image(),'L')
        x_pil_D = Image.fromarray(pi.signal_dual.reshape(pi.imgShape),'L')
        temp_dict['Fractal Final SSIM'] = compare_ssim(x_pil_O,x_pil_D,tile_size=tile_size,GPU=False)
        print("SSIM for final fractal: ", temp_dict['Fractal Final SSIM'])
        
        temp_dict['CPU-Time Recon FF'] = pi.RP.time_recon
        temp_dict['CPU-Time Recon Low FF'] = pi.RPl_sample.time_recon
        temp_dict['CPU-Time Recon High FF'] = pi.RPh_sample.time_recon
        
        ##########################################################################################################
        #ADD RESULTS TO DATAFRAME:
        df_results = df_results.append(temp_dict,ignore_index=True)
        
        #ADD DIRECTLY TO EXCEL:
        df_temp = pd.DataFrame(temp_dict, index=[0])
        df_to_excel.append_df_to_excel("results.xlsx",df_temp,header=False)
    except:
        print("Reconstruction not valid - result discarded")    
    

df_results.to_excel("Mock Trial 1.xlsx")
