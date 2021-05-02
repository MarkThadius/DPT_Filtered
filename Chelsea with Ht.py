# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:10 2019

@author: Mark
"""
#https://github.com/Ashish7129/Graph_Sampling
import rmpavage.process_image
import numpy as np
import matplotlib.pyplot as plt
import importlib
from skimage.metrics import structural_similarity as ssim
import seaborn as sns
import pandas as pd
importlib.reload(rmpavage.process_image)

img = r"chelsea.jpg"#import the kitty
img = r"8049.jpg"
img = r"12074.jpg"
img = r"23084.jpg"
img = r"nordic_landscape-wallpaper-1280x720.jpg"
img = r"forest_trees_stones_205266_1280x720.jpg"

pi = rmpavage.process_image.ProcessImage(img)
pi.plotOriginalImage()
pi.defineGraphAttributes()
pi.createDualFilters(w=1, m=500)
pi.filterHighLow()
pi.createSubGraphs()
pi.triple_DPT(feedback=3)
#reconstruct all signals using all scales:
pi.triple_recon_full()
#Original Reconstructed:
pi.plotDPTRecon()
#Filtered Signals Reconstructed:
pi.plotDualDPTRecon()
# SSIM of fully reconstructed:
ssim(pi.signal_dual.reshape(pi.imgShape), pi.img)

#####################################################################################
#Get distribution of scales:
pi.getAllScaleDist()
pi.getHt_structure()
pi.Ht_means_orig
pi.Ht_means_low
pi.Ht_means_high
# histograms:
# scales = pd.DataFrame(pi.original_scale_dist,columns=['scale'])
# scales = pd.DataFrame(pi.sample_high_scale_dist,columns=['scale'])
scales = pd.DataFrame(pi.sample_low_scale_dist, columns=['scale'])
scale_count = scales['scale'].value_counts().rename_axis('Scale').reset_index(name='Relative Frequency')
scale_count.sort_values(by=['Scale'], inplace=True)
scale_count['log(Scale)'] = np.log(scale_count['Scale'])
lineplot = sns.lineplot(x='log(Scale)', y='Relative Frequency', data=scale_count)
lineplot.set_xlabel(xlabel="log(Scale)", fontsize=16)
lineplot.set_ylabel(ylabel="Relative Frequency", fontsize=16)
####################################
pi.triple_recon_partial(1, pi.Ht_means_orig[0], 1, pi.Ht_means_low[0], 1, pi.Ht_means_high[0])
# pi.triple_recon_partial(pi.Ht_means_orig[0],pi.Ht_means_orig[1],pi.Ht_means_low[0],pi.Ht_means_low[1],pi.Ht_means_high[0],pi.Ht_means_high[1])
# pi.triple_recon_partial(pi.Ht_means_orig[1],pi.Ht_means_orig[2],pi.Ht_means_low[1],pi.Ht_means_low[2],pi.Ht_means_high[1],pi.Ht_means_high[2])
# pi.triple_recon_partial(pi.Ht_means_orig[2],pi.Ht_means_orig[3],pi.Ht_means_low[2],pi.Ht_means_low[3],pi.Ht_means_high[2],pi.Ht_means_high[3])
# pi.triple_recon_partial(pi.Ht_means_orig[3],pi.Ht_means_orig[4],pi.Ht_means_low[3],pi.Ht_means_low[4],pi.Ht_means_high[3],pi.Ht_means_high[4])
# pi.triple_recon_partial(pi.Ht_means_orig[4],pi.Ht_means_orig[5],pi.Ht_means_low[4],pi.Ht_means_low[5],pi.Ht_means_high[4],pi.Ht_means_high[5])
pi.triple_recon_partial(pi.Ht_means_orig[5], np.inf, pi.Ht_means_low[5], np.inf, pi.Ht_means_high[5], np.inf)

# Other intervals:
pi.triple_recon_partial(pi.Ht_means_orig[3], np.inf, pi.Ht_means_low[3],np.inf, pi.Ht_means_high[3],np.inf)

# SSIM of partially reconstructed:
ssim(pi.signal_dual.reshape(pi.imgShape), pi.signal_orig.reshape(pi.imgShape))
ssim(pi.xl_interp.reshape(pi.imgShape), pi.signal_orig.reshape(pi.imgShape))
ssim(pi.xh_interp.reshape(pi.imgShape), pi.signal_orig.reshape(pi.imgShape))
pi.xl_interp
#Plot Results:
#Original Reconstructed:
pi.plotDPTRecon()
plt.imshow(pi.RP.signal_to_image(),cmap="gray")
#Filtered Signals Reconstructed:
pi.plotDualDPTRecon()
#
pi.plotHighLow()



#Print histograms of the log-scales
logscales = np.log10(pi.original_scale_dist)
_ = plt.hist(logscales,bins = 50)
log_lowfreq_scales = np.log10(pi.sample_low_scale_dist)
_ = plt.hist(log_lowfreq_scales,bins = 50) #underscore is unused variable
log_highfreq_scales = np.log10(pi.sample_high_scale_dist)
_ = plt.hist(log_highfreq_scales,bins = 50) #underscore is unused variable