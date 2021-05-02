# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:31:15 2020

@author: Mark
"""

from PIL import Image

import copy
import rmpavage.rmpa as rmpa
from rmpavage import graph_filter_banks
import numpy as np
import networkx as nx

import time as time

import matplotlib.pyplot as plt

# Process_image Class:
class ProcessImage:
    def __init__(self, img):
        self.img = Image.open(img)#open the file
        self.img = self.img.convert('L')#convert to grayscale
        self.img = np.array(self.img)#convert to numpy array        
        self.imgShape = self.img.shape
        self.RP = rmpa.RoadmakersPavage(self.img,True)
        
        self.original_scale_dist = None
        self.sample_low_scale_dist = None
        self.sample_high_scale_dist = None
        
        
    def loadNewImage(self,img):
        #not implemented
        raise NotImplementedError
        
    def defineGraphAttributes(self):
        WG_nozero = copy.deepcopy(self.RP.WG)
        WG_nozero.remove_node(0)
    
        #Sample bipartite set indices:
        bottom_nodes, top_nodes = nx.algorithms.bipartite.sets(WG_nozero)
        bottom_nodes = sorted(bottom_nodes)
        top_nodes = sorted(top_nodes)
        self.inds_bottom = np.array(list(bottom_nodes))-1
        self.inds_top = np.array(list(top_nodes))-1
        self.bottom_nodes = set(bottom_nodes)
        self.top_nodes = set(top_nodes)
        
        #get signal:
        self.x = np.fromiter(nx.get_node_attributes(WG_nozero,'value').values(),dtype=float)
        #define laplacian and Largest Eigenvalue:
        self.L = nx.normalized_laplacian_matrix(WG_nozero)
        self.Lmax = 1.9999999999999999999998
        self.N = len(self.x)
        
        #Signals Stored:
        self.signal_orig = 0
        self.signal_dual = 0
        
        
    def estLmax(self):
        #not implemented
        raise NotImplementedError
        
    def createDualFilters(self,w=1,m=500):
        self.filter = graph_filter_banks.GraphFilter(self.L)
        self.cl = self.filter.cheby(w=w,m=m,Lmax=self.Lmax,i=0)
        self.ch = self.filter.cheby(w=w,m=m,Lmax=self.Lmax,i=1)
        
    def filterHighLow(self,show=True):
        #Rounds off to integer for now,maybe implement parameter for decimal place later
        self.xl = np.around(self.filter.cheby_op(R=self.L, c=self.cl, x=self.x, Lmax=self.Lmax))
        self.xh = np.around(self.filter.cheby_op(R=self.L, c=self.ch, x=self.x, Lmax=self.Lmax))
        
        if show:
            plt.figure()
            imgplot = plt.imshow(self.xl.reshape(self.imgShape),cmap="gray")
            plt.show()
            plt.figure()
            imgplot = plt.imshow(self.xh.reshape(self.imgShape),cmap="gray")
            plt.show()       
        
        
    def DPT_original(self,feedback=0):
        self.RP.create_feat_table()
        self.RP.dpt(feedback=feedback)

    def DPT_low(self,feedback=0):
        #Low Frequencies:
        self.RPl_sample.create_feat_table()
        self.RPl_sample.dpt(feedback=feedback)
        
    def DPT_high(self,feedback=0):
        #High Frequencies:
        self.RPh_sample.create_feat_table()
        self.RPh_sample.dpt(feedback=feedback)
        
    def DPT_dualsub(self,feedback=0):
        #High Frequencies:
        self.DPT_low(feedback)
        self.DPT_high(feedback)
        
    def triple_DPT(self,feedback=0):
        self.DPT_original(feedback=feedback)
        self.DPT_dualsub(feedback=feedback)
        
    def triple_recon_full(self):
        self.RP.reconstruct_full()
        self.RPl_sample.reconstruct_full()
        self.RPh_sample.reconstruct_full()
        
        xl_interp = np.zeros(self.x.shape)
        xl_interp[self.inds_bottom] = self.RPl_sample.current_signal
        xl_interp = 2*np.around(self.filter.cheby_op(R=self.L, c=self.cl, x=xl_interp, Lmax=self.Lmax))
        
#        plt.figure()
#        imgplot = plt.imshow(xl_interp.reshape(self.imgShape),cmap="gray")
#        plt.show()
        
        xh_interp = np.zeros(self.x.shape)
        xh_interp[self.inds_top] = self.RPh_sample.current_signal
        xh_interp = 2*np.around(self.filter.cheby_op(R=self.L, c=self.ch, x=xh_interp, Lmax=self.Lmax))
        
#        plt.figure()
#        imgplot = plt.imshow(xh_interp.reshape(self.imgShape),cmap="gray")
#        plt.show()
        
        self.signal_dual = xl_interp + xh_interp
    
    def original_partial_recon(self,minO,maxO):
        self.RP.reconstruct_pulses_range_via_while(minO,maxO)
        
        
    def triple_recon_partial(self,minO,maxO,minl,maxl,minh,maxh):
        self.RP.reconstruct_pulses_range_via_while(minO,maxO)
        self.signal_orig = self.RP.current_signal
        self.RPl_sample.reconstruct_pulses_range_via_while(minl,maxl)
        self.RPh_sample.reconstruct_pulses_range_via_while(minh,maxh)
        
        xl_interp = np.zeros(self.x.shape)
        xl_interp[self.inds_bottom] = self.RPl_sample.current_signal
        xl_interp = 2*np.around(self.filter.cheby_op(R=self.L, c=self.cl, x=xl_interp, Lmax=self.Lmax))
        
#        plt.figure()
#        imgplot = plt.imshow(xl_interp.reshape(self.imgShape),cmap="gray")
#        plt.show()
        
        xh_interp = np.zeros(self.x.shape)
        xh_interp[self.inds_top] = self.RPh_sample.current_signal
        xh_interp = 2*np.around(self.filter.cheby_op(R=self.L, c=self.ch, x=xh_interp, Lmax=self.Lmax))
        
#        plt.figure()
#        imgplot = plt.imshow(xh_interp.reshape(self.imgShape),cmap="gray")
#        plt.show()
        
        self.signal_dual = xl_interp + xh_interp
        
        
    def createSubGraphs(self):        
        #create subgraphs and backup:
        width = self.img.shape[1]
        #create temporary graph with same structure as original DPT working graph:
        RPtemp = rmpa.RoadmakersPavage(self.img,True)
        #Add more edges:
        nodes = list(RPtemp.WG.nodes())[1:]
        for i in nodes:
            if ((i-width-1)>0) and ((i+width-1) % width !=0):
                RPtemp.WG.add_edge(i, i-width-1)
            if ((i-width+1)>0) and (i % width !=0): 
                RPtemp.WG.add_edge(i, i-width+1)
            if (i+width-1)<len(RPtemp.WG) and ((i+width-1) % width !=0): 
                RPtemp.WG.add_edge(i, i+width-1)
            if (i+width+1)<len(RPtemp.WG) and (i % width !=0):
                RPtemp.WG.add_edge(i, i+width+1)
                                
        self.RPl_sample = copy.deepcopy(RPtemp)
        self.bottom_nodes.update([0])
        self.RPl_sample.WG = copy.deepcopy(self.RPl_sample.WG.subgraph(self.bottom_nodes))
        self.RPl_sample.WG = nx.convert_node_labels_to_integers(self.RPl_sample.WG, first_label=0, ordering='sorted')
        #Find a way to use networkx.classes.function.set_node_attributes instead of this loop
        for  pulse_i in self.RPl_sample.WG.nodes(): 
            if pulse_i !=0: self.RPl_sample.WG.nodes[pulse_i]['pulses'] = set([int(pulse_i)])
        self.RPl_sample.PG = copy.deepcopy(self.RPl_sample.PG.subgraph(self.bottom_nodes-set([0])))
        self.RPl_sample.PG = nx.convert_node_labels_to_integers(self.RPl_sample.PG, first_label=1, ordering='sorted')
                
        self.RPh_sample = copy.deepcopy(RPtemp)
        self.top_nodes.update([0])
        self.RPh_sample.WG = copy.deepcopy(self.RPh_sample.WG.subgraph(self.top_nodes))
        self.RPh_sample.WG = nx.convert_node_labels_to_integers(self.RPh_sample.WG, first_label=0, ordering='sorted')
        for  pulse_i in self.RPh_sample.WG.nodes(): 
            if pulse_i !=0: self.RPh_sample.WG.nodes[pulse_i]['pulses'] = set([int(pulse_i)])
        self.RPh_sample.PG = copy.deepcopy(self.RPh_sample.PG.subgraph(self.top_nodes-set([0])))
        self.RPh_sample.PG = nx.convert_node_labels_to_integers(self.RPh_sample.PG, first_label=1, ordering='sorted')
        
        
        temp_xl = [{'value':i} for i in self.xl[self.inds_bottom]]
        temp_xl = dict(enumerate(temp_xl,start=1))
        
        temp_xh = [{'value':i} for i in self.xh[self.inds_top]]
        temp_xh = dict(enumerate(temp_xh,start=1))
        
        nx.set_node_attributes(self.RPl_sample.WG,temp_xl)
        nx.set_node_attributes(self.RPh_sample.WG,temp_xh)
        
        #createbackup structures:
        self.RPl_sample_back = copy.deepcopy(self.RPl_sample)
        self.RPh_sample_back = copy.deepcopy(self.RPh_sample)
                
                
    def loadBackupGraphs(self):
        def loadBackUp_WG(self):
            self.WG = copy.deepcopy(self.WG_bak)
            
        #load the backup of the WG into the usable WG in case something went wrong:
        def loadBackUp_PG(self):
            self.PG = copy.deepcopy(self.PG_bak)
        #not implemented
        raise NotImplementedError  


    def plotOriginalImage(self):
        plt.figure()
#        imgplot = plt.imshow(self.img,cmap="gray",vmin=0,vmax=255)
        imgplot = plt.imshow(self.img,cmap="gray")
        plt.show()
        
    def plotDPTRecon(self,exact=False):
        plt.figure()
        if exact:
            imgplot = plt.imshow(self.RP.signal_to_image(),cmap="gray",vmin=0,vmax=255)
        else:
            imgplot = plt.imshow(self.RP.signal_to_image(),cmap="gray")
        plt.show()

    def plotDualDPTRecon(self):
        plt.figure()
#        imgplot = plt.imshow(self.signal_dual.reshape(self.imgShape),cmap="gray",vmin=0,vmax=255)
        imgplot = plt.imshow(self.signal_dual.reshape(self.imgShape),cmap="gray")
        plt.show()
        
    #Statistical analysis:
    def getOriginalScaleDist(self):
        temp_dict = nx.get_node_attributes(self.RP.PG,'scale')
        temp_dict = {k:v for k,v in temp_dict.items() if np.inf > v > 0}
        self.original_scale_dist = np.array(list(temp_dict.values()))
    
    def getSampleLowScaleDist(self):
        temp_dict = nx.get_node_attributes(self.RPl_sample.PG,'scale')
        temp_dict = {k:v for k,v in temp_dict.items() if np.inf > v > 0}
        self.sample_low_scale_dist = np.array(list(temp_dict.values())) 

    def getSampleHighScaleDist(self):
        temp_dict = nx.get_node_attributes(self.RPh_sample.PG,'scale')
        temp_dict = {k:v for k,v in temp_dict.items() if np.inf > v > 0}
        self.sample_high_scale_dist = np.array(list(temp_dict.values()))
    
    def getAllScaleDist(self):
        self.getOriginalScaleDist()
        self.getSampleLowScaleDist()
        self.getSampleHighScaleDist()


    @staticmethod
    def calculateHt(scale_dist,feedback=False):
        n = len(scale_dist)
        tempdist = np.array(scale_dist)
        mean = np.around(tempdist.mean())
        n_less = len(tempdist[tempdist<mean])
        n_more = n-n_less
        Ht_means = []
        count = 0
        if feedback:
            print("first ",n_less,n_more, mean)
        while n_less > n_more:
            Ht_means.append(int(mean))
            count = count+1
            tempdist = tempdist[tempdist>mean]#reduce tempdist to it's right tale
            n = len(tempdist)
            mean = np.around(tempdist.mean())
            n_less = len(tempdist[tempdist<mean])
            n_more = n-n_less
            if feedback:
                print(n_less,n_more, mean)
        Ht = count+1
        return Ht,Ht_means
            
        
    
    def getHt_structure(self):
#        if None in any([self.sample_low_scale_dist,self.sample_high_scale_dist,self.original_scale_dist]):
        self.getAllScaleDist()
        self.Ht_orig,self.Ht_means_orig = self.calculateHt(self.original_scale_dist)
        self.Ht_low,self.Ht_means_low = self.calculateHt(self.sample_low_scale_dist)
        self.Ht_high,self.Ht_means_high = self.calculateHt(self.sample_high_scale_dist)
            
        #original Ht-structure:
        
        
        
        