# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:31:15 2020

@author: Mark
"""

#from PIL import Image
import copy
import rmpavage.rmpa as rmpa
import rmpavage.graph_filter_banks as graph_filter_banks
import numpy as np
import networkx as nx
import time as time
import skimage.io as io
import skimage.util
import scipy.sparse.linalg as lin
import matplotlib.pyplot as plt

# Process_image Class:
class ProcessImage:
    def __init__(self, img):
        self.img = io.imread(img,as_gray=True)#open the file
        self.img = skimage.util.img_as_ubyte(self.img)
        #self.img = np.array(self.img)#convert to numpy array
        self.imgShape = self.img.shape
        self.RP = rmpa.RoadmakersPavage(self.img,True)
        # initilize placeholders for distribution scale:
        self.original_scale_dist = None
        self.sample_low_scale_dist = None
        self.sample_high_scale_dist = None
        
    #imposes a new image on each of the graphs, graph attributes must be defined already:
    def loadNewImage(self,img):
        #load and reshape the image:
        self.img = Image.open(img)#open the file
        self.img = self.img.convert('L')#convert to grayscale
        self.img = np.array(self.img)#convert to numpy array 
        self.img = self.img.astype(int)
        self.x = copy.deepcopy(self.img.ravel())
#        print("type x:", type(self.x[0]))
#        print("min x:", min(self.x))
#        print("max x:", max(self.x))
##        
#        plt.figure()
#        imgplot = plt.imshow(self.x.reshape(self.imgShape),cmap="gray")
#        plt.show()
        
        
        #load previous shape and incarnations of the graphs by loading previous backups from before the process began:
#        self.RP.loadBackUp_PG()
#        self.RP.loadBackUp_WG() 
        
        self.RP = copy.deepcopy(self.RP_back)
        #low frequency sample graphs:
        self.RPl_sample = copy.deepcopy(self.RPl_sample_back)
        self.RPh_sample= copy.deepcopy(self.RPh_sample_back)
                
        #change the value attributes of the WG for RP to those of the new image:
        temp_x = [{'value':i} for i in self.x]
        temp_x = dict(enumerate(temp_x,start=1))
        nx.set_node_attributes(self.RP.WG,temp_x)
        
        #filter the new image:        
        t = time.process_time()
        self.filterHighLow()        
        elapsed_time = time.process_time() - t
        print("CPU time to create filter image: ", elapsed_time)
        self.time_filter = elapsed_time
                
        temp_xl = [{'value':i} for i in self.xl[self.inds_bottom]]
        temp_xl = dict(enumerate(temp_xl,start=1))
        
        temp_xh = [{'value':i} for i in self.xh[self.inds_top]]
        temp_xh = dict(enumerate(temp_xh,start=1))
        
        nx.set_node_attributes(self.RPl_sample.WG,temp_xl)
        nx.set_node_attributes(self.RPh_sample.WG,temp_xh)
        #not implemented
        
    def defineGraphAttributes(self,refmat = 'NL',Lmax = 2 ,Lmin = 0):
        #reference matrix used:
        # NL = symmetric normalised laplacian
        # L = laplacian
        # A = adjacency
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
        if refmat == 'NL':
            self.L = nx.normalized_laplacian_matrix(WG_nozero)
            self.Lmax = 2 #1.9999999999999999999998
        if refmat == 'L':
            self.L = nx.laplacian_matrix(WG_nozero)            
        if refmat == 'A':
            self.L = nx.adjacency_matrix(WG_nozero)
            
        self.Lmax = Lmax
        self.Lmin = Lmin
                       
        self.N = len(self.x)
        #Signals Stored:
        self.signal_orig = 0
        self.signal_dual = 0
        
    def defineGraphSpectrum(self,refmat):
#        raise not implemented
        return "derp"
        
        
    def estLmax(self):
        self.Lmax = lin.eigs(A=self.L.asfptype(),k=1,return_eigenvectors=False)
        self.Lmax = self.Lmax.real
        
    def estLmin(self):
        self.Lmin = lin.eigs(A=self.L.asfptype(),k=1,which='SM',return_eigenvectors=False)
        self.Lmin = self.Lmin.real
        
    def createDualFilters(self,w=1,m=500):
        self.filter = graph_filter_banks.GraphFilter(self.L)
        self.cl = self.filter.cheby(w=w,m=m,Lmax=self.Lmax,i=1,Lmin=self.Lmin)
        self.ch = self.filter.cheby(w=w,m=m,Lmax=self.Lmax,i=0,Lmin=self.Lmin)
        
    def filterHighLow(self,show=True):
        #Rounds off to integer for now,maybe implement parameter for decimal place later
        self.xl = np.around(self.filter.cheby_op(R=self.L, c=self.cl, x=self.x, Lmax=self.Lmax,Lmin=self.Lmin))
        self.xh = np.around(self.filter.cheby_op(R=self.L, c=self.ch, x=self.x, Lmax=self.Lmax,Lmin=self.Lmin))
        
        if show:
            plt.figure()
            imgplot = plt.imshow(self.xl.reshape(self.imgShape),cmap="gray")
            ax = plt.gca()
            ax.grid(False)
            # Hide axes ticks
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.show()

            plt.figure()
            imgplot = plt.imshow(self.xh.reshape(self.imgShape),cmap="gray")
            ax = plt.gca()
            # Hide axes ticks
            ax.grid(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
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
        self.RP.reconstruct_pulses_range_via_while(0,np.inf)
        self.signal_orig = self.RP.current_signal
        self.RPl_sample.reconstruct_pulses_range_via_while(0,np.inf)
        self.RPh_sample.reconstruct_pulses_range_via_while(0,np.inf)
        #self.RP.reconstruct_full()
        #self.RPl_sample.reconstruct_full()
        #self.RPh_sample.reconstruct_full()
        #interpolate and correct lower:
        xl_interp = np.zeros(self.x.shape)
        xl_interp[self.inds_bottom] = self.RPl_sample.current_signal
        xl_interp = 2*np.around(self.filter.cheby_op(R=self.L, c=self.cl, x=xl_interp, Lmax=self.Lmax))
        self.xl_interp = xl_interp
        #interpolate and correct higher:
        xh_interp = np.zeros(self.x.shape)
        xh_interp[self.inds_top] = self.RPh_sample.current_signal
        xh_interp = 2*np.around(self.filter.cheby_op(R=self.L, c=self.ch, x=xh_interp, Lmax=self.Lmax))
        self.xh_interp = xh_interp
        #add low/high frequencies:
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
        self.xl_interp = xl_interp
#        plt.figure()
#        imgplot = plt.imshow(xl_interp.reshape(self.imgShape),cmap="gray")
#        plt.show()
        
        xh_interp = np.zeros(self.x.shape)
        xh_interp[self.inds_top] = self.RPh_sample.current_signal
        xh_interp = 2*np.around(self.filter.cheby_op(R=self.L, c=self.ch, x=xh_interp, Lmax=self.Lmax))
        self.xh_interp = xh_interp
#        plt.figure()
#        imgplot = plt.imshow(xh_interp.reshape(self.imgShape),cmap="gray")
#        plt.show()
        
        self.signal_dual = xl_interp + xh_interp
        
        
    def createSubGraphs(self):       
        print("Creating Subgraphs...")
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
                
        print("New diagonal edges added")
                                
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
        self.RP_back = copy.deepcopy(self.RP)
                
                
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
        ax = plt.gca()
        # Hide axes ticks
        ax.grid(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.show()
        
    def plotDPTRecon(self,exact=False):
        plt.figure()
        if exact:
            imgplot = plt.imshow(self.RP.signal_to_image(),cmap="gray",vmin=0,vmax=255)
            ax = plt.gca()
            # Hide axes ticks
            ax.grid(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        else:
            imgplot = plt.imshow(self.RP.signal_to_image(),cmap="gray")
            ax = plt.gca()
            # Hide axes ticks
            ax.grid(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        plt.show()

    def plotDualDPTRecon(self):
        plt.figure()
#       imgplot = plt.imshow(self.signal_dual.reshape(self.imgShape),cmap="gray",vmin=0,vmax=255)
        imgplot = plt.imshow(self.signal_dual.reshape(self.imgShape),cmap="gray")
        ax = plt.gca()
        # Hide axes ticks
        ax.grid(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.show()

    def plotHighLow(self):
        plt.figure()
        imgplot = plt.imshow(self.xl_interp.reshape(self.imgShape),cmap="gray")
        ax = plt.gca()
        # Hide axes ticks
        ax.grid(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.show()
        plt.figure()
        imgplot = plt.imshow(self.xh_interp.reshape(self.imgShape),cmap="gray")
        ax = plt.gca()
        # Hide axes ticks
        ax.grid(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
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
        
        
        
        