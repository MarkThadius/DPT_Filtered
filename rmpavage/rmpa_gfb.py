""" Graph Data Structure Classes for the Roadmakers Pavage Algorithm """
__author__ = "Mark de Lancey"
__copyright__ = "Copyright 2019, The University of Pretoria"
__version__ = "0.2.0"
__email__ = "mark.stephen.del@gmail.com"
__status__ = "1.0"

# Import Packages required
import numpy as np
#import networkx as nx
import rmpavage.rmpa as rmpa
import copy as copy
import Graph_Sampling as gs
import networkx as nx
import time as time

class RMPA_GFB:
    # Initialise using a greyscale image as a signal.
    def __init__(self, Image,K,V=None,P_l=None):
        # Store the image in the class.                
        self.imgShape = Image.shape
        self.N = Image.shape[0]*Image.shape[1]#full signal length/number of pixels
        
        self.RP_full = rmpa.RoadmakersPavage(Image)
        self.x = copy.deepcopy(Image.ravel())
        self.V = V
        self.K = K#bandwidth
        #x_l objects:
        self.x_l = 0;
        self.P_l = P_l;
        self.sample_l = set();        
        self.sample_mhrw_l = 0;
        self.optimal_sample_l = 0;
        self.ntrpl8_l = 0;
        self.RP_l = 0;
        #x_h objects:
        self.P_h = 0;
        self.x_h = 0;
        self.sample_h = set();
        self.sample_mhrw_h = 0;
        self.optimal_sample_h = 0;
        self.ntrpl8_h = 0;
        self.RP_h = 0;
        #interpolated signal:
        self.x_inter = 0;
        #sampler:

        
        
    #Create a graph filter bank using eigenvectors and splits the signal into 2
    #bandlimited signals, x_l and x_h:
    #This is ONLY possible with a full NxN eigenvector matrix
    def gfb_m(self,K=None):
        if not K: self.K=K
        t = time.process_time()
#        V_inv = V.T     
        
        
        P_l = np.zeros([self.N,self.N])
        P_l[:self.K,:self.K] = np.identity(self.K)
        P_l = self.V @ P_l @ self.V.T
        self.P_l = P_l
        self.x_l = self.P_l @ self.x#bandlimit x_l
        
#        P_h = np.zeros([self.N,self.N])
#        P_h[self.K:self.N,self.K:self.N] = np.identity(self.N-self.K)
#        P_h = self.V @ P_h @ self.V.T
#        self.P_h = P_h
#        self.x_h = self.P_h @ self.x
        self.x_h = np.around(self.x-self.x_l,6)#get second bandlimited signal using difference
        elapsed_time = time.process_time() - t
        print("CPU time to construct graph filter bank: ", elapsed_time)
        
    #initiallize the respective RMPA objects for each bandlimited signal:
    def init_RPs(self):
        self.RP_l = rmpa.RoadmakersPavage(self.x_l.reshape(self.imgShape))
        self.RP_h = rmpa.RoadmakersPavage(self.x_h.reshape(self.imgShape))
        
    def get_greedy_optimal_sampler_l(self,M):
        unused_inds = set(range(0,self.N))
        s_inds = set()
        Vk = self.V[:,0:self.K]
        while len(s_inds) < M:
            m = 0
            best_i = -1
            for i in unused_inds:
                inds = s_inds.union([i])
                sigma = np.linalg.svd(Vk[list(inds),:],compute_uv=False)
                sigma = sigma.min()
                if m < sigma:
                    m = sigma
                    best_i = i
                
#                print(sigma,m,best_i)
                #end if
            #end for
            s_inds.update([best_i])
            unused_inds.remove(best_i)
            print(len(s_inds), " of ", M, " complete with best index: ", best_i)
        #end while
        s_inds = set(np.array(list(s_inds))+1)
        self.optimal_sample_l = set(sorted(s_inds))
        self.optimal_sample_l.update([0])
        
    def get_greedy_optimal_sampler_h(self,M):
        unused_inds = set(range(0,self.N))
        s_inds = set()
        Vk = self.V[:,self.K:self.N]
        while len(s_inds) < M:
            m = 0
            best_i = -1
            for i in unused_inds:
                inds = s_inds.union([i])
                sigma = np.linalg.svd(Vk[list(inds),:],compute_uv=False)
                sigma = sigma.min()
                if m < sigma:
                    m = sigma
                    best_i = i
                
#                print(sigma,m,best_i)
                #end if
            #end for
            s_inds.update([best_i])
            unused_inds.remove(best_i)
            print(len(s_inds), " of ", M, " complete with best index: ", best_i)
        #end while
        s_inds = set(np.array(list(s_inds))+1)
        self.optimal_sample_h = set(sorted(s_inds))
        self.optimal_sample_h.update([0])
        
                
    def set_current_sampler_l(self):
        #Add mode options and params to this later.
        self.sample_l = set(self.optimal_sample_l)
        self.sample_l.update([0])
        
    def set_current_sampler_h(self):
        #Add mode options and params to this later.
        self.sample_h = set(self.optimal_sample_h)
        self.sample_h.update([0])
        
    def RW_l(self,n):
        self.sampler=gs.SRW_RWF_ISRW()
        N = len(self.RP_l.WG)
#        n = int(np.around(len(self.RP_l.WG)/2,0))
        print(n)
        self.sample_l = 0
        self.sample_l = self.sampler.random_walk_induced_graph_sampling(self.RP_l.WG,n+1) # graph, seed, sample size
        print(len(self.sample_l))
        self.RP_l.WG.remove_nodes_from(np.arange(N,len(self.RP_l.WG)))
#        print(list(self.sample_l))
        self.sample_l = set(sorted(self.sample_l))
        self.sample_l.update([0])
        
    def RW_h(self,n):
        self.sampler=gs.SRW_RWF_ISRW()
        N = len(self.RP_h.WG)
#        n = int(np.around(len(self.RP_h.WG)/2,0))
        print(n)
        self.sample_h = 0
        self.sample_h = self.sampler.random_walk_induced_graph_sampling(self.RP_h.WG,n+1) # graph, seed, sample size
        print(len(self.sample_h))
        self.RP_h.WG.remove_nodes_from(np.arange(N,len(self.RP_h.WG)))
#        print(list(self.sample_h))
        self.sample_h = set(sorted(self.sample_h))
        self.sample_h.update([0])
        
    def MHRW_l(self,n,seed=0):
        self.sampler=gs.MHRW()

        N = len(self.RP_l.WG)
#        n = int(np.around(len(self.RP_l.WG)/2,0))
        print(n)
        self.sample_l = 0
        self.sample_l = self.sampler.mhrw(self.RP_l.WG,seed,n+1) # graph, seed, sample size
        print(len(self.sample_l))
        self.RP_l.WG.remove_nodes_from(np.arange(N,len(self.RP_l.WG)))
#        print(list(self.sample_l))
        self.sample_l = set(sorted(self.sample_l))
        
    def MHRW_h(self,n,seed=0):
        N = len(self.RP_h.WG)
#        n = int(np.around(len(self.RP_h.WG)/2,0))
        print(n)
        self.sample_h = self.sampler.mhrw(self.RP_h.WG,seed,n+1) # graph, seed, sample size
        self.RP_h.WG.remove_nodes_from(np.arange(N,len(self.RP_h.WG)))
#        print(list(self.sample_h))
        self.sample_h = set(sorted(self.sample_h))
        
    def sample_signal_l(self):    
        self.sample_l.update([0])
#        print(self.sample_l)
        self.RP_l.WG = self.RP_l.WG.subgraph(self.sample_l)
        self.RP_l.WG = nx.convert_node_labels_to_integers(self.RP_l.WG , first_label=0, ordering='sorted')
        for  x in self.RP_l.WG.nodes(): 
            if x !=0: self.RP_l.WG.nodes[x]['pulses'] = set([int(x)])
            
        self.RP_l.PG = self.RP_l.PG.subgraph(self.sample_l-set([0]))
        self.RP_l.PG = nx.convert_node_labels_to_integers(self.RP_l.PG, first_label=1, ordering='sorted')
        
            
    def sample_signal_h(self):    
        self.sample_h.update([0])
#        print(self.sample_h)
        self.RP_h.WG = self.RP_h.WG.subgraph(self.sample_h)
        self.RP_h.WG = nx.convert_node_labels_to_integers(self.RP_h.WG , first_label=0, ordering='sorted')
        for  x in self.RP_h.WG.nodes(): 
            if x !=0: self.RP_h.WG.nodes[x]['pulses'] = set([int(x)])
            
        self.RP_h.PG = self.RP_h.PG.subgraph(self.sample_h-set([0]))
        self.RP_h.PG = nx.convert_node_labels_to_integers(self.RP_h.PG, first_label=1, ordering='sorted')
            
    def dual_construct_interpolators(self):
        sample_l = np.array(list(self.sample_l-set([0])))-1  
#        print(sample_l)
        sample_h = np.array(list(self.sample_h-set([0])))-1
        Vk = self.V[:,0:self.K]
        
        
        Phi_l = Vk @ np.linalg.pinv(Vk[sample_l,:])
#        Phi_l = Vk @ Vk.T[:,sample_l]
#
        Vk = self.V[:,self.K:self.N]
        Phi_h = Vk @ np.linalg.pinv(Vk[sample_h,:])
#        Phi_h = Vk @ Vk.T[:,sample_h]
        
        self.ntrpl8_l = Phi_l
        self.ntrpl8_h = Phi_h
        
        
#        Vk = self.V[:,0:self.K]
#        self.ntrpl8_l = Vk @ (Vk[sample_l,:]).T
#        Phi = Vk @ np.linalg.inv(Vk[cols,:])
#        self.ntrpl8_h = Vk @ (Vk[sample_h,:]).T      
#        self.ntrpl8_l = self.V[:,:self.K+1] @ np.linalg.pinv(self.V[sample_l,:self.K+1])
#        self.ntrpl8_h = self.V[:,:self.K+1] @ np.linalg.pinv(self.V[sample_h,:self.K+1])

            
    def dual_dpt(self):
        self.RP_l.create_feat_table()
        self.RP_l.dpt()
        self.RP_h.create_feat_table()
        self.RP_h.dpt()
        
    def dual_rpr(self,minscale,maxscale):
        self.RP_l.reconstruct_pulses_range(minscale,maxscale)
        self.RP_l.current_signal = np.around(self.RP_l.current_signal,0)     
        self.RP_h.reconstruct_pulses_range(minscale,maxscale)
        self.RP_h.current_signal = np.around(self.RP_h.current_signal,0)   
        
#        self.x_inter = (self.ntrpl8_h @ self.RP_h.current_signal) + (self.ntrpl8_l @ self.RP_l.current_signal) 
        self.x_inter = np.around((self.ntrpl8_l @ self.RP_l.current_signal) + self.x_h,0) #best so far
#        self.x_inter = (self.ntrpl8_l @ self.RP_l.current_signal) 
        #test:
#        self.x_inter = self.RP_l.current_signal + self.RP_h.current_signal
            
    def signal_to_image(self):
        return self.x_inter.reshape(self.imgShape)   
    
    def get_image_of_sampler_l(self):
        #black indicates pixels/nodes that were sampled
        out_image = np.ones(self.N)
        temp = self.sample_l
        temp = temp - set([0])
        temp = np.array(list(temp))-1
        out_image[temp] = 0
        out_image = out_image.reshape(self.imgShape)
        return out_image
                  
    def get_image_of_sampler_h(self):
        #black indicates pixels/nodes that were sampled
        out_image = np.ones(self.N)
        temp = self.sample_h
        temp = temp - set([0])
        temp = np.array(list(temp))-1
        out_image[temp] = 0
        out_image = out_image.reshape(self.imgShape)
        return out_image
            
            
            
            



        
        
        
