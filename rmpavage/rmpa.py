""" RoadmakersPavage Class (Governs the graphs and functions that solves the DPT problem
    using the Roadmakers Pavage Algorithm"""
__author__ = "Mark de Lancey"
__copyright__ = "Copyright 2019, The University of Pretoria"
__version__ = "0.2.0"
__email__ = "mark.stephen.del@gmail.com"
__status__ = "1.0"

# Import Packages required
import copy
import networkx as nx
import rmpavage.graphclasses as Graph
import numpy as np
import time as time

# Roadmakers_Pavage class
class RoadmakersPavage:
    # Initialise the Roadmakers Pavage process with a greyscale image.
    def __init__(self, Image, create_backup=True):        
        t = time.process_time()
        self.imgShape = Image.shape
        self.numPixels = Image.shape[0]*Image.shape[1]
        self.WG = Graph.WorkingGraph(Image,Connect=True)#initialize a working graph and connect the nodes
        self.PG = Graph.PulseGraph(np.zeros(Image.shape[0]*Image.shape[1]))# Initialise the pulse graph with data nodes of scale 1 and value 0
        self.feat_table = {}
        self.current_signal = 0;
        if create_backup:
            self.WG_bak = copy.deepcopy(self.WG)#make a backup of the WG
            self.PG_bak = copy.deepcopy(self.PG)#make a backup of the PG
        elapsed_time = time.process_time() - t
        print("CPU time to initialize RMPA object: ", elapsed_time)
        
        self.time_feat_table = 0.0
        self.time_dpt = 0.0
        self.time_recon = 0.0
            
        
    #load the backup of the WG into the usable WG in case something went wrong:
    def loadBackUp_WG(self):
        self.WG = copy.deepcopy(self.WG_bak)
        
    #load the backup of the WG into the usable WG in case something went wrong:
    def loadBackUp_PG(self):
        self.PG = copy.deepcopy(self.PG_bak)

    def create_feat_table(self,ignore_padding=True):
        t = time.process_time()
        for node in set(self.WG.nodes()):
            # Test if the pulse is a min/maxfeature and place it in the feature dictionary if it is one:
            if ignore_padding:
                diffs = [self.WG.nodes[node]['value']-self.WG.nodes[n]['value'] for n in self.WG.neighbors(node) if n!=0]
            else:
                diffs = [self.WG.nodes[node]['value']-self.WG.nodes[n]['value'] for n in self.WG.neighbors(node)]
            
            if 0 in diffs:#if it is not unique then cluster
                self.WG.cluster_identical(node)#let the identical neighbor inherit features of 'node' and delete 'node'
            elif all(item > 0 for item in diffs):#is local maximum
                self.feat_table[node] = {'scale':self.WG.nodes[node]['scale'],'value':self.WG.nodes[node]['value'],'isMax':True}
            elif all(item < 0 for item in diffs):#is local minumum
                self.feat_table[node] = {'scale':self.WG.nodes[node]['scale'],'value':self.WG.nodes[node]['value'],'isMax':False}
#        node = 0
        elapsed_time = time.process_time() - t
        print("CPU time to create feature table: ", elapsed_time)
        self.time_feat_table = elapsed_time
        
    # Function to add a node as a pulse in the Pulse Graph:
    def add_pulse(self, node):     
#        diffs = [self.WG.node[node]['value']-self.WG.node[n]['value'] for n in self.WG.neighbors(node)]
        diffs = [self.WG.nodes[node]['value']-self.WG.nodes[n]['value'] for n in self.WG.neighbors(node) if n !=0]
        if not diffs:#if there are no neighbors not equal to the zeroth node:
            if (len(self.WG) ==1):#if this is the zeroth node:
                del self.feat_table[node]
                self.PG.add_node(0,self.WG.nodes[node]['scale'])#value, scale
                self.PG.add_edges_from([(i,len(self.PG)) for i in self.WG.nodes[node]['pulses']])
                self.WG.remove_node(node)
            else:
                nearest_node = 0
                self.PG.add_node(self.WG.nodes[node]['value'],self.WG.nodes[node]['scale']) # Add a new node to PG with (value j,  scale equal to node's scale)
                self.PG.add_edges_from([(i,len(self.PG)) for i in self.WG.nodes[node]['pulses']])# Add edges from this new Pulse node to all the pulse Nodes linked to the specified node.
                self.WG.nodes[nearest_node]['pulses'].update(set([int(len(self.PG))]))# replace the entire set of pulses in "node" with only the reference to the newly formed pulse
                self.WG.nodes[nearest_node]['scale'] += self.WG.nodes[node]['scale']
                self.WG.add_edges_from([(i,nearest_node) for i in self.WG.neighbors(node) if i!=nearest_node])# Add the edges of node node to node J.
                temp_dict = self.feat_table.pop(node)# create a new element in the feature table
                temp_dict['value'] = self.WG.nodes[nearest_node]['value']
                temp_dict['scale'] = self.WG.nodes[nearest_node]['scale']
                self.feat_table[nearest_node] = temp_dict
                self.WG.remove_node(node)
        else:                
            j = np.abs(diffs).argmin()    
#            nearest_node = list(self.WG.neighbors(node))[j]
            nearest_node = [n for n in self.WG.neighbors(node) if n !=0][j]
            j = diffs[j]
            self.PG.add_node(j,self.WG.nodes[node]['scale']) # Add a new node to PG with (value j,  scale equal to node's scale)
            self.PG.add_edges_from([(i,len(self.PG)) for i in self.WG.nodes[node]['pulses']])# Add edges from this new Pulse node to all the pulse Nodes linked to the specified node.
            self.WG.nodes[nearest_node]['pulses'].update(set([int(len(self.PG))]))# replace the entire set of pulses in "node" with only the reference to the newly formed pulse
            self.WG.nodes[nearest_node]['scale'] += self.WG.nodes[node]['scale']
            self.WG.add_edges_from([(i,nearest_node) for i in self.WG.neighbors(node) if i!=nearest_node])# Add the edges of node node to node J.
            temp_dict = self.feat_table.pop(node)# create a new element in the feature table
            temp_dict['value'] = self.WG.nodes[nearest_node]['value']
            temp_dict['scale'] = self.WG.nodes[nearest_node]['scale']
            self.feat_table[nearest_node] = temp_dict
            self.WG.remove_node(node)
        
        
    def check_feature(self, node):
        # Find indices of all neighbors of node with identical values: 
        nearest_node = [i for i in self.WG.neighbors(node) if (self.WG.nodes[i]['value']==self.WG.nodes[node]['value'])]
        if len(nearest_node) != 0:
            nearest_node = int(nearest_node[0])
            self.WG.nodes[nearest_node]['scale'] += self.WG.nodes[node]['scale']
            # Add the edges of node node to node J.
            self.WG.add_edges_from([(i,nearest_node) for i in self.WG.neighbors(node) if i!=nearest_node])                
            self.WG.nodes[nearest_node]['pulses'].update(self.WG.nodes[node]['pulses'])
            self.WG.remove_node(node)            
            #remove and replace in feature table:
            temp_dict = self.feat_table.pop(node)
            temp_dict['value'] = self.WG.nodes[nearest_node]['value']
            temp_dict['scale'] = self.WG.nodes[nearest_node]['scale']       
            self.feat_table[nearest_node] = temp_dict
            return False#not a feature
        else:
            diffs = [self.WG.nodes[node]['value']-self.WG.nodes[n]['value'] for n in self.WG.neighbors(node) if n!=0]
            if all(item > 0 for item in diffs):#is local maximum
                self.feat_table[node]['isMax'] = True
                return True
            elif all(item < 0 for item in diffs):
                self.feat_table[node]['isMax'] = False
                return True
            else:
                del self.feat_table[node]#remove the node from the dictionary
                return False
        print("something went wrong")
    

    # Roadmakers pavage:
    def dpt(self,feedback=0):
        #feedback levels:
        # 3 - everything, every loop, all progress
        # 2 - each iteration of while loop
        # 1 - only when the feature table changes
        # 0 - stfu, recommended for simulations
        t = time.process_time()
        scale_cur = 1
        ft_size = len(self.feat_table)
        ft_size_prev = len(self.feat_table)+500
        while len(self.feat_table) > 0:#loop until the feature_table is empty    
#            print(self.feat_table)
            if (feedback == 1) and (ft_size_prev != ft_size):
                print("Scale: ", scale_cur,"FT: ",len(self.feat_table), "WG: ", len(self.WG), "PG: ", len(self.PG))
            ft_size_prev = len(self.feat_table)
            node_inds = dict((x,self.feat_table[x]) for x in self.feat_table if self.feat_table[x]['scale'] ==scale_cur)            
            if feedback >= 2:                
                print("Scale: ", scale_cur,"FT: ",len(self.feat_table), "WG: ", len(self.WG), "PG: ", len(self.PG), "node_inds: ", len(node_inds))  
            isPulse = False   
            maxfeats = {x:self.feat_table[x]['isMax'] for x in node_inds if x in self.feat_table and self.feat_table[x]['isMax']} 
            # print(maxfeats)
            count = 0             
            for node in node_inds: 
                if (feedback == 3):
                    count += 1
                    if count % 1000 == 0:
                        print("Inner for loop progress: ", count, " of ", len(node_inds) )
                isPulse = False
                if (self.WG.has_node(node)): 
                    if self.WG.nodes[node]['scale']==scale_cur:# Ensure the node of still of the current scale.
                        if self.check_feature(node):#check if the node in the feature table is still a feature
                            isPulse = False
                            if self.feat_table[node]['isMax']:#if current node is a max feature..
                                isPulse = True#...then it is automatically a pulse...
                            else:#...otherwise if it is a min feature...
                                if not maxfeats:#and there are no max features of current scale...
                                    isPulse = True#..only then is it a pulse
                                else:
                                    isPulse = False
                        else:#else if the node was clustered...
                            isPulse = False#then it is no longer a pulse
                        if isPulse:
                            self.add_pulse(node)
                            maxfeats.pop(node,None)
            #END FOR
            #set the scale to the smallest one currently in the table:
            scales = np.array([self.feat_table[x]['scale'] for x in self.feat_table])
            if scales.any(): 
                scale_cur = min(scales)
                ft_size = len(self.feat_table)
        elapsed_time = time.process_time() - t
        print("CPU time to perform RMPA: ", elapsed_time)        
        self.time_dpt = elapsed_time
        if len(self.WG) >0:
            print("Warning - Working graph still has elements, please check indexing if the graph was edited manually without using rmpa object functions")
        #END WHILE
        
        self.values = nx.get_node_attributes(self.PG,'value')
        self.scales = nx.get_node_attributes(self.PG,'scale')
        
    def dpt_old(self,feedback=0):
        #feedback levels:
        # 3 - everything, every loop, all progress
        # 2 - each iteration of while loop
        # 1 - only when the feature table changes
        # 0 - stfu, recommended for simulations
        t = time.process_time()
        scale_cur = 1
        ft_size = len(self.feat_table)
        ft_size_prev = len(self.feat_table)+500
        while len(self.feat_table) > 0:#loop until the feature_table is empty          
            if (feedback == 1) and (ft_size_prev != ft_size):
                print("Scale: ", scale_cur,"FT: ",len(self.feat_table), "WG: ", len(self.WG), "PG: ", len(self.PG))
            ft_size_prev = len(self.feat_table)
            node_inds = dict((x,self.feat_table[x]) for x in self.feat_table if self.feat_table[x]['scale'] ==scale_cur)            
            if feedback >= 2:                
                print("Scale: ", scale_cur,"FT: ",len(self.feat_table), "WG: ", len(self.WG), "PG: ", len(self.PG), "node_inds: ", len(node_inds))  
            isPulse = False   
            count = 0             
            for node in node_inds: 
                if (feedback == 3):
                    count += 1
                    if count % 1000 == 0:
                        print("Inner for loop progress: ", count, " of ", len(node_inds) )
                isPulse = False
                if (self.WG.has_node(node)): 
                    if self.WG.node[node]['scale']==scale_cur:# Ensure the node of still of the current scale.
                        if self.check_feature(node):#check if the node in the feature table is still a feature
                            isPulse = False
                            if self.feat_table[node]['isMax']:#if current node is a max feature..
                                isPulse = True#...then it is automatically a pulse...
                            else:#...otherwise if it is a min feature...
                                #THIS NEXT LINE IS SLOW - ROOM FOR IMPROVEMENT:
#                                maxfeats = [self.feat_table[x]['isMax'] for x in node_inds if x in self.feat_table and self.feat_table[x]['isMax']]
                                maxfeats = [self.feat_table[x]['isMax'] for x in node_inds if x in self.feat_table]
                                if sum(maxfeats) == 0:#and there are no max features of current scale...
                                    isPulse = True#..only then is it a pulse
                                else:
                                    isPulse = False
                        else:#else if the node was clustered...
                            isPulse = False#then it is no longer a pulse
                        if isPulse:
                            self.add_pulse(node)
            #END FOR
            #set the scale to the smallest one currently in the table:
            scales = np.array([self.feat_table[x]['scale'] for x in self.feat_table])
            if scales.any(): 
                scale_cur = min(scales)
                ft_size = len(self.feat_table)
        elapsed_time = time.process_time() - t
        print("CPU time to perform RMPA: ", elapsed_time)
        if len(self.WG) >0:
            print("Warning - Working graph still has elements, please check indexing if the graph was edited manually without using rmpa object functions")
        #END WHILE
        
        self.values = nx.get_node_attributes(self.PG,'value')
        self.scales = nx.get_node_attributes(self.PG,'scale')


    def signal_to_image(self):
        return self.current_signal.reshape(self.imgShape)   
        
    
    def reconstruct_full(self):
        t = time.process_time()
        signal_len = len([x for x in self.PG.nodes() if self.PG.nodes[x]['scale']==0])
        self.current_signal = np.zeros(signal_len)
        for i in range(0, signal_len):
            path = nx.all_simple_paths(self.PG,source=i+1,target=len(self.PG))
            path = set(list(path)[0])#the in operator performs much faster on sets than lists
            self.current_signal[i] = sum([self.PG.nodes[x]['value'] for x in path])
        elapsed_time = time.process_time() - t
        print("CPU time to reconstruct full signal: ", elapsed_time)
 
    #extracts pulses within the given range, "scales":
    def reconstruct_pulses_range(self,minscale,maxscale):
        t = time.process_time()
        signal_len = len([x for x in self.PG.nodes() if self.PG.nodes[x]['scale']==0])
        self.current_signal = np.zeros(signal_len)
        values = nx.get_node_attributes(self.PG,'value')
        scales = nx.get_node_attributes(self.PG,'scale')
        for i in range(0, signal_len):
            path = nx.all_simple_paths(self.PG,source=i+1,target=len(self.PG))
            path = set(list(path)[0])
            self.current_signal[i] = sum([values[x] for x in path if minscale <= scales[x] <= maxscale])
        elapsed_time = time.process_time() - t
        print("CPU time to reconstruct partial signal: ", elapsed_time)

    def reconstruct_pulses_range_via_while(self,minscale,maxscale,output=True):
        t = time.process_time()
        signal_len = len([x for x in self.PG.nodes() if self.PG.nodes[x]['scale']==0])
        self.current_signal = np.zeros(signal_len)
        for i in range(0, signal_len):
            end = False             
            x = list(nx.neighbors(self.PG,i+1))[0]
            while not end:
                if self.scales[x]  <= maxscale:
                    if self.scales[x] >= minscale:
                        self.current_signal[i] += self.values[x]
                else:
                    end = True                
                x = list(nx.neighbors(self.PG,x))
                if x:
                    x = x[0]
                else:
                    end=True
                
                
        elapsed_time = time.process_time() - t
        if output:
            print("CPU time to reconstruct partial signal: ", elapsed_time)
        self.time_recon = elapsed_time
            