""" Graph Data Structure Classes for the Roadmakers Pavage Algorithm """
__author__ = "Mark de Lancey"
__copyright__ = "Copyright 2019, The University of Pretoria"
__version__ = "0.2.0"
__email__ = "mark.stephen.del@gmail.com"
__status__ = "1.0"

# Import Packages required
import numpy as np
import networkx as nx
from scipy.ndimage import generic_filter as imfilt

class WorkingGraph(nx.Graph):
    #Functions already useable as inherited:
    #add_edge
    #add_edges_from in place of add_edges
    def __init__(self,Image=None,Connect=False):
        super().__init__()
        #add initial node with value 0, scale inf and an empty set of neightbours and pulses
        self.add_node(value=0,scale=np.inf,pulses=set())
        #add nodes from Image if one was given:
        if Image is not None:        
            Image = np.array(Image,"float32")
            self.rows = Image.shape[0]
            self.cols = Image.shape[1]
            Data = Image.reshape(self.rows*self.cols)
            self.add_nodes(Data)
            if Connect==True:
                self.connect_adjacent()
    
    # Function to add a single node to the graph. 
    def add_node(self ,value ,scale = 1,pulses = None):
        ID = len(self)
        pulses = set([int(ID)]) if pulses == None else pulses
        # Adds a node with the given attributes using the superclass:
        super().add_node(ID,value=value,scale=scale,pulses=pulses)

    #adds multiple nodes to the graph
    def add_nodes(self, values, scales = None,pulses = None):
        scales = [1]*len(values) if scales==None else scales
        pulses = [None]*len(values) if pulses==None else pulses
#        nodes = list(zip(IDs,values,scales,neighbors,pulses))
#        IDs = np.arange(len(self),len(values)+len(self))
#        print(type(IDs))
#        keys=('value','scale','pulses')
#        nodes = {ID:dict.fromkeys(keys,values,scales,set(IDs)) for ID in IDs}        
#        nodes = {ID:{key:[values[ID],scales[ID],set(ID)] for key in keys} for ID in IDs}
#        nodes = {ID:dict.fromkeys(keys,values)}
#        d = {key:value for key, value in zip(keys, values)}               
#        nodes = enumerate(zip(values,scales,pulses), start = 1)
#        super().add_nodes_from(nodes) #PROBABLY THE BETTER WAY TO AVOID FUNCTION OVERHEAD
        [self.add_node(values[n], scales[n],pulses[n]) for n in range(len(values))]
        
    def replace_pulse(self,ID,pulse):
        self.node[ID]['pulses'] = set([int(pulse)]) 
       
#     Combines neighbors of the given node if they are equal in value:
    def cluster_identical(self, node):
#        if node !=0:#do not cluster the padding node!!
            # Find indices of all neighbors of node with identical values: 
    #        if (self.has_node(node)): <- Unnessessary test
#            J = [i for i in self.neighbors(node) if (self.nodes[i]['value']==self.node[node]['value'] and i!=node and i!=0)]
            J = [i for i in self.neighbors(node) if (self.nodes[i]['value']==self.nodes[node]['value'])]

            if len(J) != 0:
                J = int(J[0])
                self.nodes[J]['scale'] += self.nodes[node]['scale']
                # Add the edges of node node to node J.
                super().add_edges_from([(i,J) for i in self.neighbors(node) if i!=J])                
                self.nodes[J]['pulses'].update(self.nodes[node]['pulses'])
                self.remove_node(node)
                return True
            return False

    # Combines all neighbouring nodes with same values in the working graph.
    def cluster_identical_all(self):
        # COmbine multiple nodes with same values
        [self.cluster_identical(node) for node in list(self.nodes())[1:]]
        
#    Add edges to the graph using 4-connectivity:
    def connect_adjacent(self,Image=None):
        if Image is not None:  
            self.rows = Image.shape[0]
            self.cols = Image.shape[1]
        
        Indices = np.zeros((self.rows+2,self.cols+2),'int64')
        Node_Index = np.arange(1,self.rows*self.cols+1)
        Indices[1:-1, 1:-1] = Node_Index.reshape((self.rows,self.cols))
        Edges = list()
        for i in range(4):
            # Use the imfilt(ndimage.generic_filter) function loaded from the
            # Scipy library to scan through the image andd find the i-th
            # neighbour for each node.
            Neighbourhood = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])
            Neighbours = imfilt(Indices,lambda x:x[i],footprint=Neighbourhood)
            # Extract the Neighbours from the results of the filter
            Neighbours = Neighbours[1:-1, 1:-1]
            # Reshape the Neighbours array to one dimension.
            New_size = (Neighbours.shape[0]*Neighbours.shape[1])
            Neighbours = Neighbours.reshape(New_size)
            # Add the edges corresponding to these Neighbours to the edges
            # list.
            Edges += [(i+1, int(Neighbours[i]))
                        for i in range(len(Neighbours))]
            super().add_edges_from(Edges)
    
                    
# Create PulseGraph class, this class will be used as a data type for the
# Pulse Graph (PG) in the Roadmakers Pavage Algorithm
class PulseGraph(nx.DiGraph):
    # Initialise the Pulse Graph (The pulse graph is a graph where the
    # nodes represents the pulses extracted by the Discrete Pulse Transform.)
    def __init__(self,size=None):
        super().__init__()
        if size is not None:
            self.add_nodes(size)
        
        # Function to add a single node to the graph. 
    def add_node(self ,value ,scale = 0,pulses = None):
        ID = len(self)+1
        # Adds a node with the given attributes using the superclass:
        super().add_node(ID,value=value,scale=scale)
 
    #adds multiple nodes to the graph
    def add_nodes(self, values, scales = None):
        scales = [0]*len(values) if scales==None else scales
        [self.add_node(values[n], scales[n]) for n in range(len(values))]
        