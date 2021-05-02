# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:40:10 2019

@author: Mark
"""
from rmpavage import rmpa
from rmpavage import graph_filter_banks

from PIL import Image
import numpy as np
import networkx as nx
import scipy.linalg as la
import scipy.signal as sig
import scipy.sparse as sparse
import copy as copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting, ignore "not-used" warning

np.set_printoptions(suppress=True)#suppress scientific notation for numpy

###############################################################################
# 1.1 Creation of a viable test image - PG 1
###############################################################################

# Create a Roadmakers Object with blank image - pg2 :
RP = rmpa.RoadmakersPavage(np.zeros([3,3]),False)
WG_nozero = copy.deepcopy(RP.WG)
WG_nozero.remove_node(0)

# Get Reference operators of blank graph:
A = nx.adjacency_matrix(WG_nozero).toarray()
Lap = nx.laplacian_matrix(WG_nozero).toarray()
L = nx.normalized_laplacian_matrix(WG_nozero)

# Get eigenspace of blank graph:
Lmax = 1.9999999999999999999998
Lmax = 2
Lam, U = la.eig(L.toarray())
Lam, U = Lam.real,U.real
idx = Lam.argsort()
Lam, U = Lam[idx], U[:,idx]
Ut = U.T
Uinv = np.linalg.inv(U)

# Create GFT signal, xhat - pg 5
xhat = np.array([-400,100,-80,50,-5,5,-5,5,-5])
xhat = np.array([-400,100,-80,50,-10,10,-10,10,-10])

# Get IGFT, x - pg 5:
x = np.around(U @ xhat)
xhat = Uinv @ x
# xhat = Ut @ x
img = np.around(x.reshape([3,3]))

###############################################################################
# 1.2 Original Roadmakers pavage algorithm performed on test image - PG 5
###############################################################################
# Initialize and complete discrete pulse transform on original image: pg 6- 16
RP = rmpa.RoadmakersPavage(img,False)
RP.create_feat_table()
RP.dpt()

#Draw the final pulse graph and stats - pg 16
plt.figure()
nx.draw_shell(RP.PG,with_labels=True)
plt.figure()
labels = nx.get_node_attributes(RP.PG,'value')
nx.draw_shell(RP.PG,with_labels=True,labels=labels)# figure with values
labels = nx.get_node_attributes(RP.PG,'scale')
nx.draw_shell(RP.PG,with_labels=True,labels=labels)# figure with scales

###############################################################################
# 1.3 Reconstruction of test image using Original Roadmakers Pavage - PG 16
###############################################################################
# Partial reconstruction of image using on scales 3+ - pg 19
RP.reconstruct_pulses_range_via_while(3,100)
plt.figure()
imgplot = plt.imshow(RP.signal_to_image(),cmap="gray")
plt.show()

###############################################################################
# 1.4 Construction of a Filter Bank for a tiny graph - PG 19
###############################################################################
# 1.4.1 Kernel, Analysis and Synthesis filters - pg 20
N = len(x)
K=5;
# Kernel and Analysis Filters - pg 20
Q = np.zeros([N,N])
Q[:K,:K] = np.eye(K)
# H0 = U @ Q @ Ut
H0 = U @ Q @ Uinv

P = np.zeros([N,N])
P[K:N,K:N] = np.eye(N-K)
# H1 = U @ Ph @ Ut
H1 = U @ P @ Uinv

xl = H0 @ x
xh = H1 @ x
xl_hat = Uinv @ xl
xh_hat = Uinv @ xh


# 1.4.3 Synthesis Filters- pg 22
inds_bottom = np.array([1,3,5,7,9])-1
inds_top = np.array([2,4,6,8])-1
xl_s = xl[inds_bottom]
xh_s = xh[inds_bottom]

Uk_l = U[:,0:K]
#Phi_l = Uk_l @ np.linalg.pinv(Uk_l[inds_bottom,:])
Phi_l = Uk_l @ np.linalg.pinv(Uk_l[inds_bottom, :])
Uk_h = U[:,K:N]
Phi_h = Uk_h @ np.linalg.pinv(Uk_h[inds_bottom, :])
#Phi_h = Uk_h @ np.linalg.pinv(Uk_h[inds_top,:])

Phi_l @ xl_s
Phi_h @ xh_s
x_interp = Phi_l @ xl_s + Phi_h @ xh_s


###############################################################################
# LABEL THIS SECTION PROPERLY
###############################################################################
#Sample bipartite set indices:
bottom_nodes, top_nodes = nx.algorithms.bipartite.sets(WG_nozero)
bottom_nodes = sorted(bottom_nodes)
top_nodes = sorted(top_nodes)
inds_bottom = np.array(list(bottom_nodes))-1
inds_top = np.array(list(top_nodes))-1
bottom_nodes = set(bottom_nodes)
top_nodes = set(top_nodes)
#print sampler:
sample_test = np.ones([N, 1],dtype='int')
sample_test[inds_bottom] = 0
sample_test = sample_test.reshape(img.shape)
plt.figure()
imgplot = plt.imshow(sample_test,cmap="gray")
plt.show()

x = np.fromiter(nx.get_node_attributes(WG_nozero,'value').values(),dtype=float)
N = len(x)






############################################################################################
#TEST DPT ON THESE SAMPLES:
width = img.shape[1]
RPl = rmpa.RoadmakersPavage(np.around(xl.reshape(img.shape),0),False)
#Add more edges:
nodes = list(RPl.WG.nodes())[1:]
nodes_matrix = np.array(nodes).reshape(img.shape)
for i in nodes:
    if ((i-width-1)>0) and ((i+width-1) % width !=0):
        RPl.WG.add_edge(i, i-width-1)
    if ((i-width+1)>0) and (i % width !=0): 
#        print(i, i % width)
        RPl.WG.add_edge(i, i-width+1)
    if (i+width-1)<len(RPl.WG) and ((i+width-1) % width !=0): 
        RPl.WG.add_edge(i, i+width-1)
    if (i+width+1)<len(RPl.WG) and (i % width !=0):
        RPl.WG.add_edge(i, i+width+1)

RPl_sample = RPl
bottom_nodes.update([0])
RPl_sample.WG = copy.deepcopy(RPl_sample.WG.subgraph(bottom_nodes))
RPl_sample.WG = nx.convert_node_labels_to_integers(RPl_sample.WG, first_label=0, ordering='sorted')
for  pulse_i in RPl_sample.WG.nodes(): 
    if pulse_i !=0: RPl_sample.WG.nodes[pulse_i]['pulses'] = set([int(pulse_i)])
RPl_sample.PG = copy.deepcopy(RPl_sample.PG.subgraph(bottom_nodes-set([0])))
RPl_sample.PG = nx.convert_node_labels_to_integers(RPl_sample.PG, first_label=1, ordering='sorted')


RPl_sample.create_feat_table()
RPl_sample.dpt(feedback=0)
scales_l = nx.get_node_attributes(RPl.PG,'scale')

plt.figure()
nx.draw_shell(RPl_sample.PG,with_labels=True)
plt.figure()
labels = nx.get_node_attributes(RPl_sample.PG,'value')
nx.draw_shell(RPl_sample.PG,with_labels=True,labels=labels)# figure with values
plt.figure()
labels = nx.get_node_attributes(RPl_sample.PG,'scale')
nx.draw_shell(RPl_sample.PG,with_labels=True,labels=labels)# figure with scales


#RPl_sample.reconstruct_full()
RPl_sample.reconstruct_pulses_range_via_while(1,150000)


xl_interp = np.zeros(x.shape)
xl_interp[inds_bottom] = RPl_sample.current_signal
xl_interp = 2*np.around(myfilter.cheby_op(R=L, c=cl, x=xl_interp, Lmax=Lmax))
imgplot = plt.imshow(xl_interp.reshape(img.shape),cmap="gray",vmin=0,vmax=255)
imgplot = plt.imshow(xl_interp.reshape(img.shape),cmap="gray")

#imgplot = plt.imshow(RPl_sample.current_signal([img.shape[0]/]),cmap="gray",vmin=0,vmax=255)
plt.show()

#############################################################################################################
#TEST DPT ON THESE HIGHFREQUENCY SAMPLES:
width = img.shape[1]
RPh = rmpa.RoadmakersPavage(np.around(xh.reshape(img.shape),0),False)
#Add more edges:
nodes = list(RPh.WG.nodes())[1:]
nodes_matrix = np.array(nodes).reshape(img.shape)
for i in nodes:
    if ((i-width-1)>0) and ((i+width-1) % width !=0):
        RPh.WG.add_edge(i, i-width-1)
    if ((i-width+1)>0) and (i % width !=0): 
#        print(i, i % width)
        RPh.WG.add_edge(i, i-width+1)
    if (i+width-1)<len(RPh.WG) and ((i+width-1) % width !=0): 
        RPh.WG.add_edge(i, i+width-1)
    if (i+width+1)<len(RPh.WG) and (i % width !=0):
        RPh.WG.add_edge(i, i+width+1)

RPh_sample = RPh
top_nodes.update([0])
RPh_sample.WG = copy.deepcopy(RPh_sample.WG.subgraph(bottom_nodes))
RPh_sample.WG = nx.convert_node_labels_to_integers(RPh_sample.WG, first_label=0, ordering='sorted')
for  pulse_i in RPh_sample.WG.nodes(): 
    if pulse_i !=0: RPh_sample.WG.nodes[pulse_i]['pulses'] = set([int(pulse_i)])
RPh_sample.PG = copy.deepcopy(RPh_sample.PG.subgraph(bottom_nodes-set([0])))
RPh_sample.PG = nx.convert_node_labels_to_integers(RPh_sample.PG, first_label=1, ordering='sorted')


RPh_sample.create_feat_table()
RPh_sample.dpt()


plt.figure()
nx.draw_shell(RPh_sample.PG,with_labels=True)
plt.figure()
labels = nx.get_node_attributes(RPh_sample.PG,'value')
nx.draw_shell(RPh_sample.PG,with_labels=True,labels=labels)# figure with values
plt.figure()
labels = nx.get_node_attributes(RPh_sample.PG,'scale')
nx.draw_shell(RPh_sample.PG,with_labels=True,labels=labels)# figure with scales





RPh_sample.reconstruct_full()
RPh_sample.reconstruct_pulses_range_via_while(1,5000)

xh_interp = np.zeros(x.shape)
xh_interp[inds_top] = RPh_sample.current_signal
xh_interp = 2*np.around(myfilter.cheby_op(R=L, c=ch, x=xh_interp, Lmax=Lmax)) 
imgplot = plt.imshow(xh_interp.reshape(img.shape),cmap="gray")
imgplot = plt.imshow(xh_interp.reshape(img.shape),cmap="gray",vmin=0,vmax=255)


plt.show()


scales_h = nx.get_node_attributes(RPh.PG,'scale')



##############################################################################
# Code to use later onwards:

#
##Build Tokhonov Regularizer:
#mul = 0.05
#Lam_inv_wl = 1 + mul*Lam#inverse of weighted eigenvalues
#Lam_inv_wl = 1/Lam_inv_wl
#Tikhonov_l = U @ np.diag(Lam_inv_wl) @ U.T
#
#
#muh = 0.1
#Lam_inv_wh = 1 + muh*Lam#inverse of weighted eigenvalues
#Lam_inv_wh = 1/Lam_inv_wh
#Tikhonov_h = U @ np.diag(Lam_inv_wh) @ U.T
#
#mub = 0.1
#Lam_inv_wb = 1 + mub*Lam#inverse of weighted eigenvalues
#Lam_inv_wb = 1/Lam_inv_wb
#Tikhonov_b = U @ np.diag(Lam_inv_wb) @ U.T

#test = U@U.T


#Lmax = (sparse.linalg.eigs(L,k=1,return_eigenvectors=False)).real[0]


# Chebychev stuff:
myfilter = graph_filter_banks.GraphFilter(L)
cl = myfilter.cheby(w=1,m=200,Lmax=Lmax,i=0)
ch = myfilter.cheby(w=1,m=200,Lmax=Lmax,i=1)
xl = myfilter.cheby_op(R=L, c=cl, x=x, Lmax=Lmax)
xh = myfilter.cheby_op(R=L, c=ch, x=x, Lmax=Lmax)
xl = np.around(myfilter.cheby_op(R=L, c=cl, x=x, Lmax=Lmax))
xh = np.around(myfilter.cheby_op(R=L, c=ch, x=x, Lmax=Lmax))



xtest = xl+xh
plt.imshow(xtest.reshape(img.shape),cmap='gray')
np.sqrt(((xtest-x)**2).mean())#mse

#sample each vector:
xl_sample = xl[inds_bottom]
xh_sample = xh[inds_bottom]

#xl_sample = xl[inds_top]
#xh_sample = xh[inds_top]#
#xl_interp = np.around(myfilter.cheby_op(R=L, c=cl, x=xl_sample, Lmax=Lmax))
#xh_interp = np.around(myfilter.cheby_op(R=L, c=ch, x=xh_sample, Lmax=Lmax))
#using exact instead:




x_interp = 2*(xl_interp + xh_interp)
imgplot = plt.imshow(x_interp.reshape(img.shape),cmap="gray",vmin=0,vmax=255)
imgplot = plt.imshow(img,cmap="gray",vmin=0,vmax=255)

imgplot = plt.imshow(2*xl_interp.reshape(img.shape),cmap="gray")
imgplot = plt.imshow(2*xh_interp.reshape(img.shape),cmap="gray")

(abs(x_interp-x)).mean()
np.sqrt(((x_interp-x)**2).mean())

#Perform DPT on Full Graph:
RP.create_feat_table()
RP.dpt(1)
#Low Scales:
RP.reconstruct_pulses_range_via_while(5000,135500)
#RP.reconstruct_full()
plt.imshow(RP.signal_to_image(),cmap="gray")
#TEST OTHER FILTERS:







#test reconstruction from both graphs:
RPl.reconstruct_pulses_range(100,2000)
#RPl.reconstruct_pulses_range(1,100)
xl_dptinter = RPl_sample.current_signal
RPh.reconstruct_pulses_range(5,14)
#RPh.reconstruct_pulses_range(1,100)
xh_dptinter = RPh_sample.current_signal
print("xl_dpt_inter as is:")
imgplot = plt.imshow((Phi_l @xl_dptinter).reshape(img.shape),cmap="gray",vmin=0,vmax=255)
plt.show()
print("xl_dpt_inter  with Tikhonov:")
imgplot = plt.imshow((Phi_l @xl_dptinter  @ Tikhonov_l).reshape(img.shape),cmap="gray",vmin=0,vmax=255)
plt.show()
print("xh_dpt_inter as is:")
imgplot = plt.imshow((Phi_h @xh_dptinter).reshape(img.shape),cmap="gray",vmin=0,vmax=255)
plt.show()
print("xh_dpt_inter with Tikhonov:")
imgplot = plt.imshow((Phi_h @xh_dptinter @ Tikhonov_h).reshape(img.shape),cmap="gray",vmin=0,vmax=255)
plt.show()
#full signal:
print("Both signal as is:")
x_inter = (Phi_l @ xl_dptinter )+ (Phi_h @ xh_dptinter )
imgplot = plt.imshow(x_inter.reshape(img.shape),cmap="gray",vmin=0,vmax=255)
plt.show()
print("Both signal with Tikhonov:")
x_inter = (Phi_l @ xl_dptinter @ Tikhonov_l)+ (Phi_h @ xh_dptinter @ Tikhonov_h)
imgplot = plt.imshow(x_inter.reshape(img.shape),cmap="gray",vmin=0,vmax=255)
plt.show()

print("Both signal with Tikhonov after sum:")
x_inter = ((Phi_l @ xl_dptinter) + (Phi_h @ xh_dptinter)) @Tikhonov_b
imgplot = plt.imshow(x_inter.reshape(img.shape),cmap="gray",vmin=0,vmax=255)
plt.show()

print('rmse: ', np.sqrt(((x_inter-x)**2).mean()))






