#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:38:18 2015

@author: jpoeppel
"""

import numpy as np
from modshogun import RealFeatures, MulticlassLabels
from modshogun import LMNN
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
#f = open("../data/actionVectors2", "r")
#lines = f.readlines()
#f.close()
#
#f = open("../data/actionVectors2_n", "w")
#for l in lines:
#    f.write(l[:l.rfind(";")] + "\n")
#f.close()

data = np.loadtxt("../data/actionVectors2_Clean.csv", skiprows=1)
#data = np.loadtxt("../data/actionVectors2_n", delimiter=";", skiprows=1)

colors=["green", "red", "blue", "yellow", "cyan", "magenta", "black", "burlywood", "darkgray", "aqua"]

featureNames = np.array(["sid","oid","dist","closing","contact","relPosX", "relPosY", "relPosZ",
                         "relVlX", "relVlY", "relVlZ", "closingDivDist", "closing1", "closing2", 
                         "closing1DivDist", "closing2DivDist"])
mask = np.array([0,1,2,5,6,7,8,9,10,12,13,14,15]) #No closing,closingDivDist
mask = np.array([0,1,2,3,5,6,7,8,9,10,11]) # No closing1,closing2,closing1DivDist, closing2DivDist
mask = np.array([0,1,2,3,4,5,6,8,9,11,12,13,14,15]) # No relPosz, relVelZ
mask = np.array([0,1,4,5,6,8,9,12,13,14,15])
#mask = np.array(range(16))

dists = data[:,2]
dists[dists==0.0] = 0.001
data[:,14] = data[:,12]/dists[:]
data[:,15] = data[:,13]/dists[:]
xs = data[::4,mask]


#xs[:,6:] = 0
features = RealFeatures(xs.T)
ys = data[::4,16]
#ys = np.array([int(y) for y in ys])
labels = MulticlassLabels(ys)

k = 1

lmnn = LMNN(features,labels, k)
init_transform = np.eye(features.get_num_features())
lmnn.set_diagonal(True)
lmnn.set_maxiter(20000)
#lmnn.train(init_transform)
lmnn.train(np.zeros((features.get_num_features(),features.get_num_features())))

print "lmnn transform: ", np.diag(lmnn.get_linear_transform())

statistics = lmnn.get_statistics()
pyplot.plot(statistics.obj.get())
pyplot.grid(True)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('LMNN objective')
pyplot.show()

def plot_data(features, labels, ax):
    from modshogun import TDistributedStochasticNeighborEmbedding
    
    converter = TDistributedStochasticNeighborEmbedding()
    converter.set_target_dim(2)
    converter.set_perplexity(45)
    
    embedding = converter.embed(features)
    
    x = embedding.get_feature_matrix()
    y = labels.get_labels()
    
    for i in y:
#        ax.scatter(x[0, y==int(i)],x[1, y==int(i)], x[2, y==int(i)], color=colors[int(i)])
        ax.scatter(x[0, y==int(i)],x[1, y==int(i)], color=colors[int(i)])



# get the linear transform from LMNN
#L = lmnn.get_linear_transform()
# square the linear transform to obtain the Mahalanobis distance matrix
#M = np.matrix(np.dot(L.T,L))

# represent the distance given by LMNN
#fig = pyplot.figure()
#axis = fig.add_subplot(111, projection='3d')
figure,axis = pyplot.subplots(1,1)
plot_data(features,labels, axis)
#ellipse = make_covariance_ellipse(M.I)
#axis.add_artist(ellipse)
#axis.set_title('LMNN distance')
pyplot.show()

