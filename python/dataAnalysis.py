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

colors=["green", "red", "blue", "yellow", "cyan", "magenta", "black", "burlywood", "darkgray", "aqua"]

print data[-1]

mask = np.array([0,1,2,4,5,6,7,8,9,10,12,13,14,15]) #No closing,closingDivDist
mask = np.array([0,1,2,3,4,5,6,7,8,9,10,11]) # No closing1,closing2,closing1DivDist, closing2DivDist

xs = data[::4,mask]
features = RealFeatures(xs.T)
ys = data[::4,16]
#ys = np.array([int(y) for y in ys])
labels = MulticlassLabels(ys)

k = 1

#lmnn = LMNN(features,labels, k)
#init_transform = np.eye(len(xs[0,:]))
#lmnn.set_maxiter(2000)
#lmnn.train(init_transform)

def plot_data(features, labels, ax):
    from modshogun import TDistributedStochasticNeighborEmbedding
    
    converter = TDistributedStochasticNeighborEmbedding()
    converter.set_target_dim(2)
    converter.set_perplexity(25)
    
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

