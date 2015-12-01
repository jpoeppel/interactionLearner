# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 23:16:14 2015
TESTSUITE to try out different models
@author: jpoeppel
"""
import numpy as np
import copy

quatZero = np.array([0.0,0.0,0.0,1.0])
quatRot = np.array([0.0,0.0,0.2326,0.9726])

trans = np.array([-0.0123,0.7464,0.05])


def getTransformationMatrix(quat, trans):
    px,py,pz = trans
    x,y,z,w = quat
    return np.matrix([[1-2*y*y-2*z*z, 2*x*y + 2*w*z, 2*x*z - 2*w*y, px],[2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x, py],
                          [2*x*z+2*w*y,2*y*z-2*w*x, 1-2*x*x-2*y*y, pz],[0.0,0.0,0.0,1.0]])
                          
def transform(pos, ori, matrix, quat):
    tmpPos = np.matrix(np.concatenate((pos,[1])))
    resPos = np.array((matrix*tmpPos.T)[:3]).flatten()

    q1 = np.copy(quat)
    q2 = np.copy(ori)
    q1[:3] *= -1

    newOri = np.zeros(4)
    newOri[0] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
    newOri[1] = -q1[0]*q2[2]+q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
    newOri[2] = q1[0]*q2[1]-q1[1]*q2[0]+q1[2]*q2[3]+q1[3]*q2[2]
    newOri[3] = -q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]+q1[3]*q2[3]
    newOri /= np.linalg.norm(newOri)
    print "Q1: {},\nQ2: {},\nres: {}".format(q1,q2,newOri)
#    tmplV = np.matrix(np.concatenate((self["linVel"],[0])))
#    self["linVel"] = np.array((matrix*tmplV.T)[:3]).flatten()
#    tmpaV = np.matrix(np.concatenate((self["angVel"],[0])))
#    self["angVel"] = np.array((matrix*tmpaV.T)[:3]).flatten()
    
    return resPos, newOri
    
def getInverse(matrix):
    invTrans = np.matrix(np.zeros((4,4)))
    invTrans[:3,:3] = matrix[:3,:3].T
    invTrans[:3,3] = -matrix[:3,:3].T*matrix[:3,3]
    invTrans[3,3] = 1.0
    return invTrans

class dummy(object):
    
    def __init__(self):
        self.vec = np.array(range(10))
        self.a = self.vec[0:1]
        self.b = self.vec[0:3]


if __name__ == "__main__":
#    transM = getTransformationMatrix(quatRot, trans)
#    inv = getInverse(transM)
#    a,b = transform(trans, quatRot, inv, quatRot)
#    q = quatRot
#    q[:3] *= -1
#    print transform(a+np.array([0.0,0.1,0.0]), b, transM, q)
    c = dummy()
    print "c.vec: ", c.vec
    print "c.a: ", c.a
    print "c.b: ", c.b
    d = copy.deepcopy(c)
    e = dummy()
    e.vec = np.copy(c.vec)
    print "d.vec: ", d.vec
    print "d.a: ", d.a
    print "d.b: ", d.b
    c.a += 10
    print "c.vec: ", c.vec
    print "c.a: ", c.a
    print "c.b: ", c.b
    d.a += 20
    print "d.vec: ", d.vec
    print "d.a: ", d.a
    print "d.b: ", d.b
    e.a += 30
    print "e.vec: ", e.vec
    print "e.a: ", e.a
    print "e.b: ", e.b
    print "c.vec: ", c.vec
    print "c.a: ", c.a
    print "c.b: ", c.b
