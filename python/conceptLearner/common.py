#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:31:07 2015

@author: jpoeppel
"""

import numpy as np
import math

NUMDEC = 4
GAZEBOCMDS = { "NOTHING": 0,"MOVE": 1, "GRAB": 2, "RELEASE": 3}

GRIPPERSTATES = {"OPEN":0, "CLOSED": 1}

def quaternionToEuler(quat):
    assert len(quat)== 4, "quat should be a x,y,z,w quaternion"
    quat /= np.linalg.norm(quat)
    x,y,z,w = quat
    squ = w**2
    sqx = x**2
    sqy = y**2
    sqz = z**2
    
    res = np.zeros(3)
    res[0] = math.atan2(2 * (y*z + w*x), squ -sqx-sqy+sqz)
    sarg = -2.0 * (x*z-w*y)
    if sarg <= -1.0:
        res[1] = -0.5*math.pi
    elif sarg >= 1.0:
        res[1] = 0.5*math.pi
    else:
        math.asin(sarg)
    res[2] = math.atan2(2* (x*y + w*z), squ+sqx-sqy-sqz)
    
    return res
    
def eulerToQuat(euler):
    assert len(euler) == 3, "Euler represents roll, pitch, yaw angles (in radians)"
    phi, the, psi = euler*0.5
    res = np.zeros(4)
    cos = math.cos
    sin = math.sin
    
    res[0] = sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi)
    res[1] = cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi)
    res[2] = cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi)
    res[3] = cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi)
    res /= np.linalg.norm(res)
    return res
    
def quatPosToTransformation(quat, pos):
    px,py,pz = pos
    x,y,z,w = quat
    return np.matrix([[1-2*y*y-2*z*z, 2*x*y + 2*w*z, 2*x*z - 2*w*y, px],[2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x, py],
                      [2*x*z+2*w*y,2*y*z-2*w*x, 1-2*x*x-2*y*y, pz],[0.0,0.0,0.0,1.0]])
                      
def eulerPosToTransformation(euler, pos):
    b,g,a = euler
    px,py,pz = pos
    ca = math.cos(a)
    cb = math.cos(b)
    cg = math.cos(g)
    sa = math.sin(a)
    sg = math.sin(g)
    sb = math.sin(b)
#    return np.matrix(np.round([[ca*cg-sa*cb*sg, sa*cg+ca*cb*sg, sb*sg, px], 
#                      [-ca*sg-sa*cb*cg, -sa*sg+ca*cb*cg, sb*cg, py],
#                      [sa*sb, -ca*sb, cb, pz],
#                      [0.0, 0.0, 0.0, 1.0]],NUMDEC))
    return np.matrix([[cb*ca, sg*sb*ca-cg*sa, cg*sb*ca+sg*sa, px],
                               [cb*sa, sg*sb*sa+cg*ca, cg*sb*sa-sg*ca, py],
                               [-sb, sg*cb, cg*cb, pz],
                               [0.0,0.0,0.0,1.0]])
                      
def invertTransMatrix(matrix):
    invTrans = np.matrix(np.zeros((4,4)))
    invTrans[:3,:3] = matrix[:3,:3].T
    invTrans[:3,3] = -matrix[:3,:3].T*matrix[:3,3]
    invTrans[3,3] = 1.0
    return invTrans
    
if __name__=="__main__":
    testEuler = np.array([0.0,0.0,math.pi*0.5])
    worldOri = np.array([0.0,0.0,0.7071,0.7071])
    worldEuler = quaternionToEuler(worldOri)
    print "Euler: ", worldEuler
    worldPos = np.array([-0.5,0.0,0.03])
    gPos = np.array([-0.3,0.0,0.0,1.0])
    gOri= np.array([0.0,0.0,0.0,1.0])
    gEuler = quaternionToEuler(gOri)
    print "gEuler before: ", gEuler
    translation = np.array([0.0,0.04,0.0])
#    transM = quatPosToTransformation(worldOri, worldPos)
#    print "transM: ", transM
    transMEuler = eulerPosToTransformation(worldEuler, worldPos)
    print "trans from euler: ", transMEuler
    inv = invertTransMatrix(transMEuler)
    print "inv: ", inv
#    transPos = np.matrix(gPos)
    transGPos = np.array((inv*np.asmatrix(gPos).T)[:3]).flatten()
    print "transGPos: ", transGPos
    print "transGEuler: ", gEuler-worldEuler
    print "transGOri: ", eulerToQuat(gEuler-worldEuler)
    transBPos = np.concatenate((worldPos,[1]))
    transBPos = np.array((inv*np.asmatrix(transBPos).T)[:3]).flatten()
    print "transBPos: ", transBPos
    print "transBEuler: ", worldEuler-worldEuler
    print "transBOri: ", eulerToQuat(worldEuler-worldEuler)
    
    transGPos += translation
    print "translated GPos: ", transGPos
    translatedGPos = np.concatenate((transGPos,[1]))
    print "backTransformation of translated GPos: ", np.array((transMEuler*np.asmatrix(translatedGPos).T)[:3]).flatten()
    
    print "___________________________"
    rawAngVel = np.array([0.0,-0.1,-0.6,0.0])
    print "tranformed AngVel: ", np.array((inv*np.asmatrix(rawAngVel).T)[:3]).flatten()

#    print "eulerToQuat: {}".format(eulerToQuat(testEuler))
#    print "quatToEuker: {}".format(quaternionToEuler(trueQuat))