#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:31:07 2015

@author: jpoeppel
"""

import numpy as np
from numpy import dot as npdot
import math

from operator import itemgetter
from config import config

GAZEBOCMDS = { "NOTHING": 0,"MOVE": 1, "GRAB": 2, "RELEASE": 3}
SIDE = {"NONE": 0, "DOWN": 1, "UP": 2}

GRIPPERSTATES = {"OPEN":0, "CLOSED": 1}


        
    

def quaternionToEuler(quat):
    """
        Function to compute the euler angles around the x,y,z axes of a rotation given by a quaternion
        
        Parameters
        ---------
        quat: np.array(4)
            The quaternion representing the rotation
            
        Returns
        -------
        np.array(3)
            The angles around the x,y,z axes respectively
        
    """
    assert len(quat)== 4, "quat should be a x,y,z,w quaternion"
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.zeros(3)
    quat /= norm
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
    """
        Function to compute the rotation quaternion from angles around the x,y,z axes
        
        Paramters
        ---------
        euler: np.array(3)/np.array(1)/float
            The rotation angle(s) around the axis. If only one value is given, it is assumed to be around the z axis.
            
        Returns
        -------
        np.array(4)
            Quaternion representing the rotation
    """
    if hasattr(euler, "__len__"):
        if len(euler) == 1:
            phi = 0.0
            the = 0.0
            psi = euler[0]*0.5
        else:
            assert len(euler) == 3, "Euler represents roll, pitch, yaw angles (in radians): {}".format(euler)
            phi, the, psi = euler*0.5
    else:
        phi = 0.0
        the = 0.0
        psi = euler*0.5
    res = np.zeros(4)
    cos = math.cos
    sin = math.sin
    
    res[0] = sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi)
    res[1] = cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi)
    res[2] = cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi)
    res[3] = cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi)
    res /= np.linalg.norm(res)
    return res
    
                      
def eulerPosToTransformation(euler, pos):
    if hasattr(euler, "__len__"):
        if len(euler) == 1:
            b = 0.0
            g = 0.0
            a = euler[0]
        else:
            assert len(euler) == 3, "Euler represents roll, pitch, yaw angles (in radians): {}".format(euler)
            b, g, a = euler
    else:
        b = 0.0
        g = 0.0
        a = euler
    if len(pos) == 2:
        return eulerPosToTransformation2d(a, pos)
    px,py,pz = pos
    ca = math.cos(a)
    cb = math.cos(b)
    cg = math.cos(g)
    sa = math.sin(a)
    sg = math.sin(g)
    sb = math.sin(b)
    return np.array([[cb*ca, sg*sb*ca-cg*sa, cg*sb*ca+sg*sa, px],
                               [cb*sa, sg*sb*sa+cg*ca, cg*sb*sa-sg*ca, py],
                               [-sb, sg*cb, cg*cb, pz],
                               [0.0,0.0,0.0,1.0]])
                      
                      
def eulerPosToTransformation2d(euler, pos):
    a = euler
    px,py = pos
    ca = math.cos(a)
    sa = math.sin(a)
    return np.array([[ca, -sa, px],
                      [sa, ca, py],
                      [0.0,0.0,1.0]])

def invertTransMatrix(matrix):
    r,c = np.shape(matrix)
    invTrans = np.zeros((r,c))
    r -= 1
    c -= 1
    matrixRot = matrix[:r,:c]
    invTransRot = invTrans[:r,:c]
    invTransRot[:,:] = np.copy(matrixRot.T)
    invTrans[:r,c] = npdot(-matrixRot.T,matrix[:r,c])
    invTrans[r,c] = 1.0
    return invTrans                      
                                          
                     
def dist(center, edge1, edge2, ang, ref):
#    if len(edge1) == 2 and len(ref) == 3:
#        edge1 = np.concatenate((edge1,[0]))
#    if len(edge2) == 2 and len(ref) == 3:
#        edge2 = np.concatenate((edge2,[0]))    
    ca = math.cos(ang)
    sa = math.sin(ang)
#    r = np.array([[ca, -sa, 0.0],
#                 [sa, ca, 0.0],
#                 [0.0, 0.0, 1.0]])
    r = np.array([[ca, -sa], 
                  [sa, ca]])
    edge1N = npdot(r, edge1)
    edge2N = npdot(r, edge2)
    v = (edge1N+center)
    w = (edge2N+center)
    l2 = npdot(v-w, v-w)
    if l2 == 0.0:
        return np.sqrt(npdot(v-ref, v-ref)), v
    t = npdot(ref-v, w-v) / l2
    if t < 0.0:
        return np.sqrt(npdot(v-ref,v-ref)), v
    elif t > 1.0:
        return np.sqrt(npdot(w-ref,w-ref)), w
    projection = v + t * (w - v)
    return np.sqrt(npdot(ref-projection, ref-projection)), projection
    
    
def relPos(p1, ang,  p2):
    """
        Calculates the position of p2 relativ to the relevance frame of p1
    """
    l = len(p1)
    ca = math.cos(ang)
    sa = math.sin(ang)
    if l == 3:
        trans = np.array([[ca, -sa, 0.0, p1[0]],
                 [sa, ca, 0.0, p1[1]],
                 [0.0, 0.0, 1.0, p1[2]],
                 [0.0,0.0,0.0,1.0]])
    else:
        trans = np.array([[ca, -sa, p1[0]],
                          [sa,ca,p1[1]],
                          [0.0,0.0,1.0]])
    invTrans = invertTransMatrix(trans)
    tmpPos = np.ones(l+1)
    tmpPos[:l] = np.copy(p2)
    newPos = npdot(invTrans, tmpPos)[:l]
    return np.round(newPos, config.NUMDEC)
    
def relPosVel(p1,v1, ang, p2,v2):
    ca = math.cos(ang)
    sa = math.sin(ang)
    l = len(p1)
    if l == 3:
        trans = np.array([[ca, -sa, 0.0, p1[0]],
                 [sa, ca, 0.0, p1[1]],
                 [0.0, 0.0, 1.0, p1[2]],
                 [0.0,0.0,0.0,1.0]])
    elif l == 2:
        trans = np.array([[ca, -sa, p1[0]],
                          [sa,ca,p1[1]],
                          [0.0,0.0,1.0]])
    invTrans = invertTransMatrix(trans)
    tmpPos = np.ones(l+1)
    tmpPos[:l] = np.copy(p2)
    newPos = npdot(invTrans, tmpPos)[:l]
    tmpVel = np.zeros(l+1)
    tmpVel[:l] = v2-v1
    newRelVel = npdot(invTrans, tmpVel)[:l]
    tmpVel[:l] = v2
    newVel = npdot(invTrans, tmpVel)[:l]
    return np.round(newPos, config.NUMDEC), np.round(newRelVel, config.NUMDEC), np.round(newVel, config.NUMDEC)
    
def relPosVelChange(ang, pdif, vdif):
    ca = math.cos(ang)
    sa = math.sin(ang)
    l = len(pdif)
    if l == 3:
        trans = np.array([[ca, sa, 0.0],
                      [-sa, ca, 0.0],
                      [0.0,0.0,1.0]])
    elif l == 2:
        trans = np.array([[ca,sa],
                          [-sa,ca]])
    else:
        raise NotImplementedError("Only works for 2d or 3d positions differences. l: ", l)
    newPDif = npdot(trans,pdif)
    newVDif = npdot(trans,vdif)
    return np.round(newPDif, config.NUMDEC), np.round(newVDif, config.NUMDEC)
    
def globalPosVelChange(ang, pdif, vdif):
    ca = math.cos(ang)
    sa = math.sin(ang)
    l = len(pdif)
    if l == 3:
        trans = np.array([[ca, -sa, 0.0],
                      [sa, ca, 0.0],
                      [0.0,0.0,1.0]])
    elif l == 2:
        trans = np.array([[ca,-sa],
                          [sa,ca]])
    else:
        raise NotImplementedError("Only works for 2d or 3d positions differences. l: ", l)
                          
    newPDif = npdot(trans,pdif)
    newVDif = npdot(trans,vdif)
#    return newPDif, newVDif
    return np.round(newPDif, config.NUMDEC), np.round(newVDif, config.NUMDEC)
    
def globalPosVel(p1, ang, relPos, relVel):
    ca = math.cos(ang)
    sa = math.sin(ang)
    l = len(p1)
    if l == 3:
        trans = np.array([[ca, -sa, 0.0, p1[0]],
                 [sa, ca, 0.0, p1[1]],
                 [0.0, 0.0, 1.0, p1[2]],
                 [0.0,0.0,0.0,1.0]])
    elif l == 2:
        trans = np.array([[ca, -sa, p1[0]],
                          [sa,ca,p1[1]],
                          [0.0,0.0,1.0]])
    else:
        raise NotImplementedError("Only works for 2d or 3d positions. l: ", l)
    tmpPos = np.ones(l+1)
    tmpPos[:l] = np.copy(relPos)
    newPos = npdot(trans, tmpPos)[:l]
    tmpVel = np.zeros(l+1)
    tmpVel[:l] = relVel
    newVel = npdot(trans, tmpVel)[:l]
    return newPos, newVel
    


def computeDistanceClosing(id1, p1, v1, ang1, id2, p2, v2, ang2):
    edges  = {15: [(-0.25,0.05),(-0.25,-0.05),(0.25,-0.05),(0.25,0.05)], 8: [(0.0,0.0)]}
    segments = {15: [(edges[15][0], edges[15][1]),(edges[15][1],edges[15][2]),(edges[15][2],edges[15][3]),(edges[15][3],edges[15][0])], 8:[]}    
    if id1 == 8:
        ref = p1[:2]
#        ref[2] = p2[2]
        vel = (v1-v2)[:2]
        ang = ang2
        center = p2[:2]
        segId = id2
    elif id1 == 15:
        ref = p2[:2]
#        ref[2] = p1[2]
        vel = (v2-v1)[:2]
        ang = ang1
        center = p1[:2]
        segId = id1
    else:
        raise NotImplementedError("Currently only distance between gripper and object is implemented. (id1: {})".format(int(id1)))
    dpList = [dist(center, e1, e2, ang, ref) for e1,e2 in segments[segId]]
    sortedL = sorted(dpList, key = itemgetter(0))
#    print "sorted list: ", sortedL
    normal = (ref-sortedL[0][1])
    norm = np.linalg.norm(normal)
    if norm > 0.0:
        normal /= np.linalg.norm(normal)
    return np.round(max(sortedL[0][0]-0.025,0.0), config.NUMDEC), np.round(npdot(normal, vel), config.NUMDEC)
    
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