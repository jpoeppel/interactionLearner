#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:31:07 2015

@author: jpoeppel
"""

import numpy as np
import math

from operator import itemgetter

NUMDEC = 3
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
    
def quatPosToTransformation(quat, pos):
    px,py,pz = pos
    x,y,z,w = quat
    return np.matrix([[1-2*y*y-2*z*z, 2*x*y + 2*w*z, 2*x*z - 2*w*y, px],[2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x, py],
                      [2*x*z+2*w*y,2*y*z-2*w*x, 1-2*x*x-2*y*y, pz],[0.0,0.0,0.0,1.0]])
                      
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
#    b,g,a = euler
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
    return np.array([[cb*ca, sg*sb*ca-cg*sa, cg*sb*ca+sg*sa, px],
                               [cb*sa, sg*sb*sa+cg*ca, cg*sb*sa-sg*ca, py],
                               [-sb, sg*cb, cg*cb, pz],
                               [0.0,0.0,0.0,1.0]])
                      
                      
def eulerPosToTransformation2d(euler, pos):
    a = euler
    px,py = pos
    ca = math.cos(a)
    sa = math.sin(a)
#    return np.matrix(np.round([[ca*cg-sa*cb*sg, sa*cg+ca*cb*sg, sb*sg, px], 
#                      [-ca*sg-sa*cb*cg, -sa*sg+ca*cb*cg, sb*cg, py],
#                      [sa*sb, -ca*sb, cb, pz],
#                      [0.0, 0.0, 0.0, 1.0]],NUMDEC))
    return np.matrix([[ca, -sa, px],
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
    invTrans[:r,c] = np.dot(-matrixRot.T,matrix[:r,c])
    invTrans[r,c] = 1.0
    return invTrans                      
                      
#def invertTransMatrix2(matrix):
#    invTrans = np.matrix(np.zeros((4,4)))
#    invTrans[:3,:3] = matrix[:3,:3].T
#    invTrans[:3,3] = -matrix[:3,:3].T*matrix[:3,3]
#    invTrans[3,3] = 1.0
#    return invTrans
                      
                      
                     
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
    edge1N = np.dot(r, edge1)
    edge2N = np.dot(r, edge2)
    v = (edge1N+center)
    w = (edge2N+center)
    
    l2 = np.dot(v-w, v-w)
    if l2 == 0.0:
        return np.sqrt(np.dot(v-ref, v-ref)), v
    t = np.dot(ref-v, w-v) / l2
    if t < 0.0:
        return np.sqrt(np.dot(v-ref,v-ref)), v
    elif t > 1.0:
        return np.sqrt(np.dot(w-ref,w-ref)), w
    projection = v + t * (w - v)
    return np.sqrt(np.dot(ref-projection, ref-projection)), projection
    
def relPos(p1, ang,  p2):
    """
        Calculates the position of p2 relativ to the relevance frame of p1
    """
    ca = math.cos(ang)
    sa = math.sin(ang)
    trans = np.array([[ca, -sa, 0.0, p1[0]],
                 [sa, ca, 0.0, p1[1]],
                 [0.0, 0.0, 1.0, p1[2],
                 [0.0,0.0,0.0,1.0]]])
    invTrans = invertTransMatrix(trans)
    tmpPos = np.ones(4)
    tmpPos[:3] = np.copy(p2)
    newPos = np.dot(invTrans, tmpPos)[:3]
    return np.round(newPos, NUMDEC)
    
def relPosVel(p1,v1, ang, p2,v2):
    ca = math.cos(ang)
    sa = math.sin(ang)
    trans = np.array([[ca, -sa, 0.0, p1[0]],
                 [sa, ca, 0.0, p1[1]],
                 [0.0, 0.0, 1.0, p1[2]],
                 [0.0,0.0,0.0,1.0]])
    invTrans = invertTransMatrix(trans)
    tmpPos = np.ones(4)
    tmpPos[:3] = np.copy(p2)
    newPos = np.dot(invTrans, tmpPos)[:3]
    tmpVel = np.zeros(4)
    tmpVel[:3] = v2-v1
    newRelVel = np.dot(invTrans, tmpVel)[:3]
    tmpVel[:3] = v2
    newVel = np.dot(invTrans, tmpVel)[:3]
    return np.round(newPos, NUMDEC), np.round(newRelVel, NUMDEC), np.round(newVel, NUMDEC)
    
def relPosVelChange(ang, pdif, vdif):
    ca = math.cos(ang)
    sa = math.sin(ang)
    trans = np.array([[ca, sa, 0.0],
                      [-sa, ca, 0.0],
                      [0.0,0.0,1.0]])
    newPDif = np.dot(trans,pdif)
    newVDif = np.dot(trans,vdif)
    return np.round(newPDif, NUMDEC), np.round(newVDif, NUMDEC)
    
def globalPosVelChange(ang, pdif, vdif):
    ca = math.cos(ang)
    sa = math.sin(ang)
    trans = np.array([[ca, -sa, 0.0],
                      [sa, ca, 0.0],
                      [0.0,0.0,1.0]])
    newPDif = np.dot(trans,pdif)
    newVDif = np.dot(trans,vdif)
    return np.round(newPDif, NUMDEC), np.round(newVDif, NUMDEC)
    
def globalPosVel(p1, ang, relPos, relVel):
    ca = math.cos(ang)
    sa = math.sin(ang)
    trans = np.array([[ca, -sa, 0.0, p1[0]],
                 [sa, ca, 0.0, p1[1]],
                 [0.0, 0.0, 1.0, p1[2]],
                 [0.0,0.0,0.0,1.0]])
    tmpPos = np.ones(4)
    tmpPos[:3] = np.copy(relPos)
    newPos = np.dot(trans, tmpPos)[:3]
    tmpVel = np.zeros(4)
    tmpVel[:3] = relVel
    newVel = np.dot(trans, tmpVel)[:3]
    return np.round(newPos, NUMDEC), np.round(newVel, NUMDEC)
    

#    

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
    return np.round(sortedL[0][0]-0.025, NUMDEC), np.round(np.dot(normal, vel), NUMDEC)
    
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