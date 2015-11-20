#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:31:07 2015
File containing common helper functions.
@author: jpoeppel
"""

import numpy as np
# Explicit import to save some performance
from numpy import dot as npdot
import math

from operator import itemgetter
from configuration import config

GAZEBOCMDS = { "NOTHING": 0,"MOVE": 1, "GRAB": 2, "RELEASE": 3}
SIDE = {"NONE": 0, "DOWN": 1, "UP": 2}

GRIPPERSTATES = {"OPEN":0, "CLOSED": 1}


        
def quaternionToEuler(quat):
    """
        Function to compute the euler angles around the x,y,z axes of a 
        rotation given by a quaternion
        
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
        Function to compute the rotation quaternion from angles around 
        the x,y,z axes
        
        Paramters
        ---------
        euler: np.array(3)/np.array(1)/float
            The rotation angle(s) around the axis. If only one value is given, 
            it is assumed to be around the z axis.
            
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
    """
        Function to compute the transformation matrix for an object with 
        given orientation and position.
        
        Paramters
        ---------
        euler: np.array(3)/np.array(1)/float
            The orientation of the object whos transformation 
            matrix is to be computed.
        pos: np.array(3)/np.array(2)
            3d or 2d position of the object whos transformation 
            matrix is to be computed.
            
        Returns
        -------
        np.array(4x4)/np.array(3x3)
            4d or 3d transformation matrix to transform homogenous coordinates.
    """
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
    """
        Function to compute the 3d transformation matrix for an object with 
        given orientation and position in the 2d plane only.
        
        Paramters
        ---------
        euler: float
            The orientation of the object whos transformation 
            matrix is to be computed.
        pos: np.array(2)
            2d position of the object whos transformation 
            matrix is to be computed.
            
        Returns
        -------
        np.array(3x3)
            3d transformation matrix to transform homogenous coordinates.
    """
    a = euler
    px,py = pos
    ca = math.cos(a)
    sa = math.sin(a)
    return np.array([[ca, -sa, px],
                      [sa, ca, py],
                      [0.0,0.0,1.0]])

def invertTransMatrix(matrix):
    """
        Inverts a given transformation matrix.
        
        Parameters
        ----------
        matrix: np.array(4x4)/np.array(3x3)
            4d or 3d transformation matrix
            
        Returns
        -------
        np.array(4x4)/np.array(3x3)
            The inverted transformation matrix
    """
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
                                          
    
def relPos(p1, ang,  p2):
    """
        Calculates the position of one object relativ to the coordinate frame of 
        another object with position p1.
        
        Parameters
        ---------
        p1: np.array(2)/np.array(3)
            Global 2d or 3d position of the reference object
        ang: float
            Rotation around the z-axis of the reference object
        p2: np.array(2)/np.array(3)
            Global 2d or 3d position of the second object whose relative position is 
            to be calculated
        
        Returns
        -------
        np.array(2)/np.array(3)
            Transformed position of the object
    """
    l = len(p1)
    ca = np.cos(ang)
    sa = np.sin(ang)
    if l == 3:
        trans = np.array([[ca, -sa, 0.0, p1[0]],
                 [sa, ca, 0.0, p1[1]],
                 [0.0, 0.0, 1.0, p1[2]],
                 [0.0,0.0,0.0,1.0]])
    elif l ==2:
        trans = np.array([[ca, -sa, p1[0]],
                          [sa,ca,p1[1]],
                          [0.0,0.0,1.0]])
    else:
        raise AttributeError("The position needs to be either 2 or 3 dimensional")
    invTrans = invertTransMatrix(trans)
    tmpPos = np.ones(l+1)
    tmpPos[:l] = np.copy(p2)
    newPos = npdot(invTrans, tmpPos)[:l]
    return np.round(newPos, config.NUMDEC)
    
def relPosVel(p1,v1, ang, p2,v2):
    """
        Calculates the position and velocity of the second object relativ to 
        the coordinate frame of another object with position p1. Both the 
        local velocity as well as the local velocity difference is computed.
        
        Parameters
        ---------
        p1: np.array(2)/np.array(3)
            Global 2d or 3d position of the reference object
        v1: np.array(2)/np.array(3)
            Global 2d or 3d velocity of the reference object
        ang: float
            Rotation around the z-axis of the reference object
        p2: np.array(2)/np.array(3)
            Global 2d or 3d position of the second object whose relative position is 
            to be calculated
        v2: np.array(2)/np.array(3)
            Global 2d or 3d velocity of the second object
        
        Returns
        -------
        relPos: np.array(2)/np.array(3)
            Transformed position of the object
        relDifVel: np.array(2)/np.array(3)
            The difference in the objects velocity relative to the coordinate
            frame of the first object.
        relVel: np.array(2)/np.array(3)
            Velocity of the second object relative to the coordinate frame of
            the first object.
    """
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
        raise AttributeError("The position needs to be either 2 or 3 dimensional")
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
    """
        Calculates the relative change in position and velocity to a coordinate
        frame rotated by the angle ang around the z-axis.
        
        Parameters
        ---------
        ang: float
            Rotation around the z-axis of the desired coordinate frame
        pdif: np.array(2)/np.array(3)
            Global positional difference vector that is to be transformed
        vdif: np.array(2)/np.array(3)
            Global velocity difference vector that is to be transformed
        
        Returns
        -------
        relPDif: np.array(2)/np.array(3)
            Local positional difference
        relVDif: np.array(2)/np.array(3)
            Local velocity difference
    """
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
        raise AttributeError("Only works for 2d or 3d positions differences. l: ", l)
    newPDif = npdot(trans,pdif)
    newVDif = npdot(trans,vdif)
    return np.round(newPDif, config.NUMDEC), np.round(newVDif, config.NUMDEC)
    
#Might be analogous to the function above with negative angle, but makes the
#Using code more explicit
def globalPosVelChange(ang, pdif, vdif):
    """
        Transforms local changes in position and velocity to global ones.
        
        Parameters
        ---------
        ang: float
            Rotation around the z-axis of the local coordinate frame
        pdif: np.array(2)/np.array(3)
            Local positional difference vector that is to be transformed
        vdif: np.array(2)/np.array(3)
            Local velocity difference vector that is to be transformed
        
        Returns
        -------
        relPDif: np.array(2)/np.array(3)
            Global positional difference
        relVDif: np.array(2)/np.array(3)
            Global velocity difference
    """
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
        raise AttributeError("Only works for 2d or 3d positions differences. l: ", l)
                          
    newPDif = npdot(trans,pdif)
    newVDif = npdot(trans,vdif)
    return np.round(newPDif, config.NUMDEC), np.round(newVDif, config.NUMDEC)
    
def globalPosVel(p1, ang, relPos, relVel):
    """
        Transformes position and velocity relative to the coordinate frame of
        an object given py position p1 and rotation ang back to the global
        coordinate frame.
        
        Parameters
        ---------
        p1: np.array(2)/np.array(3)
            Global 2d or 3d position of the reference object
        ang: float
            Rotation around the z-axis of the reference object
        relPos: np.array(2)/np.array(3)
            Local 2d or 3d position 
        relVel: np.array(2)/np.array(3)
            Local 2d or 3d velocity
        
        Returns
        -------
        globPos: np.array(2)/np.array(3)
            Global position 
        globVel: np.array(2)/np.array(3)
            Global velocity
    """
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
        raise AttributeError("Only works for 2d or 3d positions. l: ", l)
    tmpPos = np.ones(l+1)
    tmpPos[:l] = np.copy(relPos)
    globPos = npdot(trans, tmpPos)[:l]
    tmpVel = np.zeros(l+1)
    tmpVel[:l] = relVel
    globVel = npdot(trans, tmpVel)[:l]
    return globPos, globVel
    

def distPointSeg(v, w, ref, radius=0.0):
    """
        Function to compute the distance between a given point ref and a segment
        defined by its endpoints v and w
        
        Parameters
        ----------
        v: np.array(2)
            First endpoint of the segment
        w: np.array(2)
            Second endpoint of the segment
        ref: np.array(2)
            Reference point to which the distance is to be computs
        radius: float, optional
            Radius of the reference point. Is subtracted from the resulting distance
            to compute segment to sphere distances.
            
        Returns
        -------
        dist: float
            Computed distance
        projection: np.array(2)
            Closest point on the segment to the reference point
        ref: np.array(2)
            The reference point        
    """
    l2 = npdot(v-w, v-w)
    if l2 == 0.0:
        return np.sqrt(npdot(v-ref, v-ref))-radius, v, ref
    t = npdot(ref-v, w-v) / l2
    if t < 0.0:
        return np.sqrt(npdot(v-ref,v-ref))-radius, v, ref
    elif t > 1.0:
        return np.sqrt(npdot(w-ref,w-ref))-radius, w, ref
    projection = v + t * (w - v)
    return np.sqrt(npdot(ref-projection, ref-projection))-radius, projection, ref
    
def generalDistClosing(id1, p1, v1, ang1, id2, p2, v2, ang2):
    """
        Computes the distance between two known objects as well as the closing
        feature. Distance is determined by the closest corner to edge
        distance between the two objects.
        
        Parameters
        ---------
        id1 : int
            Identifier for the first object
        p1 : np.array(2)
            Global 2d position of the first object
        v1 : np.array(2)
            Global velocity of the first object
        ang1 : float
            Rotation around the z-axis of the first object
        id2 : int
            Identifier for the second object
        p2 : np.array(2)
            Global 2d position of the second object
        v2 : np.array(2)
            Global velocity of the second object
        ang2 : float
            Rotation around the z-axis of the second object
            
        Returns
        -------
        dist: float
            Closest distance between the two objects
        closing: float
            Closing feature of both objects
    """
    localEdges = {27: [(-0.25,0.05),(-0.25,-0.05),(0.25,-0.05),(0.25,0.05)], 
                       15: [(-0.25,0.05),(-0.25,-0.05),(0.25,-0.05),(0.25,0.05)], 
                        8: [(0.0,0.0)]}
    segmentIds = {27: [[0,1],[1,2],[2,3],[3,0]],
                  15: [[0,1],[1,2],[2,3],[3,0]],
                  8: []}
    radii = {27: 0.0, 15: 0.0, 8: 0.025}
    vel = (v2-v1)[:2]
    
    ca = math.cos(ang1)
    sa = math.sin(ang1)
    r = np.array([[ca, -sa], 
                  [sa, ca]])
    edges1 = [npdot(r, e)+p1 for e in localEdges[id1]]
    ca = math.cos(ang2)
    sa = math.sin(ang2)
    r = np.array([[ca, -sa], 
                  [sa, ca]])
    edges2 = [npdot(r, e)+p2 for e in localEdges[id2]]
    
    dpList = [distPointSeg(edges2[s[0]], edges2[s[1]], ref, radius=radii[id1]) for ref in edges1 for s in segmentIds[id2]]
    dpList2 = [distPointSeg(edges1[s[0]], edges1[s[1]], ref, radius=radii[id2]) for ref in edges2 for s in segmentIds[id1]]
    
    sortedL = sorted(dpList, key = itemgetter(0))
    sortedL2 = sorted(dpList2, key = itemgetter(0))
            
    if len(sortedL) > 0 and len(sortedL2) > 0:
        if sortedL[0][0] < sortedL2[0][0]:
            minDist = sortedL[0][0]
            normal = sortedL[0][1]-sortedL[0][2]
        else:
            minDist = sortedL2[0][0]
            normal = sortedL2[0][2]-sortedL2[0][1]
    else:
        if len(sortedL) > 0:
            minDist = sortedL[0][0]
            normal = sortedL[0][1]-sortedL[0][2]
        elif len(sortedL2) > 0:
            minDist = sortedL2[0][0]
            normal = sortedL2[0][2]-sortedL2[0][1]
        else:
            minDist = 0
            normal = np.zeros(2)
    
    norm = np.linalg.norm(normal)
    if norm > 0.0:
        normal /= np.linalg.norm(normal)
    return np.round(minDist, config.NUMDEC), np.round(npdot(normal, vel), config.NUMDEC)
    

def generalDist(id1, p1, ang1, id2, p2, ang2):
    """
        Computes the distance between two known objects. 
        Distance is determined by the closest corner to edge
        distance between the two objects.
        
        Parameters
        ---------
        id1 : int
            Identifier for the first object
        p1 : np.array(2)
            Global 2d position of the first object
        ang1 : float
            Rotation around the z-axis of the first object
        id2 : int
            Identifier for the second object
        p2 : np.array(2)
            Global 2d position of the second object
        ang2 : float
            Rotation around the z-axis of the second object
            
        Returns
        -------
        dist: float
            Closest distance between the two objects    
    """
    localEdges = {27: [(-0.25,0.05),(-0.25,-0.05),(0.25,-0.05),(0.25,0.05)], 
                       15: [(-0.25,0.05),(-0.25,-0.05),(0.25,-0.05),(0.25,0.05)], 
                        8: [(0.0,0.0)]}
    segmentIds = {27: [[0,1],[1,2],[2,3],[3,0]],
                  15: [[0,1],[1,2],[2,3],[3,0]],
                  8: []}
    radii = {27: 0.0, 15: 0.0, 8: 0.025}
    
    ca = math.cos(ang1)
    sa = math.sin(ang1)
    r = np.array([[ca, -sa], 
                  [sa, ca]])
    edges1 = [npdot(r, e)+p1 for e in localEdges[id1]]
    ca = math.cos(ang2)
    sa = math.sin(ang2)
    r = np.array([[ca, -sa], 
                  [sa, ca]])
    edges2 = [npdot(r, e)+p2 for e in localEdges[id2]]
    
    dpList = [distPointSeg(edges2[s[0]], edges2[s[1]], ref, radius=radii[id1]) for ref in edges1 for s in segmentIds[id2]]
    dpList2 = [distPointSeg(edges1[s[0]], edges1[s[1]], ref, radius=radii[id2]) for ref in edges2 for s in segmentIds[id1]]
    
    sortedL = sorted(dpList, key = itemgetter(0))
    sortedL2 = sorted(dpList2, key = itemgetter(0))
            
    if len(sortedL) > 0 and len(sortedL2) > 0:
        if sortedL[0][0] < sortedL2[0][0]:
            minDist = sortedL[0][0]
            normal = sortedL[0][1]-sortedL[0][2]
        else:
            minDist = sortedL2[0][0]
            normal = sortedL2[0][2]-sortedL2[0][1]
    else:
        if len(sortedL) > 0:
            minDist = sortedL[0][0]
            normal = sortedL[0][1]-sortedL[0][2]
        elif len(sortedL2) > 0:
            minDist = sortedL2[0][0]
            normal = sortedL2[0][2]-sortedL2[0][1]
        else:
            minDist = 0
            normal = np.zeros(2)

    return np.round(max(minDist,0.0), config.NUMDEC)


"""
Legacy for the non 2d model. Deprecated!
"""

def computeDistanceClosing(id1, p1, v1, ang1, id2, p2, v2, ang2):
    """
        Computes the distance as well as the closing feature between two objects
        where one object has to be the spherical actuator.
        
        Parameters
        ---------
        id1 : int
            Identifier for the first object
        p1 : np.array(2)
            Global 2d position of the first object
        v1 : np.array(2)
            Global velocity of the first object
        ang1 : float
            Rotation around the z-axis of the first object
        id2 : int
            Identifier for the second object
        p2 : np.array(2)
            Global 2d position of the second object
        v2 : np.array(2)
            Global velocity of the second object
        ang2 : float
            Rotation around the z-axis of the second object
            
        Returns
        -------
        dist: float
            Closest distance between the two objects
        closing: float
            Closing feature of both objects  
    """
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
    

def dist(center, corner1, corner2, ang, ref, radius=0.0):
    """
        Function to compute the distance from a reference point to the 
        edge of a given object in 2d.
        
        Parameters
        ----------
        center: np.array(2)
            Center of the object to whose edge the distance is to be computed
        corner1: np.array(2)
            First endpoint of the edge, relative to the object's coordinate frame
        corner2: np.array(2)
            Second endpoint of the edge, relative to the object's coordinate frame
        ang: float
            Rotation around the z-axis of the object
        ref: np.array(2)
            Reference point to which the distance is to be computs
        radius: float, optional
            Radius of the reference point. Is subtracted from the resulting distance
            to compute edge to sphere distances.
            
        Returns
        -------
        dist: float
            Computed distance
        projection: np.array(2)
            Closest point on the edge to the reference point
        ref: np.array(2)
            The reference point        
    """
    ca = math.cos(ang)
    sa = math.sin(ang)
    r = np.array([[ca, -sa], 
                  [sa, ca]])
    corner1N = npdot(r, corner1)
    corner2N = npdot(r, corner2)
    v = (corner1N+center)
    w = (corner2N+center)
    l2 = npdot(v-w, v-w)
    if l2 == 0.0:
        return np.sqrt(npdot(v-ref, v-ref)), v, ref
    t = npdot(ref-v, w-v) / l2
    if t < 0.0:
        return np.sqrt(npdot(v-ref,v-ref)), v, ref
    elif t > 1.0:
        return np.sqrt(npdot(w-ref,w-ref)), w, ref
    projection = v + t * (w - v)
    return np.sqrt(npdot(ref-projection, ref-projection))-radius, projection, ref    
    
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