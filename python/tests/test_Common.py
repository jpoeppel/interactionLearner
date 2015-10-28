#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:35:34 2015

@author: jpoeppel
"""

import pytest

from conceptLearner.common import *

import numpy as np

class TestCommons:
    
    def test_dist(self):
        center = np.array([0.0,0.5])
        ang = 0.0
        e1 = (-0.25,-0.05)
        e2 = (0.25,-0.05)
        ref = np.array([0.0,0.0])
        d, p, r = dist(center,e1,e2,ang,ref)
        assert np.linalg.norm(d-0.45) < 0.001
        assert np.linalg.norm(p-np.array([0.0,0.45])) < 0.001
        e2 = (-0.25,0.05)
        d, p, r = dist(center,e1,e2,ang,np.array([-0.3,0.5]))
        assert np.linalg.norm(d-0.05) < 0.001
        assert np.linalg.norm(p-np.array([-0.25,0.5])) < 0.001
        e2 = (0.25,-0.05)
        d, p, r = dist(center,e1,e2,-0.7,ref)
        assert np.linalg.norm(d-0.34) < 0.001
        assert np.linalg.norm(p-np.array([0.159,0.3])) < 0.001
        
    def test_distClosing(self):
        id1 = 15
        id2 = 8
        p1 = np.array([0.0,0.5])
        v1 = np.zeros(2)
        ang1 = 0.0
        p2 = np.array([0.0,0.0])
        v2 = np.array([0.0,0.5])
        ang2 = 0.0
        d, c = computeDistanceClosing(id1,p1,v1,ang1,id2, p2,v2,ang2)
        assert np.linalg.norm(d-0.425) < 0.001
        assert np.linalg.norm(c-(-0.5)) < 0.001
        
        v2 = np.array([0.2,0.1])
        d, c = computeDistanceClosing(id1,p1,v1,ang1,id2, p2,v2,ang2)
        assert np.linalg.norm(d-0.425) < 0.001
        assert np.linalg.norm(c-(-0.1)) < 0.001        
        
    def test_generalDistClosing(self):
        id1 = 21
        id2 = 8
        p1 = np.array([0.0,0.5])
        v1 = np.zeros(2)
        ang1 = 0.0
        p2 = np.array([0.0,0.0])
        v2 = np.array([0.0,0.5])
        ang2 = 0.0
        d, c = generalDistClosing(id1,p1,v1,ang1,id2, p2,v2,ang2)
        assert np.linalg.norm(d-0.425) < 0.001
        assert np.linalg.norm(c-(-0.5)) < 0.001
        
        id1 = 15
        d, c = generalDistClosing(id1,p1,v1,ang1,id2, p2,v2,ang2)
        assert np.linalg.norm(d-0.425) < 0.001
        assert np.linalg.norm(c-(-0.5)) < 0.001
        
        id2 = 21
        p2 = np.array([0.0,0.0])
        d, c = generalDistClosing(id1,p1,v1,ang1,id2, p2,v2,ang2)
        assert np.linalg.norm(d-0.4) < 0.001
        assert np.linalg.norm(c-(-0.5)) < 0.001
        
        ang1 = np.pi/2.0
        d, c = generalDistClosing(id1,p1,v1,ang1,id2, p2,v2,ang2)
        assert np.linalg.norm(d-0.2) < 0.001
        assert np.linalg.norm(c-(-0.5)) < 0.001
        
        ang1 = 0.0
        ang2 = np.pi/2.0
        p2 = np.array([0.0,0.0])
        d, c = generalDistClosing(id1,p1,v1,ang1,id2, p2,v2,ang2)
        assert np.linalg.norm(d-0.2) < 0.001
        assert np.linalg.norm(c-(-0.5)) < 0.001
        
    def test_quaternionToEuler(self):
        testquat = np.array([0.0,0.0,0.0,0.0])
        euler = quaternionToEuler(testquat)
        assert np.linalg.norm(euler-np.zeros(3)) < 0.0001
        testquat = np.array([0.0,0.0,0.0,1.0])
        euler = quaternionToEuler(testquat)
        assert np.linalg.norm(euler-np.zeros(3)) < 0.0001
        testquat = np.array([0.0,0.0,1.0,0.5])
        euler = quaternionToEuler(testquat)
        assert np.linalg.norm(euler-np.array([0.0,0.0,2.2143])) < 0.0001
        testquat = np.array([0.0,0.0,0.0])
        with pytest.raises(AssertionError):
            euler = quaternionToEuler(testquat)
            
    def test_eulerToQuat(self):
        testquat = np.array([0.0,0.0,0.0,1.0])
        testeuler = np.zeros(3)
        assert np.linalg.norm(eulerToQuat(testeuler)-testquat) < 0.0001
        testquat = np.array([0.0,0.0,1.0,0.5])
        testquat /= np.linalg.norm(testquat)
        testeuler = np.array([0.0,0.0,2.2143])
        assert np.linalg.norm(eulerToQuat(testeuler)-testquat) < 0.0001


    def test_eulerPosToTransformation(self):
        testEuler = np.array([0.0,0.0,0.0])
        testPos = np.array([0.0,0.0,0.0])
        trans = eulerPosToTransformation(testEuler,testPos)
        assert np.linalg.norm(trans-np.identity(4)) < 0.0001
        trans = eulerPosToTransformation(0.0,testPos)
        assert np.linalg.norm(trans-np.identity(4)) < 0.0001
        trans = eulerPosToTransformation(np.array([0.0]),testPos)
        assert np.linalg.norm(trans-np.identity(4)) < 0.0001
        testPos = np.array([0.0,0.5,0.0])
        trans = eulerPosToTransformation(testEuler,testPos)
        testTrans = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.5],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
        assert np.linalg.norm(trans-testTrans) < 0.0001
        trans = eulerPosToTransformation(math.pi/2.0,testPos)
        testTrans = np.array([[0.0,-1.0,0.0,0.0],[1.0,0.0,0.0,0.5],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
        assert np.linalg.norm(trans-testTrans) < 0.0001
        
        testPos = np.array([1.0,0.5])
        trans = eulerPosToTransformation(np.pi/2.0, testPos)
        testTrans = np.array([[0.0,-1.0,1.0],
                              [1.0,0.0,0.5],
                              [0.0,0.0,1.0]])
        assert np.linalg.norm(trans-testTrans) < 0.0001
        
                              
        
    def test_eulerPosToTransformation2d(self):
        testEuler = 0.0
        testPos = np.array([0.0,0.0])
        trans = eulerPosToTransformation2d(testEuler,testPos)
        assert np.linalg.norm(trans-np.identity(3)) < 0.0001
        testEuler = math.pi/2
        testPos = np.array([0.0,0.0])
        trans = eulerPosToTransformation2d(testEuler,testPos)
        testTrans = np.array([[0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]])
        assert np.linalg.norm(trans-testTrans) < 0.0001
        testPos = np.array([0.0,0.5])
        trans = eulerPosToTransformation2d(testEuler,testPos)
        testTrans = np.array([[0.0,-1.0,0.0],[1.0,0.0,0.5],[0.0,0.0,1.0]])
        assert np.linalg.norm(trans-testTrans) < 0.0001
        testPos = np.array([2.0,1.5])
        testEuler = -np.pi/3.0
        trans = eulerPosToTransformation2d(testEuler,testPos)
        testTrans = np.array([[0.5,np.sqrt(3)/2.0,2.0],[-np.sqrt(3)/2.0,0.5,1.5],[0.0,0.0,1.0]])
        assert np.linalg.norm(trans-testTrans) < 0.0001
        
    def test_invertTransMatrix(self):
        trans = eulerPosToTransformation(0.5,[0.0,0.5,0.2])
        inv = invertTransMatrix(trans)
        assert np.linalg.norm(inv-np.linalg.inv(trans)) < 0.0001
        trans = np.array([[1.0,0.0,1.0],
                          [0.0,1.0,-0.5],
                          [0.0,0.0,1.0]])
        inv = invertTransMatrix(trans)
        assert np.linalg.norm(inv- np.linalg.inv(trans)) < 0.0001       
        trans = np.array([[0.5, np.sqrt(3)/2.0, 0.0],
                           [-np.sqrt(3)/2.0, 0.5, 0.5],
                           [0.0,0.0,1.0]])
        inv = invertTransMatrix(trans)
        assert np.linalg.norm(inv- np.linalg.inv(trans)) < 0.0001    
        #2D
        trans = np.array([[0.5,np.sqrt(3)/2.0,2.0],[-np.sqrt(3)/2.0,0.5,1.5],[0.0,0.0,1.0]])                           
        inv = invertTransMatrix(trans)
        assert np.linalg.norm(inv- np.linalg.inv(trans)) < 0.0001    
        
    def test_relPos(self):
        p1 = np.array([0.0,0.5,0.0])
        p2 = np.zeros(3) 
        ang = 0.0
        rPos = relPos(p1,ang,p2)
        assert np.linalg.norm(rPos-np.array([0.0,-0.5,0.0])) < 0.0001
        rPos = relPos(p1,math.pi/2.0,p2)
        assert np.linalg.norm(rPos-np.array([-0.5,0.0,0.0])) < 0.0001
        #2D
        p1 = np.array([3,2,0])
        p2 = np.array([2.5,-1.1,0])
        ang = np.pi/3.0
        rPos = relPos(p1,ang,p2)
        assert np.linalg.norm(rPos-np.array([-2.935,-1.117,0.0])) < 0.0001
        
    def test_relPosVel(self):
        p1 = np.array([0.0,0.5,0.0])
        v1 = np.zeros(3)
        p2 = np.zeros(3) 
        v2 = np.array([0.0,0.5,0.0])
        ang = 0.0
        rPos, rVel, rVel2 = relPosVel(p1,v1,ang,p2,v2)
        assert np.linalg.norm(rPos-np.array([0.0,-0.5,0.0])) < 0.0001
        assert np.linalg.norm(rVel-np.array([0.0,0.5,0.0])) < 0.0001
        assert np.linalg.norm(rVel2-np.array([0.0,0.5,0.0])) < 0.0001
        rPos, rVel, rVel2 = relPosVel(p1,v1,math.pi,p2,v2)
        assert np.linalg.norm(rPos-np.array([0.0,0.5,0.0])) < 0.0001
        assert np.linalg.norm(rVel-np.array([0.0,-0.5,0.0])) < 0.0001
        assert np.linalg.norm(rVel2-np.array([0.0,-0.5,0.0])) < 0.0001
        v1 = np.array([0.0,0.2,0.0])
        rPos, rVel, rVel2 = relPosVel(p1,v1,math.pi,p2,v2)
        assert np.linalg.norm(rPos-np.array([0.0,0.5,0.0])) < 0.0001
        assert np.linalg.norm(rVel-np.array([0.0,-0.3,0.0])) < 0.0001
        assert np.linalg.norm(rVel2-np.array([0.0,-0.5,0.0])) < 0.0001
        #2D
        p1 = np.array([0.0,0.5])
        v1 = np.array([0.0,0.2])
        p2 = np.zeros(2)
        v2 = np.array([0.0,0.5])
        rPos, rVel, rVel2 = relPosVel(p1,v1,math.pi,p2,v2)
        assert np.linalg.norm(rPos-np.array([0.0,0.5])) < 0.0001
        assert np.linalg.norm(rVel-np.array([0.0,-0.3])) < 0.0001
        assert np.linalg.norm(rVel2-np.array([0.0,-0.5])) < 0.0001
        
    def test_globalPosVel(self):
        p1 = np.array([0.0,0.5,0.0])
        ang = 0.0
        rpos = np.array([0.0,-0.5,0.0])
        rv = np.array([0.0,0.5,0.0])
        gpos, gvel = globalPosVel(p1,ang,rpos,rv)
        assert np.linalg.norm(gpos-np.array([0.0,0.0,0.0])) < 0.0001
        assert np.linalg.norm(gvel-np.array([0.0,0.5,0.0])) < 0.0001
        ang = math.pi/2.0
        gpos, gvel = globalPosVel(p1,ang,rpos,rv)
        assert np.linalg.norm(gpos-np.array([0.5,0.5,0.0])) < 0.0001
        assert np.linalg.norm(gvel-np.array([-0.5,0.0,0.0])) < 0.0001
        
        #2D
        
        p1 = np.array([0.0,0.5])
        ang = math.pi/2.0
        rpos = np.array([0.0,-0.5])
        rv = np.array([0.0,0.5])
        gpos, gvel = globalPosVel(p1,ang,rpos,rv)
        assert np.linalg.norm(gpos-np.array([0.5,0.5])) < 0.0001
        assert np.linalg.norm(gvel-np.array([-0.5,0.0])) < 0.0001

        
    def test_relPosVelChange(self):
        pdif = np.array([0.0,0.5,0.0])
        vdif = np.array([0.0,0.5,0.0])
        ang = 0.0
        npdif, nvdif = relPosVelChange(ang, pdif, vdif)
        assert np.linalg.norm(npdif-np.array([0.0,0.5,0.0])) < 0.0001
        assert np.linalg.norm(nvdif-np.array([0.0,0.5,0.0])) < 0.0001
        ang = math.pi/2.0
        npdif, nvdif = relPosVelChange(ang, pdif, vdif)
        assert np.linalg.norm(npdif-np.array([0.5,0.0,0.0])) < 0.0001
        assert np.linalg.norm(nvdif-np.array([0.5,0.0,0.0])) < 0.0001
        
        #2D
        
        pdif = np.array([0.0,0.5])
        vdif = np.array([0.0,0.5])
        npdif, nvdif = relPosVelChange(ang, pdif, vdif)
        assert np.linalg.norm(npdif-np.array([0.5,0.0])) < 0.0001
        assert np.linalg.norm(nvdif-np.array([0.5,0.0])) < 0.0001
        
    def test_globalPosVelChange(self):
        pdif = np.array([0.0,0.5,0.0])
        vdif = np.array([0.0,0.5,0.0])
        ang = 0.0
        npdif, nvdif = globalPosVelChange(ang, pdif, vdif)
        assert np.linalg.norm(npdif-np.array([0.0,0.5,0.0])) < 0.0001
        assert np.linalg.norm(nvdif-np.array([0.0,0.5,0.0])) < 0.0001
        ang = math.pi/2.0
        npdif, nvdif = globalPosVelChange(ang, pdif, vdif)
        assert np.linalg.norm(npdif-np.array([-0.5,0.0,0.0])) < 0.0001
        assert np.linalg.norm(nvdif-np.array([-0.5,0.0,0.0])) < 0.0001
        
        #2D 
        
        pdif = np.array([0.0,0.5])
        vdif = np.array([0.0,0.5])
        npdif, nvdif = globalPosVelChange(ang, pdif, vdif)
        assert np.linalg.norm(npdif-np.array([-0.5,0.0])) < 0.0001
        assert np.linalg.norm(nvdif-np.array([-0.5,0.0])) < 0.0001