#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:03:55 2015

@author: jpoeppel
"""

from conceptLearner.modelGate_2D_config import Object

import numpy as np

class TestObject:
    
    def setup(self):
        self.o1 = Object() 
        self.o1.vec = np.array([0.0,0.5,0.0]) # With z coordinate
        self.o1.lastVec = np.copy(self.o1.vec)
        self.o1.id = 15
        self.o2 = Object()
        self.o2.vec = np.array([0.0,0.0,0.0])
        self.o2.lastVec = np.copy(self.o2.vec)
        self.o2.id = 8
        
    def test_getKeyPoints(self):
        o = Object()
        o.id = 15
        o.vec = np.array([0.0,0.5,0.0])
        p1,p2,p3 =o.getKeyPoints()
        assert np.linalg.norm(p1-o.vec[:2]) < 0.0001
        assert np.linalg.norm(p2-np.array([0.25,0.55])) < 0.0001
        assert np.linalg.norm(p3-np.array([-0.25,0.45])) < 0.0001
        o.vec[2] = np.pi/2.0
        p1,p2,p3 =o.getKeyPoints()
        assert np.linalg.norm(p1-o.vec[:2]) < 0.0001
        assert np.linalg.norm(p2-np.array([-0.05,0.75])) < 0.0001
        assert np.linalg.norm(p3-np.array([0.05,0.25])) < 0.0001
    
    def test_getRelVec(self):
        relVec = self.o1.getRelVec(self.o2)
        assert np.linalg.norm(relVec-np.array([15,8,0.425,0.0, 0.0,-0.5,0.0,0.0,0.0,0.0])) < 0.0001
        self.o2.lastVec = np.array([0.0,-0.005,0.0])
        relVec = self.o1.getRelVec(self.o2)
        assert np.linalg.norm(relVec-np.array([15,8,0.425,-0.005, 0.0,-0.5,0.0,0.005,0.0,0.005])) < 0.0001
        
    def test_getRelObjectVec(self):
        vec = self.o1.getRelObjectVec(self.o2)
        assert np.linalg.norm(vec-np.array([0.0,-0.5,0.0])) < 0.0001
        self.o1.vec[2] = np.pi/2.0
        vec = self.o1.getRelObjectVec(self.o2)
        assert np.linalg.norm(vec-np.array([-0.5,0.0,-np.pi/2.0])) < 0.0001
        
    def test_getGlobalPosVel(self):
        rpos = np.array([0.0,-0.5])
        rvel= np.array([0.0,0.5])
        gpos, gvel = self.o1.getGlobalPosVel(rpos,rvel)
        assert np.linalg.norm(gpos-np.zeros(2)) < 0.0001
        assert np.linalg.norm(gvel-np.array([0.0,0.5])) < 0.0001
        self.o1.vec[2] = np.pi/2.0
        gpos, gvel = self.o1.getGlobalPosVel(rpos,rvel)
        assert np.linalg.norm(gpos-np.array([0.5,0.5])) < 0.0001
        assert np.linalg.norm(gvel-np.array([-0.5,0.0])) < 0.0001
        
    def test_getLocalChangeVec(self):
        opre = Object()
        opre.id = 15
        opre.vec = np.array([0.0,0.5,0.0])
        oPost = Object()
        oPost.id = 15
        oPost.vec = np.array([-0.001,0.501,0.5])
        changeVec = opre.getLocalChangeVec(oPost)
        assert np.linalg.norm(changeVec-np.array([-0.001, 0.001,0.5])) < 0.0001
        opre.vec = np.array([0.0,0.5,np.pi/2.0])
        oPost.vec = np.array([0.0,0.55,np.pi/2.0])
        changeVec = opre.getLocalChangeVec(oPost)
        assert np.linalg.norm(changeVec-np.array([0.05, 0.0,0.0])) < 0.0001
        
