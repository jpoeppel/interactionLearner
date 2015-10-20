#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:03:55 2015

@author: jpoeppel
"""

from conceptLearner.modelGate_2D_config import Object
from conceptLearner.modelGate_2D_config import MetaNode
from conceptLearner.modelGate_2D_config import MetaNetwork
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
        
class TestMetaNode:
    
    def test_train(self):
        n = MetaNode()
        pre = np.array([15, 8, 0.5, -0.02, -0.25,-0.05,0.0,0.0,0.5,0.0])
        signComp ="1;1;1;-1;-1;-1;0;0;1;0"
        dif = 0.1
        n.train(pre, dif)
        assert n.signCombinations.has_key(signComp)
        assert n.signCombinations[signComp] == dif
        pre2 = np.array([15, 8, 0.5, -0.02, -0.2,-0.05,0.0,0.0,0.5,0.0])
        signComp ="1;1;1;-1;-1;-1;0;0;1;0"
        dif2 = 0.2
        n.train(pre2, dif2)
        assert n.signCombinations.has_key(signComp)
        assert n.signCombinations[signComp] == dif+dif2
        pre3 = np.array([15, 8, 0.5, -0.001, 0.25,-0.05,0.0,0.5,0.0,0.0])
        signComp = "1;1;1;0;1;-1;0;1;0;0"
        dif3 = 0.3 
        n.train(pre3, dif3)
        assert n.signCombinations.has_key(signComp)
        assert n.signCombinations[signComp] == dif3
        
    def test_getPreconditions(self):
        n = MetaNode()
        pre = np.array([15, 8, 0.5, -0.02, -0.25,-0.05,0.0,0.0,0.5,0.0])
        dif = 0.1
        n.train(pre, dif)
        mPre1, mPre2 = n.getPreconditions()
        assert np.linalg.norm(mPre1-pre) < 0.0001
        assert mPre2 is None
        pre2 = np.array([15, 8, 0.5, -0.02, -0.2,-0.05,0.0,0.0,0.5,0.0])
        dif2 = 0.2
        n.train(pre2, dif2)
        mPre1, mPre2 = n.getPreconditions()
        assert np.linalg.norm(mPre1-(0.1*pre+0.2*pre2)/0.3) < 0.0001
        assert mPre2 is None
        pre3 = np.array([15, 8, 0.5, -0.001, 0.25,-0.05,0.0,0.5,0.0,0.0])
        dif3 = 0.3 
        n.train(pre3, dif3)
        mPre1, mPre2 = n.getPreconditions()
        avg1 = (0.1*pre+0.2*pre2)/0.3
        avg1[3] = 0.5*(avg1[3]+pre3[3])
        avg1[7] = 0.5*(avg1[7]+pre3[7])
        avg1[8] = 0.5*(avg1[8]+pre3[8])
        assert np.linalg.norm(mPre1-avg1) < 0.0001
        assert np.linalg.norm(mPre2-np.array([15,8,0.5,-0.0105,0.25,-0.05,0.0,0.25,0.25,0.0])) < 0.0001
        
class TestMetaNetwork:
    
    def test_train(self):
        n = MetaNetwork()
        pre = np.array([15, 8, 0.5, -0.02, -0.25,-0.05,0.0,0.0,0.5,0.0])
        difs = np.array([0.0,0.1,0.0,0.1])
        n.train(pre,difs)
        assert n.nodes.has_key("1.0")
        assert n.nodes.has_key("3.0")
        pre2 = np.array([15, 8, 0.5, -0.02, -0.2,-0.05,0.0,0.0,0.5,0.0])
        difs2 = np.array([0.0,0.1,-0.1,0.1])
        n.train(pre2,difs2)
        assert n.nodes.has_key("-2.0")
        
    def test_getPreconditions(self):
        n = MetaNetwork()
        pre = np.array([15, 8, 0.5, -0.02, -0.25,-0.05,0.0,0.0,0.5,0.0])
        difs = np.array([0.0,0.1,0.0,0.1])
        n.train(pre,difs)
        pre2 = np.array([15, 8, 0.5, -0.02, -0.2,-0.05,0.0,0.0,0.5,0.0])
        difs2 = np.array([0.0,0.1,-0.1,0.1])
        n.train(pre2,difs2)
        targetDifs = np.array([0.0,0.5,-0.2,0.3])
        preCons = n.getPreconditions(targetDifs)
        assert np.linalg.norm(preCons-0.5*(pre2+pre)) < 0.0001
        n.curIndex = None
        targetDifs = np.array([0.0,0.5,-0.6,0.3])
        preCons = n.getPreconditions(targetDifs)
        assert np.linalg.norm(preCons-pre2) < 0.0001