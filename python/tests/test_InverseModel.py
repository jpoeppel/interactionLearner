#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:27:38 2015

@author: jpoeppel
"""

from interactionLearner.inverseModel import MetaNode
from interactionLearner.inverseModel import MetaNetwork

import numpy as np

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
        pre = np.array([15, 8, 0.5, -0.02, -0.25,-0.05,0.1,0.0,0.5,0.0])
        dif = 0.105
        n.train(pre, dif)
        mPre1, mPre2 = n.getPreconditions()
        assert np.linalg.norm(mPre1-pre) < 0.0001
        assert mPre2 is None
        pre2 = np.array([15, 8, 0.5, -0.02, -0.2,-0.05,0.0,0.0,0.5,0.0])
        dif2 = 0.2
        n.train(pre2, dif2)
        mPre1, mPre2 = n.getPreconditions()
        avg = (0.105*pre+0.2*pre2)/0.305
        assert np.linalg.norm(mPre1-avg) < 0.0001
        assert mPre2 is None
        
        pre3 = np.array([15, 8, 0.5, -0.01, 0.25,-0.05,0.0,0.5,0.0,0.0])
        dif3 = 0.3 
        n.train(pre3, dif3)
        mPre1, mPre2 = n.getPreconditions()
        avg1 = (0.105*pre+0.2*pre2)/0.305
        avg1[3] = (0.305*avg1[3]+0.3*pre3[3])/0.605
        assert np.linalg.norm(mPre1-np.array([15,8,0.5,-0.015,0.25,-0.05,0.0,0.5,0.0,0.0])) < 0.0001
        assert np.linalg.norm(mPre2-avg1) < 0.0001
        
        
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