#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:24:16 2015

@author: jpoeppel
"""

from conceptLearner.modelInteractions import Object
from conceptLearner.modelInteractions import InteractionState

import numpy as np

class TestObject:
    
    def test_fromInteractionState(self):
        pass

class TestInteraction:
    
    def test_fromObjectStates(self):
        o1 = Object()
        o1.id = 15
        o1.vec = np.array([0.0,0.5,0.0])
        o1.lastVec = np.array([0.0,0.5,0.0])
        o2 = Object()
        o2.id = 8
        o2.vec = np.array([0.0,0.0,0.0])
        o2.lastVec = np.array([0.0,0.0,0.0])
        intState = InteractionState.fromObjectStates(o1,o2)
        assert intState.id == "15,8"
        assert np.linalg.norm(intState.trans-np.array([[]])) < 0.0001
                                                # sId, oId, x,y,  thet,dx,dy,  dthet,
        assert np.linalg.norm(intState.vec-np.array([15,8,0.0,0.0,0.0,0.0,-0.5,0.0])) < 0.0001
        assert np.linalg.norm(intState.lastVec-np.array())
        pass

class TestAbstractCollection:
    
    pass

class TestModelInteraction:
    
    pass