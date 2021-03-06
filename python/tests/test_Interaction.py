#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:24:16 2015
Simple tests to test some core functions in the interactionState model.
Still missing tests for ACs..
@author: jpoeppel
"""

from interactionLearner.modelInteractions_config import Object
from interactionLearner.modelInteractions_config import InteractionState
from interactionLearner.modelInteractions_config import Episode
import interactionLearner.common as common

import numpy as np

class TestObject:
    
    def test_fromInteractionState(self):
        intState = InteractionState()
        intState.id = "15,8"
        intState.vec = np.array([15,8,0.0,0.0,0.0,0.0,-0.5])
        intState.trans = np.array([[1.0,0.0,0.0],
                                   [0.0,1.0,0.5],
                                   [0.0,0.0,1.0]])
        intState.invTrans = common.invertTransMatrix(intState.trans)
        intState.ori= 0.0
        o1, o2 = Object.fromInteractionState(intState)
        assert o1.id == 15
        assert np.linalg.norm(o1.vec-np.array([0.0,0.5,0.0])) < 0.0001
        assert o2.id == 8
        assert np.linalg.norm(o2.vec-np.array([0.0,0.0,0.0])) < 0.0001
        
        intState.vec = np.array([15,8,0.0,0.1,0.0,-0.5,0.1])
        intState.trans = np.array([[0.0,-1.0,0.0],
                                   [1.0,0.0,0.5],
                                   [0.0,0.0,1.0]])
        intState.invTrans = common.invertTransMatrix(intState.trans)
        intState.ori= np.pi/2.0
        o1, o2 = Object.fromInteractionState(intState)
        assert o1.id == 15
        assert np.linalg.norm(o1.vec-np.array([-0.1,0.5,np.pi/2.0])) < 0.0001
        assert o2.id == 8
        assert np.linalg.norm(o2.vec-np.array([-0.1,0.0,0.0])) < 0.0001
                                   

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
        assert np.linalg.norm(intState.vec[:7]-np.array([15,8,0.0,0.0,0.0,0.0,-0.5])) < 0.0001
        assert np.linalg.norm(intState.lastVec[:7]-np.array([15,8,0.0,0.0,0.0,0.0,-0.5])) < 0.0001
        
        assert np.linalg.norm(intState.trans-np.array([[1.0,0.0,0.0],
                                                       [0.0,1.0,0.5],
                                                       [0.0,0.0,1.0]])) < 0.0001
        assert np.linalg.norm(intState.invTrans-np.array([[1.0,0.0,0.0],
                                                       [0.0,1.0,-0.5],
                                                       [0.0,0.0,1.0]])) < 0.0001
        assert intState.ori == 0.0       

        o1.vec = np.array([0.0,0.5,np.pi/2.0])                                                
        intState = InteractionState.fromObjectStates(o1,o2)
        assert intState.id == "15,8"
        assert np.linalg.norm(intState.vec[:7]-np.array([15,8,0.0,0.0,0.0,-0.5,0.0])) < 0.0001
        assert np.linalg.norm(intState.lastVec[:7]-np.array([15,8,0.0,0.0,0.0,-0.5,0.0])) < 0.0001
        
        assert np.linalg.norm(intState.trans-np.array([[0.0,-1.0,0.0],
                                                       [1.0,0.0,0.5],
                                                       [0.0,0.0,1.0]])) < 0.0001
        assert np.linalg.norm(intState.invTrans-np.array([[0.0,1.0,-0.5],
                                                       [-1.0,0.0,0.0],
                                                       [0.0,0.0,1.0]])) < 0.0001
        assert intState.ori == np.pi/2.0     
        o1.vec = np.array([3.0,2.0,np.pi/3.0])
        o2.vec = np.array([2.5, -1.1, 0.0])
        intState = InteractionState.fromObjectStates(o1,o2)
        assert intState.id == "15,8"
        
        assert np.linalg.norm(intState.vec[:7]-np.array([15,8,0.0,0.0,0.0,-2.935,-1.117])) < 0.0001
        assert np.linalg.norm(intState.trans-np.array([[0.5,-np.sqrt(3)/2.0,3.0],
                                                       [np.sqrt(3)/2.0,0.5,2],
                                                       [0.0,0.0,1.0]])) < 0.0001
        assert np.linalg.norm(intState.invTrans-np.array([[0.5,np.sqrt(3)/2.0,-3.232],
                                                       [-np.sqrt(3)/2.0,0.5,1.598],
                                                       [0.0,0.0,1.0]])) < 0.0001
        
class TestEpisode:
    
    def setup(self):
        o1 = Object()
        o1.id = 15
        o1.vec = np.array([0.0,0.5,0.0])
        o1.lastVec = np.array([0.0,0.5,0.0])
        o2 = Object()
        o2.id = 8
        o2.vec = np.array([0.0,0.0,0.0])
        o2.lastVec = np.array([0.0,0.0,0.0])
        self.int1 = InteractionState.fromObjectStates(o1,o2)
        o2 = Object()
        o2.id = 8
        o2.vec = np.array([0.0,0.05,0.0])
        o2.lastVec = np.array([0.0,0.0,0.0])
        self.int2 = InteractionState.fromObjectStates(o1,o2)
    
    def test_constructor(self):
        episode = Episode(self.int1,np.array([0.0,0.5]),self.int2)
        assert np.linalg.norm(episode.difs-np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.05,0.0])) < 0.0001
        
    def test_getChangingFeatures(self):
        episode = Episode(self.int1,np.array([0.0,0.5]),self.int2)
        changingFeatures = episode.getChangingFeatures()
        assert changingFeatures == np.array([6])
        
        self.int2.vec = np.array([15,8,0.0,0.0,0.0,0.1,-0.45,0.0])
        episode = Episode(self.int1,np.array([0.0,0.5]),self.int2)
        changingFeatures = episode.getChangingFeatures()
        assert np.all(changingFeatures == np.array([5,6]))
        
class TestAbstractCollection:
    
    def test_update(self):
        pass
    
    def test_predict(self):
        pass
    
    

class TestModelInteraction:
    
    pass