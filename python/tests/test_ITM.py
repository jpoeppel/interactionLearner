#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:22:13 2015
Simple tests for the aitm
@author: jpoeppel
"""

#from conceptLearner.topoMaps import ITM
from interactionLearner.aitm import Node
from interactionLearner.aitm import AITM
import numpy as np

class TestITM:
    
    def setup(self):
        self.itm = AITM()
        self.x1 = np.array([1.0,0.2,0.3,-0.4])
        self.y1 = np.array([0.5])
        self.x2 = np.array([-1.0,0.4,0.1,0.4])
        self.y2 = np.array([-0.2])
        self.x3 = np.array([0.0,0.3,0.2,0.0])
        self.y3 = np.array([0.15])
        pass
    
    def test_update(self):
        #Set sigmae according to the order of magnitude in the input
        assert len(self.itm.nodes) == 0
        self.itm.update(self.x1,self.y1)
        assert len(self.itm.nodes) == 1
        self.itm.update(self.x2,self.y2)
        assert len(self.itm.nodes) == 2
        self.itm.update(self.x1,self.y1)
        assert len(self.itm.nodes) == 2
        #Testing no need to update because interpolation already covers this
        print "updating the third time"
        self.itm.update(self.x3,self.y3)
        print "after update"
        assert len(self.itm.nodes) == 2
    
    def test_test(self):
        self.itm.update(self.x1,self.y1)
        assert np.linalg.norm(self.itm.test(self.x1) - self.y1) < 0.001
        self.itm.update(self.x2,self.y2)
        assert np.linalg.norm(self.itm.test(self.x2) - self.y2) < 0.001
        assert np.linalg.norm(self.itm.test(self.x3) - self.y3) < 0.001
    
class TestNode():
    
    def setup(self):
        self.Node1 = Node(np.array([1.0,0.2,0.3,-0.4]), 0, np.array([0.5]))
        self.Node2 = Node(np.array([-1.0,0.4,0.1,0.4]), 1, np.array([-0.2]))
        pass
    
    def test_addNeighbour(self):
        assert len(self.Node1.neig) == 0
        self.Node1.addNeighbour(self.Node2.id, self.Node2)
        assert len(self.Node1.neig) == 1
        assert self.Node1.neig.values()[0].id == self.Node2.id
        
    def test_removeNeigbour(self):
        self.Node1.addNeighbour(self.Node2.id, self.Node2)
        self.Node1.remNeighbour(self.Node2.id)
        assert len(self.Node1.neig) == 0


