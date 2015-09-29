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
        d, p = dist(center,e1,e2,ang,ref)
        assert np.linalg.norm(d-0.45) < 0.001
        assert np.linalg.norm(p-np.array([0.0,0.45])) < 0.001
        e2 = (-0.25,0.05)
        d, p = dist(center,e1,e2,ang,np.array([-0.3,0.5]))
        assert np.linalg.norm(d-0.05) < 0.001
        assert np.linalg.norm(p-np.array([-0.25,0.5])) < 0.001
        e2 = (0.25,-0.05)
        d, p = dist(center,e1,e2,-0.7,ref)
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
        
        v2 = np.array()