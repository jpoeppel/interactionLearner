#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:38:02 2015

@author: jpoeppel
"""

import math
import numpy as np

def cosD(a, b):
    try:
        if np.linalg.norm(a) == 0:
            if np.linalg.norm(b) == 0:
                return 1
            else:
                return 0
        else:
            return abs(a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    except Exception, e:
        print e
        print "a: " + str(a)
        print "b: " + str(b)
    
def expD(a,b):
    return math.exp(-(np.linalg.norm(a-b)))
    
def equalD(a,b):
    if a == b:
        return 1
    else:
        return 0

def participantD(a,b):
    return a.score(b)
    
def zeroD(a,b):
    return 0

"""
Metrics to be used when comparing features
"""
metrics = {"dir": expD, "linVel": expD, "orientation": cosD, "pos": expD, 
           "angVel": expD, "name": zeroD, "id": equalD, "cmd":equalD,
           "A": participantD, "B": participantD, "dist":expD, "contact":equalD,
           "type": equalD}
