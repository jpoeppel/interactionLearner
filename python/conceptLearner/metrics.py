#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:38:02 2015

@author: jpoeppel
"""

import math
import numpy as np

def cosS(a, b):
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
    
def expS(a,b):
    return math.exp(-(np.linalg.norm(a-b)))
    
def equalS(a,b):
    if a == b:
        return 1
    else:
        return 0

def participantS(a,b):
    return a.score(b)
    
def zeroS(a,b):
    return 0

"""
Metrics to be used when comparing features
#TODO Fix when actual states are finalized
"""
metrics = {"dir": expS, "linVel": expS, "orientation": cosS, "pos": expS, 
           "angVel": expS, "name": zeroS, "id": equalS, "cmd":equalS,
           "A": participantS, "B": participantS, "dist":expS, "contact":equalS,
           "type": equalS, "self": equalS, "other": equalS}
           
def cosD(a,b):
    return 1-cosS(a,b)
    
def expD(a,b):
    return 1-expS(a,b)

def equalD(a,b):
    return 1-equalS(a,b)
    

metrics2 = {"sid":equalD, "stype": equalD, "spos":expD, 
                     "sori": cosD, "slinVel": expD, 
                     "sangVel":expD, "dist": expD, "dir": expD,
                     "contact": equalD, "oid": equalD, "otype": equalD, 
                     "dori": cosD, "dlinVel": expD, "dangVel":expD}