#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:39:37 2015

@author: jpoeppel
"""

import numpy as np


class Action(object):
"""
Action are what the model can do itself.
It is limited to setting activations [-1;1] to
the different dimensions.
In the current setup it would be 2 dimensions for the x,y velocity.
"""
    
    def __init__(self, numDims = 0):
        self.numDims = numDims
        self.activation = np.zeros(numDims)
        
        
        