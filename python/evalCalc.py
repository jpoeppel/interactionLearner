#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:15:35 2015

@author: jpoeppel
"""

import numpy as np


data = np.loadtxt("../data/model4_State4_100HZ10TrainRunsDT2.txt", delimiter=";")

print np.mean(data,axis = 0)