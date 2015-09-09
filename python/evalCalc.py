#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:15:35 2015

@author: jpoeppel
"""

import numpy as np


data = np.loadtxt("../data/gateModel10Runs_Gate_Act_NoDyns.txt", delimiter=";")

print np.mean(data,axis = 0)