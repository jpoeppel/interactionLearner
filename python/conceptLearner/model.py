# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:24:53 2015

@author: jpoeppel
"""

import numpy as np
import gazeboInterface as gi


class Model():
    def __init__(self):
        self.model 
        pass
    
    def predict(self, action, state):
        """
            Predict the next world state based on the current state of the world
            and the choosen action.
            
            Parameters
            ----------
            action: 
                The chosen action that the robot should execute
            state:
                The current world state.
            
            Returns
            -------
            
                The predicted world state.
        """
        
        pass
    
    def update(self, action, state, result):
        """
            Update the model according to the actual results.
            
            Parameters
            ----------
            action: 
                The command that was used last
            state:
                The state the world was in before the action.
            result:
                The state the world was in after the action.
        """
        
        pass