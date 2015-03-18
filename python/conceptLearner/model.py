# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:24:53 2015

@author: jpoeppel
"""

import numpy as np
import gazeboInterface as gi


class State(object):
    
    def __init__(self, worldState):
        self.gripperState = worldState.gripper.toDict()
        self.objectStates = [o.toDict() for o in worldState.objects]
        self.importanteGripperAttribs = self.gripperState.keys()
        if self.objectStates:
            self.importantObjectAttribs = self.objectStates[0].keys() 
        else:
            self.importantObjectAttribs = []
            
    def score(self, otherState):
        otherGripperState = otherState.gripper.toDict()
        otherObjectsState = [o.toDict() for o in otherState.objects]
        s = 0.0
        for k in self.importanteGripperAttribs:
            #Todo compare                    
            pass
        for k in self.importantObjectAttribs:
            #Todo compare objects
        return s

class Case(object):
    
    def __init__(self, name):
        self.id = name
        self.initialState = State()
        self.action = gi.Action()
        self.resultState = State()
        
        
    def score(self, state, action):
        return self.initialState.score(state) + self.action.score(action)


class ModelCBR(object):
    """
        A simple case-base reasoning model. Creates cases and stores them for
        future generalisation.
    """
    
    
    def __init__(self):
        self.caseStore = []
        pass
    
    def searchCases(self, state,action):
        bestCase = None
        bestScore = 0.0
        for c in self.caseStore:
            tmpScore = c.score(state, action)
            if tmpScore > bestScore:
                bestCase = c
                bestScore = tmpScore
                
        return bestCase
    
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
        similarCase = self.searchCases(state,action)
        if similarCase == None:
            #If no similar case was found, say the case does not change
            return state
        else:
            if action.cmd == "MOVE":
                state.linVel = action.direction
                
            return similarCase.resultState
    
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
    
class ModelHMM(object):
    def __init__(self):
        
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
    