#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:48:10 2015

@author: jpoeppel
"""

import numpy as np

SHORT_TS = True

WINNER = 0
BESTTWO = 1
NEIGHBOUR = 2

USE_DYNS = 1
LOWFREQ = 2
ALLOWMISSPOSITIONS = 4
FIXFIRSTTHREETRAININGRUNS = 8
#GATE specific
HARDCODEDACT = 16
HARDCODEDGATE = 32
USINGPREDICTIONBOOST = 64


class Config(object):
    
    def __init__(self):
        #ITM general
        self.SIGMAE = 0.05 #For besttwo
        self.TESTMODE = BESTTWO
        self.EMAX = 0.001
        self.EMAX_2 = self.EMAX**2
        self.EMAX05_2 = (0.5*self.EMAX)**2
        #ITM Settings gate
        self.predictorEtaIn = 0.1
        self.predictorEtaOut = 0.0
        self.predictorEtaA = 0.0
        self.predictorTestMode = None #Use default = Besttwo
        self.gateClassifierEtaIn = 0.0
        self.gateClassifierEtaOut = 0.0
        self.gateClassifierEtaA = 0.0
        self.gateClassifierTestMode = 0 #Winner
        self.gateMask = [2,3,6,7]
        self.actuatorEtaIn = 0.0
        self.actuatorEtaOut = 0.1
        self.actuatorEtaA = 0.0
        self.actuatorTestMode = None
        #Gate general settings
        self.HARDCODEDACTUATOR = True
        self.HARDCODEDGATE = True
        self.metaNetIndexThr = 0.01
        self.predictionBoost = 1.0
        #ITM Settings interaction
        self.aCEtaIn = 0.0
        self.aCEtaOut = 0.0
        self.aCEtaA = 0.0
        self.aCTestMode = 0
        self.aCSelectorEtaIn = 0.0
        self.aCSelectorEtaOut = 0.0
        self.aCSelectorEtaA = 0.0
        self.aCSelectorTestMode = 0
        self.aCSelectorMask = [5,6]
        if SHORT_TS:
            self.episodeDifThr = 0.001
            self.frequency = 100
            self.metaNodeThr = 0.001
            self.metaNetDifThr = 0.002
        else:
            self.episodeDifThr = 0.01
            self.frequency = 10
            self.metaNodeThr = 0.01
            self.metaNetDifThr = 0.02
        #General
        self.USE_DYNS = False
        self.NUMDEC = 3
        self.fixedTestSeed = True
        self.testSeed = 4321
        self.fixedTrainSeed = False
        self.trainSeed = 1234
        self.startRunRange = 0.5
        self.testPositions = np.arange(-0.35,0.35,0.035) #np.arange(-0.35,0.35,0.349)#
        self.fixedFirstThreeTrains = False
        
        self.numTooSlow = 0
        self.resetErrors = 0


    def toString(self, usedGate):
        s = "###General configs###\n"
        s += "ITM Sigma: {}\n".format( self.SIGMAE)
        s += "ITM default testmode: {} \n".format(self.TESTMODE)
        s += "ITM max error: {}\n".format(self.EMAX)
        s += "ITM max error squared: {}\n".format(self.EMAX_2)
        s += "ITM half max error squared: {}\n".format(self.EMAX05_2)
        s += "Used dynamic features: {}\n".format(self.USE_DYNS)
        s += "Used frequency: {}\n".format(self.frequency)
        s += "Number of decimal places used: {} \n".format(self.NUMDEC)
        s += "Using fixed testSeed: {}\n".format(self.fixedTestSeed)
        s += "Used testing seed: {}\n".format(self.testSeed)
        s += "Using fixed trainingSeed: {}\n".format(self.fixedTrainSeed)
        s += "Used train seed: {}\n".format(self.trainSeed)
        s += "Used test starting postions: {}\n".format(self.testPositions if self.testPositions != None else "Random")
        s += "Fixed first three training runs: {}\n".format(self.fixedFirstThreeTrains)
        s += "Number of times too slow during whole experiment: {}\n".format(self.numTooSlow)
        s += "Number of reset errors: {}\n".format(self.resetErrors)
        if usedGate:
            s+= "###Gate configuration###\n"
            s+= "Gating configurations: \n"
            s+= "Used prediction boose: {}\n".format(self.predictionBoost)
            s+= "Used hardcoded actuator: {}\n".format(self.HARDCODEDACTUATOR)
            s+= "Used hardcoded gate: {}\n".format(self.HARDCODEDGATE)
            s+= "MetaNode threshold: {}\n".format(self.metaNodeThr)
            s+= "Meta Network difference threshold: {}\n".format(self.metaNetDifThr)
            s+= "Meta Network current index threshold: {}\n".format(self.metaNetIndexThr)
            s+= "Predictor etaIn: {}\n".format(self.predictorEtaIn)
            s+= "Predictor etaOut: {}\n".format(self.predictorEtaOut)
            s+= "Predictor etaA: {}\n".format(self.predictorEtaA)
            s+= "Predictor testMode: {}\n".format(self.predictorTestMode)
            if not self.HARDCODEDGATE:
                s+= "Gate etaIn: {}\n".format(self.gateClassifierEtaIn)
                s+= "Gate etaOut: {}\n".format(self.gateClassifierEtaOut)
                s+= "Gate etaA: {}\n".format(self.gateClassifierEtaA)
                s+= "Gate testMode: {}\n".format(self.gateClassifierTestMode)
                s+= "Used Gate mask: {}\n".format(self.gateMask)
            if not self.HARDCODEDACTUATOR:
                s+= "Actuator etaIn: {}\n".format(self.actuatorEtaIn)
                s+= "Actuator etaOut: {}\n".format(self.actuatorEtaOut)
                s+= "Actuator etaA: {}\n".format(self.actuatorEtaA)
                s+= "Actuator testMode: {}\n".format(self.actuatorTestMode)
        else:
            s+= "###Interaction configuration###\n"
            s+= "Interaction configurations: \n"
            s+= "Episode difference threshold: {}\n".format(self.episodeDifThr)
            s+= "Abstact collection etaIn: {}\n".format(self.aCEtaIn)
            s+= "Abstact collection etaOut: {}\n".format(self.aCEtaOut)
            s+= "Abstact collection etaA: {}\n".format(self.aCEtaA)
            s+= "Abstact collection TestMode: {}\n".format(self.aCTestMode)
            s+= "Abstact collection selector etaIn: {}\n".format(self.aCSelectorEtaIn)
            s+= "Abstact collection selector etaOut: {}\n".format(self.aCSelectorEtaOut)
            s+= "Abstact collection selector etaA: {}\n".format(self.aCSelectorEtaA)
            s+= "Abstact collection selector TestMode: {}\n".format(self.aCSelectorTestMode)
            s+= "Abstract collection selector mask: {}\n".format(self.aCSelectorMask)
            
        return s
        
    def switchToConfig(self, configNummer):
        """
            Function to switch to a specific configuration given by the 
            config nummer
        """
        if configNummer & USE_DYNS:
            self.USE_DYNS = True
        else:
            self.USE_DYNS = False
        if configNummer & LOWFREQ:
            self.episodeDifThr = 0.01
            self.frequency = 10
            self.metaNodeThr = 0.01
            self.metaNetDifThr = 0.02
        else:
            self.episodeDifThr = 0.001
            self.frequency = 100
            self.metaNodeThr = 0.001
            self.metaNetDifThr = 0.002
        if configNummer & FIXFIRSTTHREETRAININGRUNS:
            self.fixedFirstThreeTrains = True
        else:
            self.fixedFirstThreeTrains = False
        if configNummer & HARDCODEDACT:
            self.HARDCODEDACTUATOR = True
        else:
            self.HARDCODEDACTUATOR = False
        if configNummer & HARDCODEDGATE:
            self.HARDCODEDGATE = True
        else:
            self.HARDCODEDGATE = False
        if configNummer & ALLOWMISSPOSITIONS:
            self.startRunRange = 0.7
        else:
            self.startRunRange = 0.5
        if configNummer & USINGPREDICTIONBOOST:
            self.predictionBoost = 1.5
        else:
            self.predictionBoost = 1.0
        pass
        
config = Config()