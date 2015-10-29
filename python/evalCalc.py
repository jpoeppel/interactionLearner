#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:15:35 2015

@author: jpoeppel
"""
#
import numpy as np
#
#
#data = np.loadtxt("../data/gateModel10Runs_Gate_Act_NoDynsITMNewWinner.txt", delimiter=";")
#
#print np.mean(data,axis = 0)

#data = np.loadtxt("./testData.txt", delimiter=";")
#print data
"""
New records
"""


import os

import re

def natSort(s, _nsr=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsr, s)]

class Experiment(object):
    
    def __init__(self):
        self.name = ""
        self.testPosValues = {}
        self.trainRunOrder = []
        self.trainStartPosX = {}
        self.testStartPosX = {}

directory = "../evalData/"
#directory = "./"

fileList = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]

ysPos = []
ysOri = []
errorsPos = []
errorsOri = []


#testPosValues = {}
experiments = {}

currentExperiment = None

for f in sorted(fileList, key=natSort):
    
    if f.startswith(".") or "_config" in f or "ITM" in f or "old" in f:
        continue
#    if "Configuration_40" in f or "Configuration_8" in f:
#        continue
#    if not "gateModel" in f:
#        continue
    if not "Symmetric" in f:
        continue
    nameOnly = f.replace('.txt','')
    parts = nameOnly.split("_")
    modelName = parts[0]
    numTrainRuns = int(parts[parts.index("TrainRuns")-1])
    currentConfig = int(parts[parts.index("Configuration")+1])
    if "_E" in f:
        experimentSuffix = parts[-1]
    else:
        experimentSuffix = ""
    experimentName = modelName+"_C_"+str(currentConfig)+experimentSuffix
    if currentExperiment == None or experimentName != currentExperiment.name:
        if experimentName in experiments:
            currentExperiment = experiments[experimentName]
        else:
            currentExperiment = Experiment()
            currentExperiment.name = experimentName
            experiments[experimentName] = currentExperiment
        
#    trainRunOrder.append(numTrainRuns)
    currentExperiment.trainRunOrder.append(numTrainRuns)
    #load data
    print "filename: ", directory+f
    data = np.loadtxt(directory+f, delimiter = ';')
    
    trainrows = data[:,1] == 1
    
    trainData = data[trainrows,:]
    firstTrainFrames = trainData[:,3] == 0
#    startingTrainPositionsX = trainData[firstTrainFrames,12]
    currentExperiment.trainStartPosX[numTrainRuns] = np.copy(trainData[firstTrainFrames,12])
    
    testrows = np.invert(trainrows)
    testData = data[testrows,:]
    testDifs = np.copy(testData[:,:14])
    testDifs[:,5:14] -= testData[:,15:]
    
    firstFrames = testData[:,3] == 0
    
    lastFrames = np.roll(firstFrames, -1) #TODO Test if this is always correct!    

#    startingPositionsX = testData[firstFrames,12]    
    currentExperiment.testStartPosX[numTrainRuns] = testData[firstFrames,12] 
    
    #Consider filtering, i.e. only last frame
    actDifs = testDifs[lastFrames,12:14]
    oriDifs = testDifs[lastFrames,11]
    posDifs = testDifs[lastFrames,5:7]
    keyPoint1Difs = testDifs[lastFrames,7:9]
    keyPoint2Difs = testDifs[lastFrames,9:11]
    
    numTestRuns = int(np.max(testData[:,2]))+1
    testPosRes = {}
    startingTestPos = np.zeros(numTestRuns)
    #Consider each testpos separetly
    for i in xrange(numTestRuns):
        iMask = testData[lastFrames,2] == i
        posDifsTmp = posDifs[iMask]
        oriDifsTmp = oriDifs[iMask]
        testPosRes[i] = (np.mean(np.linalg.norm(posDifsTmp, axis=1)), np.mean(oriDifsTmp), 
                        np.std(np.linalg.norm(posDifsTmp, axis=1)), np.std(oriDifsTmp))
        startPosMask = testData[:,2] == i
        startingTestPos[i] = np.mean(testData[firstFrames*startPosMask,12])
#        print startingTestPos
#    testPosValues[numTrainRuns] = testPosRes
    currentExperiment.testPosValues[numTrainRuns] = testPosRes    
    
    
    
    ysPos.append(np.mean(np.linalg.norm(posDifs, axis=1)))
    ysOri.append(np.mean(oriDifs))
    errorsPos.append(np.std(np.linalg.norm(posDifs, axis=1)))
    errorsOri.append(np.std(oriDifs))
    
#    print "Starting pos: ", startingTrainPositionsX
    
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#pp = PdfPages("../gateEvaConfig12.pdf")
#
#
#fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(15,6))
#ax0.axhline(y=0, color ='lightgrey')
#ax0.errorbar(xs,ysOri, yerr=errorsOri, fmt='o')
#ax0.set_title('Mean orientational differences at the end of a run and their standard deviation.')
#ax1.axhline(y=0, color ='lightgrey')
#ax1.errorbar(xs,ysPos, yerr=errorsPos, fmt='o')
#ax1.set_title('Mean positional differences at the end of a run and their standard deviation.')
#ax1.set_xlim([0,35])

#pp.savefig()
#pp.close()


for e in experiments.values():


    pp = PdfPages("../pdfs/"+ e.name +".pdf")
    
    numSubPlotRows = len(e.testPosValues)
    fig, axes = plt.subplots(numSubPlotRows, 2)
    i = 0
    for row in axes:
        resDict = e.testPosValues[e.trainRunOrder[i]]
        xs = startingTestPos
        ysPos = [resDict[j][0] for j in xrange(len(resDict))]
        errorsPos = [resDict[j][2] for j in xrange(len(resDict))]  
        ysOri = [resDict[j][1] for j in xrange(len(resDict))]
        errorsOri = [resDict[j][3] for j in xrange(len(resDict))]#
        row[0].axhline(y=0, color ='lightgrey')
        row[1].axhline(y=0, color ='lightgrey')
        row[0].errorbar(xs, ysPos, yerr = errorsPos, fmt='o')
        row[1].errorbar(xs, ysOri, yerr = errorsOri, fmt='o')
        row[0].set_title("Positional difference with {} trainruns.".format(e.trainRunOrder[i]))
        row[1].set_title("Orientation difference with {} trainruns.".format(e.trainRunOrder[i]))
    #    row[0].set_xlim([-0.4,0.4])
    #    row[1].set_xlim([-0.4,0.4])
        i+=1
    
    #plt.tight_layout()
    fig.subplots_adjust(hspace = 1)
    fig.suptitle("Results for experiment " + e.name)
    pp.savefig()
    pp.close()
    
plt.show()