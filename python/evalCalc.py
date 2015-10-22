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

directory = "../evalData/"
#directory = "./"

fileList = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]

xs = []
ysPos = []
ysOri = []
errorsPos = []
errorsOri = []

trainRunOrder = []
testPosValues = {}

for f in sorted(fileList, key=natSort):
    
    if f.startswith(".") or "_config" in f or "ITM" in f:
        continue
    if "Configuration_44" in f:
        continue
#    if "gateModel" in f:
#        continue
    nameOnly = f.replace('.txt','')
    parts = nameOnly.split("_")
    numTrainRuns = int(parts[parts.index("TrainRuns")-1])
    trainRunOrder.append(numTrainRuns)
    currentConfig = int(parts[parts.index("Configuration")+1])
    xs.append(numTrainRuns)
    #load data
    print "filename: ", directory+f
    data = np.loadtxt(directory+f, delimiter = ';')
    
    trainrows = data[:,1] == 1
    
    testrows = np.invert(trainrows)
    testData = data[testrows,:]
    testDifs = np.copy(testData[:,:14])
    testDifs[:,5:14] -= testData[:,15:]
    
    firstFrames = testData[:,3] == 0
    
    lastFrames = np.roll(firstFrames, -1) #TODO Test if this is always correct!    

    startingPositionsX = testData[firstFrames,12]    
    
    #Consider filtering, i.e. only last frame
    actDifs = testDifs[lastFrames,12:14]
    oriDifs = testDifs[lastFrames,11]
    posDifs = testDifs[lastFrames,5:7]
    keyPoint1Difs = testDifs[lastFrames,7:9]
    keyPoint2Difs = testDifs[lastFrames,9:11]
    
    numTestRuns = int(np.max(testData[:,2]))+1
    print numTestRuns
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
    testPosValues[numTrainRuns] = testPosRes
        
    
    
    
    ysPos.append(np.mean(np.linalg.norm(posDifs, axis=1)))
    ysOri.append(np.mean(oriDifs))
    errorsPos.append(np.std(np.linalg.norm(posDifs, axis=1)))
    errorsOri.append(np.std(oriDifs))
    
    
    
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages("../gateEvaConfig12.pdf")


fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(15,6))
ax0.axhline(y=0, color ='lightgrey')
ax0.errorbar(xs,ysOri, yerr=errorsOri, fmt='o')
ax0.set_title('Mean orientational differences at the end of a run and their standard deviation.')
ax1.axhline(y=0, color ='lightgrey')
ax1.errorbar(xs,ysPos, yerr=errorsPos, fmt='o')
ax1.set_title('Mean positional differences at the end of a run and their standard deviation.')
ax1.set_xlim([0,35])

#pp.savefig()
#pp.close()

numSubPlotRows = len(testPosValues)
fig, axes = plt.subplots(numSubPlotRows, 2)

i = 0
for row in axes:
    resDict = testPosValues[trainRunOrder[i]]
    xs = startingTestPos
    print "xs: ", xs
    ysPos = [resDict[j][0] for j in xrange(len(resDict))]
    print "trainRuns: ", trainRunOrder[i]
    print "yPos: ", ysPos
    errorsPos = [resDict[j][2] for j in xrange(len(resDict))]  
    ysOri = [resDict[j][1] for j in xrange(len(resDict))]
    errorsOri = [resDict[j][3] for j in xrange(len(resDict))]
    row[0].errorbar(xs, ysPos, yerr = errorsPos, fmt='o')
    row[1].errorbar(xs, ysOri, yerr = errorsOri, fmt='o')
#    row[0].set_xlim([-0.4,0.4])
#    row[1].set_xlim([-0.4,0.4])
    i+=1


plt.show()