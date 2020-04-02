#visualize the boundary of the trained 6D network
#using isosurface

#only 3 dimensions can be checked at a time
#and those checked dimension will be used to generate 3d boundary

import sys
import numpy as np

from ShrinkNetInfer import ShrinkNetInfer

from vispy import scene
from vispy.scene import SceneCanvas

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QGridLayout, QPushButton, QCheckBox, QSlider, QMessageBox

from MyWidget import *

core="trainedNetwork/"
trainFolder=core+"test6D/"	

dataFile='D:/training/poseBoundary/poseCollection.dat'

infer=ShrinkNetInfer(trainFolder,dataFile)
D=infer.D

sliderMax=1000  #number of tick in slider

marginRatio=0.25

minPoint=np.min(infer.points,axis=0)
maxPoint=np.max(infer.points,axis=0)
rangePoint=maxPoint-minPoint
midPoint=(maxPoint+minPoint)/2
minVolume=minPoint-rangePoint*marginRatio   #for mgrid sampling
maxVolume=maxPoint+rangePoint*marginRatio   #for mgrid sampling


appQt = QtWidgets.QApplication(sys.argv)

win = QMainWindow()
win.resize(700, 500)
win.setWindowTitle('High-Dimensional Visualizer')

canvas=SceneCanvas()
canvas.create_native()
canvas.native.setParent(win)
canvas.unfreeze()
view=canvas.central_widget.add_view()   #add view to the canvas
view.camera='turntable'
canvas.freeze()
#=================================

#transparent surface must come last (order is important)
originAxis=scene.visuals.XYZAxis(parent=view.scene)    #axis length=1

scatter=scene.visuals.Markers(parent=view.scene)
#scatter.set_data(np.random.randn(100,3),size=2.0)
scatter.visible=False

scatterAll=scene.visuals.Markers(parent=view.scene)
#scatterAll.set_data(np.random.randn(100,3),size=2.0,face_color='yellow')
scatterAll.visible=False

#same in all dimension
sampleMin=-2
sampleMax=+2
sampleN=100 #per dimension
'''
def distanceFromOrigin(i,j,k):
    x = (i-50)/25
    y = (j-50)/25
    z = (k-50)/25
    return np.sqrt(x*x+y*y+z*z)
'''
line=np.linspace(sampleMin,sampleMax, num=sampleN, endpoint=False)
#vX,vY,vZ=np.meshgrid(line,line,line)
vX=line.reshape([sampleN,1,1])
vY=line.reshape([1,sampleN,1])
vZ=line.reshape([1,1,sampleN])

#volumeData = np.fromfunction(distanceFromOrigin, (100, 100, 100))
volumeData= np.sqrt(vX*vX + vY*vY + vZ*vZ)  #let the broadcasting works here
surface = scene.visuals.Isosurface(volumeData, level=1.0,#data.max()/4.,
                                   color=(0.5, 0.6, 1, 0.6), shading=None,  #'flat' is much faster than 'smooth', None removes lighting
                                   parent=view.scene)

scale=(sampleMax-sampleMin)/sampleN
surface.transform = scene.transforms.STTransform(scale=(scale,scale,scale),translate=(sampleMin, sampleMin, sampleMin)) #isosurface has no scale
surface.visible=False


#=================================
rightPanel=QWidget()

gbox = QtWidgets.QGridLayout()
#n=6
checkBoxList=[]
sliderList=[]
labelList=[]
for i in range(D):
    checkBox=QCheckBox()
    gbox.addWidget(checkBox,i,0)
    

    slider=MySlider(Qt.Horizontal)
    gbox.addWidget(slider,i,1)
    
    label=QLabel()
    gbox.addWidget(label,i,2)
    
    slider.setRangeAndLinkLabel(sliderMax,minVolume[i],maxVolume[i],label)
    #slider.valueChanged.connect(label.setNum)

    checkBoxList.append(checkBox)
    sliderList.append(slider)
    labelList.append(slider)

    if(i<3):
        checkBox.setChecked(True)
        slider.setEnabled(False)

randomButton=QPushButton()
randomButton.setText("Random Pick")
gbox.addWidget(randomButton,D,1)


testButton=QPushButton()
testButton.setText("test")
testButton.setEnabled(False)
gbox.addWidget(testButton,D+1,1)

DD=D+2
gbox.setRowStretch(DD,1)    #blank seperation

l0=QLabel()
l0.setText('Inclusive radius')
gbox.addWidget(l0,DD+1,1)
radiusSlider=MySliderForRadius(Qt.Horizontal)
gbox.addWidget(radiusSlider,DD+2,1)
radiusLabel=QLabel()
gbox.addWidget(radiusLabel,DD+2,2)
radiusSlider.setRangeAndLinkLabel(sliderMax,0.0,0.3,radiusLabel)
radiusSlider.setContinuousValue(0.05)

def changeSampleVisibility():
    #print(showScatterCheckBox.isChecked())
    scatter.visible=showScatterCheckBox.isChecked()

def changeBoundaryVisibility():
    surface.visible=showSurfaceCheckBox.isChecked()

def changeAllPointsVisibility():
    scatterAll.visible=showAllCheckBox.isChecked()

showScatterCheckBox=QCheckBox()
showScatterCheckBox.setChecked(True)
showScatterCheckBox.toggled.connect(changeSampleVisibility)
gbox.addWidget(showScatterCheckBox,DD+3,1)
showScatterCheckBox.setText('Show Samples')

showSurfaceCheckBox=QCheckBox()
showSurfaceCheckBox.setChecked(True)
showSurfaceCheckBox.toggled.connect(changeBoundaryVisibility)
gbox.addWidget(showSurfaceCheckBox,DD+4,1)
showSurfaceCheckBox.setText('Show boundary')

showAllCheckBox=QCheckBox()
showAllCheckBox.setChecked(False)
showAllCheckBox.toggled.connect(changeAllPointsVisibility)
gbox.addWidget(showAllCheckBox,DD+5,1)
showAllCheckBox.setText('Show every point')

applyButton=QPushButton()
applyButton.setText("Config Dimensions")
gbox.addWidget(applyButton,DD+6,1)



gbox.setColumnMinimumWidth(1,120)
gbox.setColumnMinimumWidth(2,35)


rightPanel.setLayout(gbox)

splitter=QtWidgets.QSplitter(QtCore.Qt.Horizontal)
splitter.addWidget(canvas.native)      #add canvas to splitter
splitter.addWidget(rightPanel)

win.setCentralWidget(splitter)      #add splitter to main window
#========================
activeIndex=[0,1,2] #for the 3D volume
inactiveIndex=[3,4,5]
def applyNewSetting():
    newActiveIndex=[]
    newInactiveIndex=[]
    for i in range(D):
        if(checkBoxList[i].isChecked()):
            newActiveIndex.append(i)
        else:
            newInactiveIndex.append(i)

    if(len(newActiveIndex)!=3):
        print("Can only have 3 active dimensions at the same time.")
        QMessageBox.question(win,"Error message", "Can only have 3 active dimensions at the same time.", QMessageBox.Ok)
        return
        
    #condition is ok
    global activeIndex
    global inactiveIndex

    activeIndex=newActiveIndex
    inactiveIndex=newInactiveIndex
    for i in range(D):
        if i in activeIndex:
            sliderList[i].setEnabled(False)
        else:
            sliderList[i].setEnabled(True)

    print(activeIndex)

    scale=[]
    trans=[]
    for i in range(3):
        j=activeIndex[i]
        scale.append((maxVolume[j]-minVolume[j])/sampleN)
        trans.append(minVolume[j])
    surface.transform = scene.transforms.STTransform(scale=scale,translate=trans)

    surface.visible=False
    scatter.visible=False

    scatterAll.set_data(infer.points[:,activeIndex], 
        symbol='o', 
        size=2.0, 
        edge_width=None,        #The width of the symbol outline in pixels.
        edge_width_rel=0.0,    #The width as a fraction of marker size. Exactly one of edge_width and edge_width_rel must be supplied.
        edge_color='white', 
        face_color='white', 
        scaling=False
    )
    scatterAll.visible=showAllCheckBox.isChecked()

applyButton.clicked.connect(applyNewSetting)
applyNewSetting()



def updateScatter():
    #scatter plot
    #print(activeIndex)
    #print(inactiveIndex)
    center=[]
    for i in range(3):
        center.append(sliderList[inactiveIndex[i]].getContinuousValue())
    pointsToDisplay=infer.filterPoints(inactiveIndex,np.array(center),radiusSlider.getContinuousValue())
    print("Number of points:",pointsToDisplay.shape[0])
    if(pointsToDisplay.shape[0]>0):
        scatter.set_data(pointsToDisplay[:,activeIndex], 
            symbol='o', 
            size=5.0, 
            edge_width=None,        #The width of the symbol outline in pixels.
            edge_width_rel=0.1,    #The width as a fraction of marker size. Exactly one of edge_width and edge_width_rel must be supplied.
            edge_color='red', 
            face_color='red', 
            scaling=False
        )
        scatter.visible=showScatterCheckBox.isChecked()
    else:
        scatter.visible=False

def updateSurface():
    #isosurface
    startList=[]
    stopList=[]
    numList=[]
    for i in range(D):
        if i in activeIndex:
            startList.append(minVolume[i])
            stopList.append(maxVolume[i])
            numList.append(sampleN)
        else:
            tmp=sliderList[i].getContinuousValue()
            startList.append(tmp)
            stopList.append(tmp)
            numList.append(1)
    flatData=infer.generateEncodedGridRadius(startList,stopList,numList)    #(?,)
    
    if(np.min(flatData)<1):
        surface.set_data(flatData.reshape([sampleN,sampleN,sampleN]))
        surface.visible=showSurfaceCheckBox.isChecked()
    else:
        surface.visible=False

def updateBoth():
    #print("test")
    updateScatter()
    updateSurface()
    #============

def randomPick():
    s=infer.randomSample()
    for i in range(D):
        sliderList[i].setContinuousValue(s[i])

    updateScatter()
    updateSurface()


radiusSlider.valueChanged.connect(updateScatter)
randomButton.clicked.connect(randomPick)
testButton.clicked.connect(updateBoth)

for i in range(D):
    sliderList[i].updateBoth=updateBoth
    #sliderMoved.connect(updateBoth)

#========================
win.show()
appQt.exec_()