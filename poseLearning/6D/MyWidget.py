from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QSlider,QLabel

class MySlider(QSlider):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updateBoth=None

    def setRangeAndLinkLabel(self, sliderMax, minRange, maxRange, label:QLabel):
        self.setTickInterval(1)
        self.setRange(0,sliderMax)
        self.setValue(sliderMax/2)
        
        self.sliderMax=sliderMax
        self.minRange=minRange
        self.maxRange=maxRange

        self.label=label

        self.valueChanged.connect(self.updateLabel)
        self.updateLabel()

    #def setSuddenUpdate

    def getContinuousValue(self):
        return self.minRange+(self.value()/self.sliderMax)*(self.maxRange-self.minRange)
    
    def updateLabel(self):
        v=self.getContinuousValue()
        self.label.setText('{0:.3f}'.format(v))

        if(self.updateBoth is not None):
            self.updateBoth()

    def setContinuousValue(self,v):   
        self.valueChanged.disconnect(self.updateLabel)     
        
        tmp=int((v-self.minRange)/(self.maxRange-self.minRange)*self.sliderMax)
        tmp=max(0,min(tmp,self.sliderMax))
        self.setValue(tmp)
        
        v=self.getContinuousValue()
        self.label.setText('{0:.3f}'.format(v))
        
        self.valueChanged.connect(self.updateLabel)

class MySliderForRadius(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setRangeAndLinkLabel(self, sliderMax, minRange, maxRange, label:QLabel):
        self.setTickInterval(1)
        self.setRange(0,sliderMax)
        self.setValue(sliderMax/2)
        
        self.sliderMax=sliderMax
        self.minRange=minRange
        self.maxRange=maxRange

        self.label=label

        self.valueChanged.connect(self.updateLabel)
        self.updateLabel()

    def getContinuousValue(self):
        return self.minRange+(self.value()/self.sliderMax)*(self.maxRange-self.minRange)
    
    def updateLabel(self):
        v=self.getContinuousValue()
        self.label.setText('{0:.3f}'.format(v))
    
    def setContinuousValue(self,v):
        
        tmp=int((v-self.minRange)/(self.maxRange-self.minRange)*self.sliderMax)
        tmp=max(0,min(tmp,self.sliderMax))
        self.setValue(tmp)

    