import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class DataGeneratorGUI:

	def __init__(self):
		self.px=[]
		self.py=[]

		self.fig=plt.figure()
		self.axMain=plt.subplot2grid((3,4),(0,0),rowspan=3,colspan=3)
		
		self.axSaveButton=plt.subplot2grid((3,4),(0,3),rowspan=1,colspan=1)
		self.saveButton=Button(self.axSaveButton,'save')
		
		self.axLoadButton=plt.subplot2grid((3,4),(1,3),rowspan=1,colspan=1)
		self.loadButton=Button(self.axLoadButton,'load')

		self.scatter=self.axMain.scatter([0],[0],color='r',marker='+')
		self.axMain.axis('equal')
		self.axMain.grid('on')
		self.axMain.set_xlim([-2,2])
		self.axMain.set_ylim([-2,2])
		

	def show(self):
		self.cid_press =  self.fig.canvas.mpl_connect('button_press_event',self.onClick)
		plt.show()
		self.fig.canvas.mpl_disconnect(self.cid_press)

	def update(self):
		self.scatter.remove()
		self.scatter=self.axMain.scatter(self.px,self.py,color='k',s=3)
		plt.draw()

	def onClick(self,event):

		if(event.inaxes==self.axMain):
			self.px.append(event.xdata)
			self.py.append(event.ydata)
			self.update()
			
		
		if(event.inaxes==self.axSaveButton):
			data=np.stack([self.px,self.py],axis=1)	#(n,2)
			data.dump('points.dat')
			print("points are saved.")

		if(event.inaxes==self.axLoadButton):
			data=np.load('points.dat',allow_pickle=True)
			self.px=data[:,0].tolist()
			self.py=data[:,1].tolist()
			self.update()

gui=DataGeneratorGUI()
gui.show()