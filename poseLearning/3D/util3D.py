import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly
import plotly.plotly as py
import plotly.graph_objs as go


def setAxEqual(ax,data):
	#ax.set_aspect('equal')

	X,Y,Z=data[:,0],data[:,1],data[:,2]
	
	#for equal scale in every axis
	max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
	mid_x = (X.max()+X.min()) * 0.5
	mid_y = (Y.max()+Y.min()) * 0.5
	mid_z = (Z.max()+Z.min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)


def saveOriginalMat(points,reconPoints,actualBoundary, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(xs=points[:,0],ys=points[:,1],zs=points[:,2],color='grey',s=0.5)
	ax.scatter(xs=reconPoints[:,0],ys=reconPoints[:,1],zs=points[:,2],color='r',s=1)
	
	ax.plot(xs=actualBoundary[:,0],ys=actualBoundary[:,1],zs=actualBoundary[:,2],color='b')	#,marker='+')
	setAxEqual(ax,actualBoundary)
	#plt.axis('equal')

	#if bound:
	#	plt.axis([-2,2,-2,2])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()	#ax is deleted


def saveLatentMat(points,actualBoundary,reconBoundary, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch	
	
	ax = fig.add_subplot(111, projection='3d')
	#ax = plt.gca()	#get current axis
	#c = ax.pcolormesh(xHeat, yHeat, zHeat, cmap='RdBu', vmin=0, vmax=1)
	#fig.colorbar(c, ax=ax)

	#plt.contour(xHeat, yHeat, zHeat,levels=[0.45,0.50,0.55], colors=['r','g','b'])

	ax.scatter(xs=points[:,0],ys=points[:,1],zs=points[:,2],color='grey',s=0.5)
	ax.plot(xs=actualBoundary[:,0],ys=actualBoundary[:,1],zs=actualBoundary[:,2],color='m')
	ax.plot(xs=reconBoundary[:,0],ys=reconBoundary[:,1],zs=reconBoundary[:,2],color='c')

	setAxEqual(ax,actualBoundary)
	#plt.axis('equal')	#allow true square scaling

	#if bound:
	#	plt.axis([-2,2,-2,2])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()	#ax is deleted

def saveOriginalPlotly(points,reconPoints,actualBoundary, filename):
	
	pointTrace=go.Scatter3d(
		x=points[:,0],
		y=points[:,1],
		z=points[:,2],
		#text=energyStringList,
		#text=np.reshape(energyShell,[-1]),
		#hoverinfo='x+y+z',
		mode='markers',
		#mode = 'lines',
		marker=dict(
			color='rgb(100,100,100)',
			size=1.5
		),
		name='data'
	)

	reconTrace=go.Scatter3d(
		x=reconPoints[:,0],
		y=reconPoints[:,1],
		z=reconPoints[:,2],
		#text=energyStringList,
		#text=np.reshape(energyShell,[-1]),
		#hoverinfo='x+y+z',
		mode='markers',
		#mode = 'lines',
		marker=dict(
			color='rgb(255,0,0)',
			size=1.5
		),
		name='recon'
	)

	boundaryTrace=go.Scatter3d(
		x=actualBoundary[:,0],
		y=actualBoundary[:,1],
		z=actualBoundary[:,2],
		#text=energyStringList,
		#text=np.reshape(energyShell,[-1]),
		#hoverinfo='x+y+z',
		#mode='markers',
		mode = 'lines',
		marker=dict(
			color='rgb(0,0,255)',
			size=1.5
		),
		name='boundary'
	)

	fig=dict(data=[pointTrace,reconTrace,boundaryTrace])	
	plotly.offline.plot(fig, filename=filename, auto_open=False)

def saveLatentPlotly(points,actualBoundary,reconBoundary, filename):
	
	pointTrace=go.Scatter3d(
		x=points[:,0],
		y=points[:,1],
		z=points[:,2],
		#text=energyStringList,
		#text=np.reshape(energyShell,[-1]),
		#hoverinfo='x+y+z',
		mode='markers',
		#mode = 'lines',
		marker=dict(
			color='rgb(100,100,100)',
			size=1.5
		),
		name='data'
	)

	boundaryTrace=go.Scatter3d(
		x=actualBoundary[:,0],
		y=actualBoundary[:,1],
		z=actualBoundary[:,2],
		#text=energyStringList,
		#text=np.reshape(energyShell,[-1]),
		#hoverinfo='x+y+z',
		#mode='markers',
		mode = 'lines',
		marker=dict(
			color='rgb(255,0,255)',
			size=1.5
		),
		name='boundary'
	)

	reconBoundaryTrace=go.Scatter3d(
		x=reconBoundary[:,0],
		y=reconBoundary[:,1],
		z=reconBoundary[:,2],
		#text=energyStringList,
		#text=np.reshape(energyShell,[-1]),
		#hoverinfo='x+y+z',
		#mode='markers',
		mode = 'lines',
		marker=dict(
			color='rgb(0,255,255)',
			size=1.5
		),
		name='boundary'
	)

	fig=dict(data=[pointTrace,boundaryTrace,reconBoundaryTrace])	
	plotly.offline.plot(fig, filename=filename, auto_open=False)