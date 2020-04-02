import numpy as np
import os

#import matplotlib
import matplotlib.pyplot as plt

def saveScatterPlotWithBoundary(points,reconPoints,actualBoundary, filename, fig=None, bound=True):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.scatter(points[:,0],points[:,1],color='k',s=1)
	plt.scatter(reconPoints[:,0],reconPoints[:,1],color='r',s=1)
	
	plt.plot(actualBoundary[:,0],actualBoundary[:,1],color='b',marker='+')
	
	plt.axis('equal')	#allow true square scaling

	if bound:
		plt.axis([-2,2,-2,2])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def saveScatterPlotWithBoundaryAndHeatMap(points,actualBoundary,xHeat,yHeat,zHeat, filename, fig=None, bound=True):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	ax = plt.gca()	#get current axis
	c = ax.pcolormesh(xHeat, yHeat, zHeat, cmap='RdBu', vmin=0, vmax=1)
	fig.colorbar(c, ax=ax)

	plt.contour(xHeat, yHeat, zHeat,levels=[0.45,0.50,0.55], colors=['r','g','b'])

	plt.scatter(points[:,0],points[:,1],color='k',s=1)
	
	plt.plot(actualBoundary[:,0],actualBoundary[:,1],color='m')
	
	plt.axis('equal')	#allow true square scaling

	if bound:
		plt.axis([-2,2,-2,2])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()



def saveScatterPlotWithBoundaryLatent(points,actualBoundary,reconBoundary, filename, fig=None, bound=True):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	#ax = plt.gca()	#get current axis
	#c = ax.pcolormesh(xHeat, yHeat, zHeat, cmap='RdBu', vmin=0, vmax=1)
	#fig.colorbar(c, ax=ax)

	#plt.contour(xHeat, yHeat, zHeat,levels=[0.45,0.50,0.55], colors=['r','g','b'])

	plt.scatter(points[:,0],points[:,1],color='k',s=1)
	
	plt.plot(actualBoundary[:,0],actualBoundary[:,1],color='m')
	plt.plot(reconBoundary[:,0],reconBoundary[:,1],color='c')

	plt.axis('equal')	#allow true square scaling

	if bound:
		plt.axis([-2,2,-2,2])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def plotOriginalSpaceWithDeformedGrid(grid,points,reconPoints,actualBoundary, filename, fig=None, bound=True):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	color='grey'
	for i in range(grid.shape[0]):
		if(i==grid.shape[0]/2):	#change color (switch line direction)
			color='brown'
		plt.plot(grid[i,:,0],grid[i,:,1], c=color, linewidth=0.5)

	plt.scatter(points[:,0],points[:,1],color='k',s=1)
	plt.scatter(reconPoints[:,0],reconPoints[:,1],color='r',s=1)
	
	plt.plot(actualBoundary[:,0],actualBoundary[:,1],color='b',marker='+')
	
	plt.axis('equal')	#allow true square scaling

	if bound:
		plt.axis([-2,2,-2,2])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def plotLatentSpaceWithDeformedGrid(grid,points,actualBoundary,reconBoundary, filename, fig=None, bound=True):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	#ax = plt.gca()	#get current axis
	#c = ax.pcolormesh(xHeat, yHeat, zHeat, cmap='RdBu', vmin=0, vmax=1)
	#fig.colorbar(c, ax=ax)

	#plt.contour(xHeat, yHeat, zHeat,levels=[0.45,0.50,0.55], colors=['r','g','b'])

	color='grey'
	for i in range(grid.shape[0]):
		if(i==grid.shape[0]/2):	#change color (switch line direction)
			color='brown'
		plt.plot(grid[i,:,0],grid[i,:,1], c=color, linewidth=0.5)


	plt.scatter(points[:,0],points[:,1],color='k',s=1)
	
	plt.plot(actualBoundary[:,0],actualBoundary[:,1],color='m')
	plt.plot(reconBoundary[:,0],reconBoundary[:,1],color='c')

	plt.axis('equal')	#allow true square scaling

	if bound:
		plt.axis([-2,2,-2,2])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()