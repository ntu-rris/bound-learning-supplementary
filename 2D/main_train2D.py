import tensorflow as tf
import numpy as np
import os

#import matplotlib
import matplotlib.pyplot as plt

import shrinkNetV2 as net
import util

points=np.load("points_banana.dat",allow_pickle=True)    #(m,2)
#points=np.load("points_mickey.dat",allow_pickle=True) 

m=points.shape[0]

#hyperparameters
maxEpoch=20001
updatesPerEpoch=1
batchSize=m		#one batch is the whole training set

genGrid=True
################################
core="training/"
trainFolder=core+"test01/"

sessionFile=trainFolder+"model.ckpt"
imageFolder=trainFolder+'/img/'

if not os.path.exists(trainFolder+'checkpoint'):
	startNewTraining=True
else:
	startNewTraining=False

os.makedirs(trainFolder,exist_ok=True)
os.makedirs(imageFolder,exist_ok=True)

#grid
gridSpace=0.1
lineResolution = 200
horizontalLines=[]

for vx in np.arange(-2.0, 2.01, gridSpace):
	for vy in np.linspace(-2.0,2.0, num=lineResolution):
		horizontalLines.append([vx,vy])

hLines=np.array(horizontalLines)	#(nLine*lineResolution,2)
vLines=np.stack([hLines[:,1],hLines[:,0]],axis=1)	#just swap x and y	#(nLine*lineResolution,2)

allLines=np.vstack([hLines,vLines])	#(2*nLine*lineResolution,2)

with net.sess.graph.as_default():
	saver = tf.train.Saver()

	if(startNewTraining):
		net.sess.run(tf.global_variables_initializer())
	else:
		saver.restore(net.sess, sessionFile)
	
	start_global_step=tf.train.global_step(net.sess, net.global_step_tensor)

	feed_dict = {net.x_in:points}

	fig=plt.figure(figsize=(8, 8))	#reuse fig will save a lot of memory

	def saveGrid(iteration):
		#points
		latentPoints,reconPoints=net.sess.run([net.zFake,net.x_recon],feed_dict={net.x_in:points})
		
		#boundary
		angle=np.linspace(0,2*np.pi, num=100, endpoint=True)
		latentBoundary=np.stack([np.cos(angle),np.sin(angle)], axis=-1) #(100,2)
		actualBoundary,reconBoundary=net.sess.run([net.x_generated,net.bound_test],feed_dict={net.z_in:latentBoundary})
		
		#grid
		encodedGrid,=net.sess.run([net.zFake],feed_dict={net.x_in:allLines})
		encodedGrid=encodedGrid.reshape([-1,lineResolution,2])	#in latent sapce	

		decodedGrid,=net.sess.run([net.x_generated],feed_dict={net.z_in:allLines})
		decodedGrid=decodedGrid.reshape([-1,lineResolution,2])	#in original space

		util.plotLatentSpaceWithDeformedGrid(encodedGrid,latentPoints,latentBoundary,reconBoundary,trainFolder+'img/gridLatentSpace'+str(iteration)+'.png',fig=fig)
		util.plotOriginalSpaceWithDeformedGrid(decodedGrid,points,reconPoints,actualBoundary,trainFolder+'img/gridOriginalSpace'+str(iteration)+'.png',fig=fig,bound=False)

	if(genGrid):
		saveGrid(start_global_step)

	for epoch in range(start_global_step,start_global_step+maxEpoch+1):	#one epoch go through the whole dataset

		net.sess.run([net.copyEncoderActions])	#prepare the frozen encoder
		_,_ = net.sess.run([net.trainer,net.increment_global_step_op],feed_dict=feed_dict)

		if(epoch%50==0):
			print(net.sess.run([net.global_step_tensor,net.point_rec_loss,net.bound_rec_loss,net.out_loss,net.shrink_loss,net.deformedArea],feed_dict=feed_dict))

		if(epoch in [0,1,2,4,8,16,32,64,128,256,512,1024,1500,2000] or epoch%2500==0):
					
			#save boundary snapshot
			angle=np.linspace(0,2*np.pi, num=100, endpoint=True)
			latentBoundary=np.stack([np.cos(angle),np.sin(angle)], axis=-1) #(100,2)
			actualBoundary,reconBoundary=net.sess.run([net.x_generated,net.bound_test],feed_dict={net.z_in:latentBoundary})

			#save latent snapshot
			latentPoints,reconPoints=net.sess.run([net.zFake,net.x_recon],feed_dict={net.x_in:points})
			
			util.saveScatterPlotWithBoundary(points,reconPoints,actualBoundary,trainFolder+'img/recon'+str(epoch)+'.png',fig=fig,bound=False)
			util.saveScatterPlotWithBoundaryLatent(latentPoints,latentBoundary,reconBoundary,trainFolder+'img/laten'+str(epoch)+'.png',fig=fig)

			if(genGrid and epoch%2500==0):
				saveGrid(epoch)

		if epoch%500==0:
			save_path = saver.save(net.sess, sessionFile)	#save sess	
	
	save_path = saver.save(net.sess, sessionFile)	#save sess

	if(genGrid):
		saveGrid('Last')