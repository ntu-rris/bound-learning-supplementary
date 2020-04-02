import tensorflow as tf
import numpy as np
import os

#import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import shrinkNet3D_v4 as net
import util3D
import myMath

from threading import Thread

src='../poseData/poseCollection.dat'

#(m,3)
points=np.load(src,allow_pickle=True).astype(dtype='float32')[:,0:3]	 #shoulder
#points=np.load(src,allow_pickle=True).astype(dtype='float32')[:,3:6]    #elbow

m=points.shape[0]

#hyperparameters
maxEpoch=100001
updatesPerEpoch=1
batchSize=m		#one batch is the whole training set

genPlotly=True
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


sphericalTrace=myMath.genSphericalLine(30,0.02)		#(sn,3)

with net.sess.graph.as_default():
	saver = tf.train.Saver()

	if(startNewTraining):
		net.sess.run(tf.global_variables_initializer())
	else:
		saver.restore(net.sess, sessionFile)
	
	#load static training data
	net.loadTrainingData(points)
	net.loadSphereData(sphericalTrace)

	start_global_step=tf.train.global_step(net.sess, net.global_step_tensor)

	fig=plt.figure(figsize=(8, 8))	#reuse fig will save a lot of memory

	for epoch in range(start_global_step,start_global_step+maxEpoch+1):	#one epoch go through the whole dataset

		net.sess.run([net.trainer,net.increment_global_step_op])	#,feed_dict=feed_dict)

		if(epoch%100==0):
			print(net.sess.run([net.global_step_tensor,net.point_rec_loss,net.bound_rec_loss,net.out_loss,net.shrink_loss,net.collapse_loss]))


		if(epoch in [0,500,1000] or epoch%2000==0):
			
			actualBoundary,reconBoundary=net.sess.run([net.bound_generated,net.sphere_recon])

			
			latentPoints,reconPoints=net.sess.run([net.zFake,net.x_recon])
			
			#matplotlib is very slow, so put in a thread
			def saveImage():
				epochStamp=epoch
				util3D.saveOriginalMat(points,reconPoints,actualBoundary,trainFolder+'img/recon'+str(epochStamp)+'.png',fig=fig)
				util3D.saveLatentMat(latentPoints,sphericalTrace,reconBoundary,trainFolder+'img/laten'+str(epochStamp)+'.png',fig=fig)
			Thread(target=saveImage).start()
			
			if(genPlotly and epoch!=0 and epoch%10000==0):	#extremely slow
				def savePlotly1():
					util3D.saveOriginalPlotly(points,reconPoints,actualBoundary,trainFolder+'img/plotly_recon'+str(epoch)+'.html')
				
				def savePlotly2():
					util3D.saveLatentPlotly(latentPoints,sphericalTrace,reconBoundary,trainFolder+'img/plotly_laten'+str(epoch)+'.html')

				Thread(target=savePlotly1).start()
				Thread(target=savePlotly2).start()			
				
		
		if epoch%1000==0:
			save_path = saver.save(net.sess, sessionFile)	#save sess	
	
	save_path = saver.save(net.sess, sessionFile)	#save sess

	if(genPlotly):
		actualBoundary,reconBoundary=net.sess.run([net.bound_generated,net.sphere_recon])
		latentPoints,reconPoints=net.sess.run([net.zFake,net.x_recon])
		util3D.saveOriginalPlotly(points,reconPoints,actualBoundary,trainFolder+'img/plotly_recon'+'Last'+'.html')
		util3D.saveLatentPlotly(latentPoints,sphericalTrace,reconBoundary,trainFolder+'img/plotly_laten'+'Last'+'.html')