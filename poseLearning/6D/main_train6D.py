import tensorflow as tf
import numpy as np
import os

import shrinkNetND as net
import myMath

from threading import Thread

src='../poseData/poseCollection.dat'
points=np.load(src,allow_pickle=True).astype(dtype='float32')    #(m,6)

m=points.shape[0]

#hyperparameters
maxEpoch=1000001
updatesPerEpoch=1
batchSize=m		#one batch is the whole training set


################################
core="training/"
trainFolder=core+"test01/"	#gpu1

sessionFile=trainFolder+"model.ckpt"
#imageFolder=trainFolder+'/img/'

if not os.path.exists(trainFolder+'checkpoint'):
	startNewTraining=True
else:
	startNewTraining=False

os.makedirs(trainFolder,exist_ok=True)
#os.makedirs(imageFolder,exist_ok=True)

with net.sess.graph.as_default():
	saver = tf.train.Saver()

	if(startNewTraining):
		net.sess.run(tf.global_variables_initializer())
	else:
		saver.restore(net.sess, sessionFile)
	
	#load static training data
	net.loadTrainingData(points)

	start_global_step=tf.train.global_step(net.sess, net.global_step_tensor)


	for epoch in range(start_global_step,start_global_step+maxEpoch+1):	#one epoch go through the whole dataset


		net.sess.run([net.trainer,net.increment_global_step_op])	

		#_ = net.sess.run([])

		if(epoch%100==0):
			print(net.sess.run([net.global_step_tensor,net.point_rec_loss,net.bound_rec_loss,net.out_loss,net.shrink_loss,net.collapse_loss]))
		
		if epoch%1000==0:
			save_path = saver.save(net.sess, sessionFile)	#save sess	
	
	save_path = saver.save(net.sess, sessionFile)	#save sess
