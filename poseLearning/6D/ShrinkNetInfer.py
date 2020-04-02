import os
import numpy as np
import tensorflow as tf
import shrinkNetND as net
#import util3D
#import myMath
import random
class ShrinkNetInfer:

	def __init__(self,trainFolder,dataFile):
		self.D=net.D

		self.points=np.load(dataFile,allow_pickle=True).astype(dtype='float32')    #(m,6)

		sessionFile=trainFolder+"model.ckpt"
		if not os.path.exists(trainFolder+'checkpoint'):
			print('Invalid training folder')
			return
		
		with net.sess.graph.as_default():
			saver = tf.train.Saver()
			saver.restore(net.sess, sessionFile)

	def generateEncodedGridRadius(self,startList,stopList,numList):   #only encoder is needed

		with net.sess.graph.as_default():

			return net.sess.run([net.encodedGridRadius],feed_dict={
				net.startPlace:startList,
				net.stopPlace:stopList,
				net.numPlace:numList   
			})[0]   #will be reshaped outside

	def filterPoints(self,selectedDimensionList,center3D,radius):    #selectedDimensionList=[1,2,5] #fixed dimension
		diff=self.points[:,selectedDimensionList]-center3D.reshape([1,3])
		distance=np.sqrt(np.square(diff).sum(axis=1))
		selected=(distance<radius)
		return self.points[selected,:]  #to be displayed
	
	def randomSample(self):
		index=random.randint(0,self.points.shape[0]-1)
		return self.points[index,:]
	