import numpy as np
import tensorflow as tf

encodingChainSize=60 
gpuID=0	

D=6 
latentSize=D

#resnet block
resNetSize=[D,D] 
def resNetBlock(x,preName,trainable=True):	#x=(None,D)

	layers=[x]
	for i in range(len(resNetSize)):
		layers.append(
			tf.layers.dense(	#output=activ(input*kernel+bias)
				inputs=layers[-1],
				units=resNetSize[i],
				activation=tf.nn.relu,
				#activation=tf.nn.softplus,
				kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.1),
				name=preName+'_'+str(i),
				reuse=tf.AUTO_REUSE,
				trainable=trainable
			)
		)

	z=tf.layers.dense(		#output=input*kernel+bias
		inputs=layers[-1],
		units=D,
		activation=None,
		kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.1),
		name=preName+'_z',
		reuse=tf.AUTO_REUSE,
		trainable=trainable
	)

	return x+z	#No relu after sum (identical version of resnet)

def rawFrozenResNetBlock(x,weights,biases):
	layers=[x]
	for i in range(len(resNetSize)):
		layers.append(
			tf.nn.relu(layers[-1] @ weights[i] + biases[i])
		)
	
	z=layers[-1] @ weights[-1] + biases[-1]	#linear activation function on the last one

	return x+z

#encoder
encodingPrefix='enc_'

#decoder
decodingPrefix='dec_'
decodingChainSize=encodingChainSize


def encode(x):
	layers=[x]
	for i in range(encodingChainSize):
		layers.append(
			resNetBlock(layers[-1],encodingPrefix+str(i))
		)
	return layers[-1]


def frozenEncode(x):
	all_vars = tf.trainable_variables()
	
	layers=[x]
	for i in range(encodingChainSize):
		frozenWeights = [tf.stop_gradient(var) for var in all_vars if var.name.startswith(encodingPrefix+str(i)+'_') and 'kernel' in var.name]
		frozenBiases  = [tf.stop_gradient(var) for var in all_vars if var.name.startswith(encodingPrefix+str(i)+'_') and 'bias' in var.name]

		layers.append(
			rawFrozenResNetBlock(layers[-1],frozenWeights,frozenBiases)
		)
	return layers[-1]

def decode(x):
	layers=[x]
	for i in range(decodingChainSize):
		layers.append(
			resNetBlock(layers[-1],decodingPrefix+str(i))
		)
	return layers[-1]

periodGraph=tf.Graph()

with periodGraph.as_default():
	
	global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
	increment_global_step_op = tf.assign(global_step_tensor, global_step_tensor+1)

	with tf.device('/device:GPU:'+str(gpuID)):
		#data upload section
		pointsAssignPlace = tf.placeholder(tf.float32, shape=[None,D])
		pointsDataVar 	  = tf.Variable([], trainable=False, validate_shape=False,dtype=tf.float32)
		assignPoints      = tf.assign(pointsDataVar, pointsAssignPlace, validate_shape=False)
		pointsData=tf.reshape(pointsDataVar,[-1,D])

		#data use for visualization
		sphereAssignPlace = tf.placeholder(tf.float32, shape=[None,D])
		sphereDataVar 	  = tf.Variable([], trainable=False, validate_shape=False,dtype=tf.float32)
		assignSphere      = tf.assign(sphereDataVar, sphereAssignPlace, validate_shape=False)
		sphereData=tf.reshape(sphereDataVar,[-1,D])

		zFake=encode(pointsData)
		x_recon=decode(zFake)
		pointDistanceSquare=tf.reduce_sum(tf.square(x_recon-pointsData),axis=1)
		point_rec_loss=17*tf.reduce_mean(pointDistanceSquare) + 10*tf.reduce_max(pointDistanceSquare)	#reconstructionTrainer try to optimize rec_loss

		boundarySize=100000	

		random_sample=tf.random_normal([boundarySize,D])
		boundary_sample=random_sample/tf.norm(random_sample,axis=1,keepdims=True)

		boundary_inter=decode(boundary_sample)
		boundary_recon=encode(boundary_inter)
		boundDistanceSquare=tf.reduce_sum(tf.square(boundary_recon-boundary_sample),axis=1)
		bound_rec_loss=17*tf.reduce_mean(boundDistanceSquare)

		out_loss=100*tf.reduce_mean(tf.nn.relu(tf.reduce_sum(tf.square(zFake),axis=1)-1))	#map points beyond unit circle, 
																			#replace 1 with 0.95 to leave some margin
		
		frozen_boundary_recon=frozenEncode(boundary_inter)

		#just a quick check
		checksum=tf.reduce_sum(tf.square(frozen_boundary_recon-boundary_recon))

		#shrink_loss=tf.reduce_mean(tf.nn.relu(tf.reduce_sum(tf.square(frozen_boundary_recon),axis=1)-0))	#more aggresive version (radius square)	
		shrink_loss=tf.reduce_mean(tf.nn.relu(tf.sqrt(tf.reduce_sum(tf.square(frozen_boundary_recon),axis=1))-0))	#this will stay around 1

		#collapse_loss (not used)
		pointCentroidLatent=tf.reduce_mean(zFake,axis=0)	#(3,)
		latentCentroidDistance=tf.reduce_sum(tf.square(pointCentroidLatent))

		pointCentroidOriginal=tf.reduce_mean(pointsData,axis=0)	#static	#(3,)
		boundCentroid=tf.reduce_mean(boundary_inter,axis=0)	#(3,)
		originalCentroidDistance=tf.reduce_sum(tf.square(boundCentroid-pointCentroidOriginal))

		collapse_loss=latentCentroidDistance+originalCentroidDistance

		############################
		
		#define 3 trainers
		all_vars = tf.trainable_variables()
		enc_vars = [var for var in all_vars if var.name.startswith(encodingPrefix)]
		dec_vars = [var for var in all_vars if var.name.startswith(decodingPrefix)]

		optimizer1 = tf.train.AdamOptimizer()
		trainer = optimizer1.minimize(point_rec_loss+bound_rec_loss+out_loss+shrink_loss, var_list=enc_vars+dec_vars)

		#define a way to generate data without messing with the graph
		bound_generated=decode(sphereData)
		sphere_recon=encode(bound_generated)
		
		startPlace = tf.placeholder(tf.float32, shape=[D])
		stopPlace = tf.placeholder(tf.float32, shape=[D])
		numPlace = tf.placeholder(tf.int32, shape=[D])
		lineList=[]
		for i in range(D):
			aLine=tf.linspace(startPlace[i],stopPlace[i],numPlace[i])
			lineList.append(aLine)

		dataGridList=tf.meshgrid(*lineList,indexing='ij')	#return list of 6D tensor
		dataGrid=tf.stack(dataGridList,axis=-1)	#(?,?,?,?,?,?,D)	#three of ? is greater than one 
		flatGrid=tf.reshape(dataGrid,[-1,D])
		encodedGrid=encode(flatGrid)	#(?,D)
		encodedGridRadius=tf.sqrt(tf.reduce_sum(tf.square(encodedGrid),axis=1))	#(?,)	#will be reshaped to 3d grid outside
		
	
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(graph=periodGraph,config=config)

def loadTrainingData(data):	#(None,D)
	print('training points:',data.shape)
	#print(data.dtype)
	sess.run([assignPoints],feed_dict={pointsAssignPlace:data})

def loadSphereData(data):	#(None,D)
	print('sphere points (for visualization):',data.shape)
	#print(data.dtype)
	sess.run([assignSphere],feed_dict={sphereAssignPlace:data})

def printDecoderWeight():
	wb=sess.run(dec_vars)
	for v in wb:
		print(v)
		print('==========')
