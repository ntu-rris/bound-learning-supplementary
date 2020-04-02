import numpy as np
import tensorflow as tf

D=2 #core dimension without pose
latentSize=D

#resnet block
resNetSize=[2,2] #two hidden layers
def resNetBlock(x,preName,trainable=True):	#x=(None,D)

	layers=[x]
	for i in range(len(resNetSize)):
		layers.append(
			tf.layers.dense(	#output=activ(input*kernel+bias)
				inputs=layers[-1],
				units=resNetSize[i],
				activation=tf.nn.relu,
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

frozenPrefix='frozen_'

#encoder
encodingPrefix='enc_'
encodingChainSize=15 

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
	layers=[x]
	for i in range(encodingChainSize):
		layers.append(
			resNetBlock(layers[-1],frozenPrefix+encodingPrefix+str(i),trainable=False)
		)
	return layers[-1]

def decode(x):
	layers=[x]
	for i in range(decodingChainSize):
		layers.append(
			resNetBlock(layers[-1],decodingPrefix+str(i))
		)
	return layers[-1]


def crossProductSum(u,v):	#(None,2)
	return tf.reduce_sum(u[:,0]*v[:,1]-v[:,0]*u[:,1])

periodGraph=tf.Graph()

with periodGraph.as_default():
	global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
	increment_global_step_op = tf.assign(global_step_tensor, global_step_tensor+1)

	#define 3 loss (the full graph)
	x_in=tf.placeholder(tf.float32, shape=[None,D], name='x_in')	#weight will multiply from the right side
	zFake=encode(x_in)
	x_recon=decode(zFake)
	point_rec_loss=tf.reduce_mean(tf.square(x_recon-x_in))	#reconstructionTrainer try to optimize rec_loss

	boundarySize=100
	evenAngle=tf.linspace(0.0,2*np.pi,boundarySize+1)+(tf.random_uniform([1])*(2*np.pi/(boundarySize)))
	unitBound=tf.stack([tf.cos(evenAngle),tf.sin(evenAngle)],axis=1)	#(boundarySize,2)
	
	boundary_sample=unitBound[0:boundarySize,:]
	boundary_inter=decode(boundary_sample)
	boundary_recon=encode(boundary_inter)
	bound_rec_loss=tf.reduce_mean(tf.square(boundary_recon-boundary_sample))

	rec_loss=50*(bound_rec_loss+point_rec_loss)

	out_loss=100*tf.reduce_mean(tf.nn.relu(tf.reduce_sum(tf.square(zFake),axis=1)-1))	#map points beyond unit circle, 
																		#replace 1 with 0.95 to leave some margin
	
	#evenly-spaced angles at random, endpoint not removed
	deformedBound=decode(unitBound)
	A=deformedBound[0:boundarySize,:]
	B=deformedBound[1:boundarySize+1,:]
	deformedArea=0.5*crossProductSum(A,B)	#(A,B-A)	#for reference only, not for training
	#deformedArea_loss=0.01*tf.maximum(deformedArea, 0)	#0.01 is too aggressive		#0.002 is ok but slow
	
	with tf.variable_scope("frozen",reuse=tf.AUTO_REUSE):
		frozen_boundary_recon=frozenEncode(boundary_inter)	#this use untrainable version of the encoder	#(boundarySize,2)
	
	#shrink_loss=tf.reduce_mean(tf.nn.relu(tf.reduce_sum(tf.square(frozen_boundary_recon),axis=1)-0))	#more aggresive version (radius square)	
	shrink_loss=tf.reduce_mean(tf.nn.relu(tf.sqrt(tf.reduce_sum(tf.square(frozen_boundary_recon),axis=1))-0))	#this will stay around 1

	
	all_vars = tf.trainable_variables()
	enc_vars = [var for var in all_vars if var.name.startswith(encodingPrefix)]
	dec_vars = [var for var in all_vars if var.name.startswith(decodingPrefix)]

	copyEncoderActions=[]	#must run this list before running the training
	untrainbleVars=tf.global_variables(scope='frozen')
	for dstVar,srcVar in zip(untrainbleVars,enc_vars):
		copyEncoderActions.append(tf.assign(dstVar,srcVar))


	optimizer1 = tf.train.AdamOptimizer()
	trainer = optimizer1.minimize(rec_loss+out_loss+shrink_loss, var_list=enc_vars+dec_vars)

	#define a way to generate data without messing with the graph
	z_in=tf.placeholder(tf.float32, shape=[None,latentSize], name='z_in')
	x_generated=decode(z_in)
	bound_test=encode(x_generated)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(graph=periodGraph,config=config)