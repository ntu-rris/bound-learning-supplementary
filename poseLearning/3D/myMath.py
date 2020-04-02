import numpy as np
import cv2

def combineRotVecAndTranslation(rotVec,translation):
	return combineRotMatAndTranslation(rotVecToRotMat(rotVec), translation)

def getTransformation(xAxis,yAxis,zAxis,origin):
	t=np.zeros([3,4])
	t[:,0]=xAxis
	t[:,1]=yAxis
	t[:,2]=zAxis
	t[:,3]=origin

	return t #3x4 transformation matrix (X,Y,Z,Origin)

def rotMatToRotVec(rotMat):
	rotVec,_=cv2.Rodrigues(rotMat)
	return np.ravel(rotVec)

def rotVecToRotMat(rotVec):
	rotMat,_=cv2.Rodrigues(rotVec)
	return rotMat

def getRotVecBetweenFrames(w_a,w_b):
	a_b = w_a[:3,:3].transpose() @ w_b[:3,:3]
	return rotMatToRotVec(a_b)

def allKeyExist(d,listOfKeys):
	return all(name in d for name in listOfKeys)

def getMissingMarkers(d,listOfKeys):
	a=[]
	for name in listOfKeys:
		if not (name in d):
			a.append(name) 
	return a

def length(v):
	return np.linalg.norm(v)

def normalize(v):
	return v/np.linalg.norm(v)

def crossAxis(v1,v2):
	return normalize(np.cross(v1,v2))

def homo(v):
	if(len(v.shape)==1 and v.shape[0]==3):
		return np.concatenate([v,np.array([1])],axis=0)
	elif(len(v.shape)==2 and v.shape[0]==3 and v.shape[1]==4):
		return np.concatenate([v,np.array([[0,0,0,1]])])
	elif(len(v.shape)==2 and v.shape[0]==3 and v.shape[1]==3):
		ans=np.identity(4)
		ans[:3,:3]=v
		return ans
	else:
		return v

def inverseHomo(T):
	ans=np.zeros([4,4])
	ans[3,3]=1
	ans[:3,:3]=T[:3,:3].transpose()
	ans[:3,3]= - ans[:3,:3] @ T[:3,3]
	return ans

def combine3axis(A_xAxisB,A_yAxisB,A_ZaxisB):
	A_Bframe=np.stack([A_xAxisB,A_yAxisB,A_ZaxisB],axis=-1)
	return A_Bframe

def combineRotMatAndTranslation(rotMat,tran):
	ans=homo(rotMat)
	ans[:3,3]=tran
	return ans

def doubleCrossAxis():
	return 0

def segmentR_segmentM_orientation():
	return 0

def genSphericalLine(layer=30,pointDistance=0.02):  #for display purpose only
	s=[]
	for zAngle in np.linspace(-np.pi/2,np.pi/2,num=layer,endpoint=True):
		z=np.sin(zAngle)
		if(zAngle==-np.pi/2 or zAngle==np.pi/2):
			s.append(np.array([[0,0,z]]))	#at the pole	(1,3)
		else:
			r=np.cos(zAngle)	#radius
			#calculate how many points needed in this layer
			n=np.ceil(2*np.pi*r/pointDistance)
			hAngle=np.linspace(0,2*np.pi,num=n,endpoint=True)
			s.append(np.stack([r*np.cos(hAngle),r*np.sin(hAngle),np.array([z]*int(n))],axis=-1))

	return np.concatenate(s,axis=0).astype(dtype='float32')

def mirrorPelvis(a):	#(m,3)	#rotVec
	modifier=np.array([	#mirror axis Y and Z of pelvis across lab_XY plane
		[-1, 1, 1],
		[ 1,-1,-1],
		[-1, 1, 1]
	],dtype='float32')
	ans=np.zeros(a.shape)
	for i in range(a.shape[0]):
		rotVec=a[i,:]	
		ans[i,:]=rotMatToRotVec(rotVecToRotMat(rotVec)*modifier)
	return ans