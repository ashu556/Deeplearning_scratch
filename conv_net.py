import numpy as np

class conv:
	def __init__(self):
		self.W=None
		self.b=None
		self.hparameters={}

	def pad(self,X,pad):
		X_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)))
		return X_pad

	def conv_forward(self,A_prev):
		(m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
		(f,f,n_C_prev,n_C) = self.W.shape
		W=self.W
		b=self.b
		stride=hparameters['stride']
		pad=hparameters['pad']
		n_H=int((n_H_prev+2*pad-f)/stride)+1
		n_W=int((n_W_prev+2*pad-f)/stride)+1

		Z=np.zeros((m,n_H,n_W,n_C))
		A_prev_pad=pad(A_prev,pad)

		for i in range(m):
			a_prev_pad=A_prev_pad[i]
			for h in range(n_H):
				for w in range(n_W):
					for c in range(n_C):
						vert_start= h*stride
						vert_end=vert_start+f
						horiz_start=w*stride
						horiz_end=horiz_start+f

						a_slice_prev=a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

						Z[i,h,w,c]=_con_single_step(a_slice_prev,W[...,c],b[...,c])

		conv_cache=(A_prev,W,b,hparameters)
		return Z,conv_cache


	def _con_single_step(a_slice_prev,W,b):
		s=np.multiply(a_slice_prev,W)+b
		Z=np.sum(s)

		return Z

	def pool_forward(A_prev,hparameters,mode='max'):
		(m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
		(f,f,n_C_prev,n_C) = self.W.shape
		W=self.W
		b=self.b
		stride=hparameters['stride']
		pad=hparameters['pad']
		n_H=int((n_H_prev-f)/stride)+1
		n_W=int((n_W_prev-f)/stride)+1
		n_C=n_C_prev

		A=np.zeros((m,n_h,n_W,n_C))

		for i in range(m):
			for h in range(n_H):
				for w in range(n_W):
					for c in range(n_C):
						vert_start=h*stride
						vert_end=vert_start+f
						horiz_start=w*stride
						horiz_end=horiz_start+f

						a_slice_prev=A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]

						if mode=='max':
							A[i,h,w,c]=np.max(a_slice_prev)

						elif mode=='average':
							A[i,h,w,c]=np.mean(a_slice_prev)


		pool_cache=(A_prev,hparameters)
		return A,pool_cache


	def conv_backward(dZ,conv_cache):
		A_prev,W,b,hparameters=conv_cache
		(m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
		f,f,n_C_prev,n_C=W.shape
		stride=hparameters['stride']
		pad=hparameters['pad']
		m,n_H,n_W,n_C=dZ.shape
		dA_prev=np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
		dW=np.zeros((f,f,n_C_prev,n_C))
		db=np.zeros((1,1,1,n_C))
		A_prev_pad=pad(A_prev,pad)
		dA_prev_pad=pad(dA_prev,pad)

		for i in range(m):
			a_prev_pad=A_prev_pad[i]
			da_prev_pad=dA_prev_pad[i]

			for h in range(n_H):
				for w in range(n_W):
					for c in range(n_C):
						vert_start=h*stride
						vert_end=vert_start+f
						horiz_start=h*stride
						horiz_end=horiz_start+f

						a_slice=a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

						da_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]+= W[:,:,:,c]*dZ[i,h,w,c]
						dW[:,:,:,c]+=a_slice*dZ[i,h,w,c]
						db[:,:,:,c]+=dZ[i,h,w,c]

			dA_prev[i,:,:,:]=dA_prev_pad[pad:-pad,pad:-pad,:]


		return dA_prev,dW,db


	def pool_backward(dA,pool_cache,mode='max'):
		(A_prev,hparameters)=pool_cache
		f=hparameters['f']
		stride=hparameters['stride']
		(m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
		n_H,n_W,n_C=dA.shape
		dA_prev=np.zeros((A_prev.shape))
		for i in range(m):
			a_prev=A_prev[i]
			for h in range(n_H):
				for w in range(n_W):
					for c in range(n_C):
						vert_start=h*stride
						vert_end=vert_start+f
						horiz_start=h*stride
						horiz_end=horiz_start+f
						if mode=='max':
							a_prev_slice=a_prev[vert_start:vert_end,horiz_start:horiz_end,:]
							mask=_create_mask(a_prev_slice)
							dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]+= np.multiply(mask,dA[i,h,w,c])

						elif mode=="average":
							da=dA[i,h,w,c]
							shape=(f,f)
							dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]+= _distribute_values(da,shape)					
		return dA_prev


	def _create_mask(a_prev_slice):
		mask=a_prev_slice==np.max(a_prev_slice)
		return mask

	def _distribute_values(da,shape):
		n_H,n_W=shape
		avearge=da/(n_H*n_W)
		a=np.ones(shape)*average
		return a


