import numpy as np
from neural_net import nn

class initialization(nn):
	def __init__(self,lr=0.01,n_iter=100,layers):
		self.lr=lr,
		self.n_iter=n_iter
		self.parameters={}
		self.layers=layers
		self.caches=[]
		self.grads={}



    
      ###### INITIALIZING PARAMETERS ######
	
	def initialize_params_zeros(self,layers):
		L=len(self.layers)
		for l in range(1,L+1):
			self.parameters['W'+str(l)]=np.zeros(self.layers[l],self.layers[l-1])*0.01
			self.parameters['b'+str(l)]=np.zeros(self.layers[l],1)

	def initialize_params(self):
		L=len(self.layers)
		for l in range(1,L+1):
			self.parameters['W'+str(l)]=np.random.randn(self.layers[l],self.layers[l-1])*0.01
			self.parameters['b'+str(l)]=np.zeros(self.layers[l],1)

	def initialize_params_he(self,layers):
		L=len(self.layers)
		for l in range(1,L+1):
			self.parameters['W'+str(l)]=np.random.randn(self.layers[l],self.layers[l-1])*np.sqrt(2/self.layers[l-1])
			self.parameters['b'+str(l)]=np.zeros(self.layers[l],1)
      


      ######  COST FUNCTION WITH REGULARIZATION #######
	

	def compute_cost_regularization(self,AL,Y,parameters,lambd):
		L=len(self.layers)
		m=Y.shape[1]
		regularised_cost=0
		for l in range(1,L+1):
			regularised_cost+=np.sum(np.square(self.parameters['W'+str(l)]))	
		cost=self.compute_cost(AL,Y) + lambd*regularised_cost/(2*m)
		return cost

      ###### BACKPROP WITH REGULARIZATION #######


	def back_prop_regularization(self,AL,Y,lambd):
		L=len(self.caches)
		print("length of caches and layers is:",L,len(self.layers))
		m=AL.shape[1]
		Y=Y.reshape(AL.shape)
		dAL= - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
		dZ=AL-Y
		current_cache=self.caches[-1]
		self.grads["dA"+str(L)],self.grads["dW"+str(L)],self.grads["db"+str(L)]=_linear_backward_regularization(dZ,current_cache[0])

		for l in reversed(range(1,L)):
			current_cache=self.caches[l-1]
			linear_cache,activation_cache=current_cache[0],current_cache[1]
			self.grads["dA"+str(l-1)],self.grads["dW"+str(l)],self.grads["db"+str(l)]= _linear_backward_regularization( relu_backward(self.grads["dA"+str(l)],activation_cache) , linear_cache )

		return self.grads


	def _linear_backward_regularization(self,dZ,cache):
		A_prev,W,b=cache
		m=A_prev.shape[1]

		dW=np.dot(dZ,A_prev.T)/m+lambd*W/m
		db=np.squeeze(np.sum(dZ,axis=1,keepdims=True))/m
		dA_prev=np.dot(W.T,dZ)
		return dA_prev,dW,db





                       ############## USING DROPOUT ################


	def forward_prop_dropout(self,X,keep_prob=0.5):
		A=X
		L=len(self.layers)
		for l in range(1,L):
			A_prev=A
			A,cache=linear_activation_forward_dropout(A_prev,self.parameters['W'+str(l)],self.parameters['b'+str(l)],activation='relu',keep_prob)
			self.caches.append(cache)

		AL,cache=linear_activation_forward_dropout(A_prev,self.parameters['W'+str(l)],self.parameters['b'+str(l)],activation='relu',keep_prob)
		self.caches.append(cache)
		return AL,self.caches

	def _linear_activation_forward_dropout(A_prev,W,b,activation,keep_prob):
		Z,linear_cache=linear_forward(A_prev,W,b)
		if activation=='sigmoid':
			A,activation_cache=sigmoid(Z)

		elif activation=='relu':
			A,activation_cache=relu(Z)
			D=np.random.randn(A.shape[0],A.shape[1])
			D=D<keep_prob
			A=A*D
			A=A/keep_prob
		linear_cache.append(D)

		cache=(linear_cache,activation_cache)

		return A,cache


	def back_prop_dropout(self,AL,Y,cache,keep_prob):
		L=len(self.caches)
		print("length of caches and layers is:",L,len(self.layers))
		m=AL.shape[1]
		Y=Y.reshape(AL.shape)
		dAL= - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
		dZ=AL-Y
		current_cache=self.caches[-1]
		self.grads["dA"+str(L)],self.grads["dW"+str(L)],self.grads["db"+str(L)]=_linear_backward_dropout(dZ,current_cache[0])

		for l in reversed(range(1,L)):
			current_cache=self.caches[l-1]
			linear_cache,activation_cache=current_cache[0],current_cache[1]
			self.grads["dA"+str(l-1)],self.grads["dW"+str(l)],self.grads["db"+str(l)]=_linear_backward_dropout( relu_backward(self.grads["dA"+str(l)],activation_cache) , linear_cache )

		return self.grads


	def _linear_backward_dropout(self,dZ,cache):
		A_prev,W,b,D=cache
		m=A_prev.shape[1]

		dW=np.dot(dZ,A_prev.T)/m
		db=np.squeeze(np.sum(dZ,axis=1,keepdims=True))/m
		dA_prev=np.dot(W.T,dZ)
		dA_prev*=D
		dA_prev/=keep_prob
		return dA_prev,dW,db



     ########### CREATING MINI BATCHES ############



	def create_batches(X,Y,batch_size=64):
		m=X.shape[1]
		mini_batches=[]
		permutation=list(np.random.permutation(m))
		shuffled_X=X[:,permutation]
		shuffled_Y=Y[:,permutation]
		n_complete_batches=math.floor(m/batch_size)

		for k in range(n_complete_batches):
			mini_batch_x=shuffled_X[:,k*batch_size:(k+1)*batch_size]
			mini_batch_y=shuffled_Y[:,k*batch_size:(k+1)*batch_size]
			batch=[mini_batch_x,mini_batch_y]
			mini_batches.append(batch)

		if m%batch_size!=0:
			mini_batch_x=shuffled_X[:,n_complete_batches*batch_size,:]
			mini_batch_y=shuffled_Y[:,n_complete_batches*batch_size,:]
			batch=[mini_batch_x,mini_batch_y]
			mini_batches.append(batch)

		return mini_batches


###############  UPDATING PARAMETERS WITH MOMENTUM ################


	def _initialize_velocity(self):
		L=len(self.parameters)//2
		v={}
		for l in range(L):
			v['dW'+str(l+1)]=np.zeros_like(parameters['W'+str(l+1)])
			v['db'+str(l+1)]=np.zeros_like(parameters['b'+str(l+1)])

		return v


	def update_params_momentum(self,beta):
		v=_initialize_velocity(self)
		L=len(self.parameters) // 2

		for l in range(L):
			v['dW'+str(l+1)]=beta*v['dW'+str(l+1)] + (1-beta)*self.grads['dW'+str(l+1)]
			v['db'+str(l+1)]=beta*v['db'+str(l+1)] + (1-beta)*self.grads['db'+str(l+1)]

			self.parameters['W'+str(l+1)]-=self.lr*v['dW'+str(l+1)]
   			self.parameters['b'+str(l+1)]-=self.lr*v['db'+str(l+1)]



    ################# ADAM OPTIMIZATION ###################

    def _initialize_adams(self):
    	L=len(self.parameters)//2
		v={}
		s={}
		for l in range(L):
			v['dW'+str(l+1)]=np.zeros_like(parameters['W'+str(l+1)])
			v['db'+str(l+1)]=np.zeros_like(parameters['b'+str(l+1)])

			s['dW'+str(l+1)]=np.zeros_like(parameters['W'+str(l+1)])
			s['db'+str(l+1)]=np.zeros_like(parameters['b'+str(l+1)])

		return v,s

	def update_params_adams(self,beta1,beta2,epsilon,t):
		v,s=_initialize_adams(self)
		L=len(self.parameters) // 2
		v_corrected={}
		s_corrected={}

		for l in range(L):
			v['dW'+str(l+1)]=beta*v['dW'+str(l+1)] + (1-beta)*self.grads['dW'+str(l+1)]
			v['db'+str(l+1)]=beta*v['db'+str(l+1)] + (1-beta)*self.grads['db'+str(l+1)]

			v_corrected['dW'+str(l+1)]=v['dW'+str(l+1)]/(1-np.power(beta1,t))
			v_corrected['db'+str(l+1)]=v['db'+str(l+1)]/(1-np.power(beta1,t))

			s['dW'+str(l+1)]=beta2*s['dW'+str(l+1)] + (1-beta2)*np.power(self.grads['dW'+str(l+1)],2)
			s['db'+str(l+1)]=beta2*s['db'+str(l+1)] + (1-beta2)*np.power(self.grads['db'+str(l+1)],2)

			s_corrected['dW'+str(l+1)]=s['dW'+str(l+1)]/(1-np.power(beta2,t))
			s_corrected['db'+str(l+1)]=s['db'+str(l+1)]/(1-np.power(beta2,t))

			self.parameters['W'+str(l+1)]-=self.lr*v_corrected['dW'+str(l+1)]/np.sqrt(s_corrected['dW'+str(l+1)]+epsilon)
    		self.parameters['b'+str(l+1)]-=self.lr*v_corrected['db'+str(l+1)]/np.sqrt(s_corrected['db'+str(l+1)]+epsilon)

 


############## CHECKING OPTIMIZATION #################


    def check_optimization(self,X,Y,epsilon=1e-7):
    	params=dictionary_to_vector(self.parameters)
    	grad=gradients_to_vector(gradients)
    	num_parameters=params.shape[0]
    	J_plus=np.zeros((num_parameters,1))
    	J_minus=np.zeros((num_parameters,1))
    	grad_approx=np.zeros((num_parameters,1))

    	for i in range(num_parameters):
    		theta_plus=np.copy(params)
    		theta_plus[i][0]=theta_plus[i][0]+epsilon
    		AL,_=forward_prop(self,X)
    		J_plus[i]=compute_cost(AL,Y)
    		theta_minus=np.copy(params)
    		theta_minus[i][0]=theta_minus[i][0]-epsilon
    		AL,_=forward_prop(self,X)
    		J_minus[i]=compute_cost(AL,Y)
    		grad_approx[i]=J_plus[i]-J_minus[i]/(2*epsilon)


    	numerator=np.linalg.norm(grad-grad_approx)
    	denominator=np.linalg.norm(grad)+np.linalg.norm(grad_approx)

    	diff=numerator/denominator

    	if diff>1e-7:
    		print("backprop is not correct")
    	else:
    		print("backprop is correct")

    	return diff