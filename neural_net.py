import numpy as np
class nn:
	def __init__(self,lr=0.01,n_iter=100,layers):
		self.lr=lr,
		self.n_iter=n_iter
		self.parameters={}
		self.layers=layers
		self.caches=[]
		self.grads={}

	def initialize_params(self,layers):
		L=len(self.layers)
		for l in range(L):
			self.parameters['W'+str(l+1)]=np.random.randn(self.layers[l+1],self.layers[l])*0.01
			self.parameters['b'+str(l+1)]=np.zeros(self.layers[l+1],1)


	def forward_prop(self,X):
		A=X
		L=len(self.layers)
		for l in range(1,L+1):
			A_prev=A
			A,cache=linear_activation_forward(A_prev,self.parameters['W'+str(l)],self.parameters['b'+str(l)],activation='relu')
			self.caches.append(cache)

		AL,cache=linear_activation_forward(A_prev,self.parameters['W'+str(l)],self.parameters['b'+str(l)],activation='relu')
		self.caches.append(cache)
		return AL,self.caches

	def _linear_activation_forward(A_prev,W,b,activation):
		Z,linear_cache=linear_forward(A_prev,W,b)
		if activation=='sigmoid':
			A,activation_cache=sigmoid(Z)

		elif activation=='relu':
			A,activation_cache=relu(Z)

		cache=(linear_cache,activation_cache)

		return A,cache

	def _linear_forward(A_prev,W,b):
		Z=np.dot(W,A_prev) + b
		linear_cache=(A_prev,W,b)

		return Z,linear_cache
    

	def _sigmoid(Z):
		return 1/(1+np.exp(-Z))


	def _relu(Z):
		return np.maximum(0,Z)



	def compute_cost(AL,Y):
		m=Y.shape[1]
		cost=(-1/m)*np.sum(np.multiply(Y,np.log(AL)),np.multiply(1-Y,np.log(1-AL)))
		cost=np.squeeze(cost)
   		return cost


    def back_prop(self,Y,AL,caches):
    	L=len(caches)
    	print("length of caches and layers is:",L,len(self.layers))
    	m=AL.shape[1]
    	Y=Y.reshape(AL.shape)
    	dAL= - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    	current_cache=self.caches[-1]
    	self.grads["dA"+str(L)],self.grads["dW"+str(L)],self.grads["db"+str(L)]=
    	linear_backward(sigmoid_backward(dAL,current_cache[1]),current_cache[0])

    	for l in reversed(range(1,L)):
    		current_cache=self.caches[l-1]
    		linear_cache,activation_cache=current_cache[0],current_cache[1]
    		self.grads["dA"+str(l-1)],self.grads["dW"+str(l)],self.grads["db"+str(l)]=
    	    linear_backward( relu_backward(self.grads["dA"+str(l)],activation_cache) , linear_cache )

    	return self.grads


    def _linear_backward(self,dZ,cache):
    	A_prev,W,b=cache
    	m=A_prev.shape[1]

    	dW=np.dot(dZ,A_prev.T)/m
    	db=np.squeeze(np.sum(dZ,axis=1,keepdims=True))/m
    	dA_prev=np.dot(W.T,dZ)
        return dA_prev,dW,db



    def _sigmoid_backward(dA,cache):
    	Z=cache
    	s=1/(1+np.exp(-Z))
    	dZ=dA*s*(1-s)

    	return dZ

    def _relu_backward(dA,cache):
    	Z=cache
    	dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0

        return dZ


    def update_params(self,grads):
    	L=len(self.parameters)//2
    	for l in range(L):
    	self.parameters['W'+str(l+1)]-=self.lr*grads['dW'+str(l+1)]
    	self.parameters['b'+str(l+1)]-=self.lr*grads['db'+str(l+1)]
        return parameters



    def fit(self,X,y,lr,n_iter):
    	costs=[]
    	self.parameters=initialize_params(layers)
    	for i in n_iter:
    		AL,self.caches=forward_prop(self,X)
    		cost=compute_cost(AL,y)
    		self.grads=back_prop(self,y,AL,caches)
    		self.parameters=update_params(self,grads)
    		if i % 100 == 0:
            	print ("Cost after iteration %i: %f" %(i, cost))
            if i % 100 == 0:
            	costs.append(cost)
            
        	# plot the cost
        plt.plot(np.squeeze(costs))
       	plt.ylabel('cost')
       	plt.xlabel('iterations (per hundreds)')
       	plt.title("Learning rate =" + str(learning_rate))
       	plt.show()


    def predict(self,X,y):
    	m=X.shape[1]
    	n=len(self.parameters)//2
    	probabs,_= forward_prop(self,X)
    	for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    	#print results
    	#print ("predictions: " + str(p))
    	#print ("true labels: " + str(y))
    	print("Accuracy: "  + str(np.sum((p == y)/m)))


