from random import *
from math import *
import numpy as np

#Readind training data into an array
def training_data():
	train_data = open('hw3trainingdata.txt','r')
	x_train = []
	y_train = []

	for line in train_data:
		num = line.strip('\r\n').split(' ')
		x_train.append([1,float(num[1])])
		y_train.append([float(num[2])])

	return x_train, y_train

#Readind testing data into an array
def testing_data():
	test_data = open('hw3testingdata.txt','r')
	x_test = []
	y_test = []

	for line in test_data:
		num = line.strip('\r\n').split(' ')
		x_test.append([1,float(num[1])])
		y_test.append([float(num[2])])

	return x_test, y_test

#Sigmoid Function - f()
def sigmoid(x):
    f = 1.0/(1.0 + np.exp(-x*10.0))
    return f

#Differential of the sigmoid function - f'()
def sigmoid_prime(x):
    fp = sigmoid(x)*(1.0-sigmoid(x))
    return fp

#initializing the weights
def initialize_weights(m,n,o):
	#wjk is the weight from hidden neuron j and x_train k 
	w = []
	for j in range(n):
		w1 = []
		for k in range(o+1):
			weight = uniform(-0.5,0.5)
			w1.append(weight)
		w.append(w1)
	
	#Wij is the weight from output i and hidden neuron j	
	W = []
	for i in range(m):
		W1 = []
		for j in range(n+1):
			weight = uniform(-0.5,0.5)
			W1.append(weight)
		W.append(W1)

	return w, W

#Calculating (y-y_est)2
def error(y,y_est):
	e = (y-y_est)**2
	return e

#algorithm running forward pass
def forward_pass(w,W,l,x_train,m ,n,o):
	#algorithm to calculate hj
	h = []
	s_j = []
	for j in range(n):
		s = 0
		for k in range(o+1):
			s += w[j][k] * x_train[l][k]
		s_j.append(s)
		h.append(sigmoid(s))

	
	#algorithm to calculate estimated error	
	y_est = []
	s_i = []
	for i in range(m):
		s=0
		for j in range(n):
			s+=W[i][j]*h[j]
		s_i.append(s)
		y_est.append(sigmoid(s))

	return s_i, s_j, h, y_est

#algorithm running backward pass
def backward_pass(s_i,s_j,y_est,W,l,y_train,m,n,o):
	del_i = []
	for i in range(m):
		delta = ((y_train[l][i-1]) - y_est[i])*sigmoid_prime(s_i[i])
		del_i.append(delta)
	
	del_j = []
	for j in range(n):
		delta = None
		for i in range(m):
			delta = del_i[i]*W[i][j]
		delta *= sigmoid_prime(s_j[j])
		del_j.append(delta)

	return del_i, del_j	

#training neurons and updating weights till we reach a stopping condition
def train_neurons(stop, eeta,m,n,o):
	w,W = initialize_weights(m,n,o)
	J = 10
	x_train, y_train = training_data()
	epoch = 0

	while J>stop:
		for l in range(len(x_train)):
			s_i, s_j, h, y_est = forward_pass(w,W,l,x_train,m,n,o)
			del_i, del_j = backward_pass(s_i,s_j,y_est,W,l,y_train,m,n,o)

			for i in range(m):
				for j in range(n):
					W[i][j] += eeta*del_i[i]*h[j]

			for j in range(n):
				for k in range(o+1):
					w[j][k] += eeta*del_j[j]*x_train[l][k]

		J=0
		for l in range(len(x_train)):
			s_i, s_j, h, y_est = forward_pass(w,W,l,x_train,m,n,o)

			for i in range(m):
				J += error(y_train[l][i], y_est[i])

		epoch += 1
		print epoch
		print J

	W=W
	w=w

	return J

#Testing the data using the trained weights
def test_neurons(x_test,y_test, eeta,m,n,o):
	W=W
	w=w
	final_J = 0
	for l in range(len(x_test)):
		s_i, s_j, h, y_est = forward_pass(w,W,l,x_test,m,n,o)

		J=0
		for i in range(m):
			J += error(y_test[l][i], y_est[i])

		final_J += J

	return final_J

J = train_neurons(0.01,0.1,1,2,1)