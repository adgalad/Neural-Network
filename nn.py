
from math import e
import numpy

eps = pow(10,-6)
class NN:
	def __init__(self):
		self.nin = 0
		self.nhidden = 0
		self.nout = 0
		self.b0 = []
		self.b1 = []
		self.inputsW  = []
		self.hiddensW = []
		self.hiddensO = []
		self.outputsD = []
		self.outputsO = []
		self.errorX = []
		self.errorY = []
		
	def sigmoid(self,x):
		return 1.0/(1.0 + pow(e,-x))

	def evaluate(self, x):
		self.propagation(x)
		return self.outputsO

	def propagation(self,x):	
		for k in range(self.nout):
			s = self.b1[k]
			for j in range(self.nhidden):
				s1 = self.b0[j]
				for i in range(self.nin):
					s1 += self.inputsW[i][j] * x[i]
				self.hiddensO[j] = self.sigmoid(s1)
				s += self.hiddensW[j][k] * self.hiddensO[j]
			self.outputsO[k] = self.sigmoid(s)

	def backPropagation(self,trainingSet, n, nin, nhidden, nout):
		self.nin = nin
		self.nhidden = nhidden
		self.nout = nout
		# BIAS
		self.b0 = numpy.zeros(nhidden)
		self.b1 = numpy.zeros(nout)

		# Input, hidden and output layer vectors (O: output of the neuron, W: weight, D: delta)
		self.inputsW  = numpy.zeros((self.nin,self.nhidden))

		self.hiddensW = numpy.zeros((self.nhidden,self.nout))
		self.hiddensO = numpy.zeros((self.nhidden))
		self.hiddensD = numpy.zeros(self.nhidden)

		self.outputsD = numpy.zeros(self.nout)
		self.outputsO = numpy.zeros(self.nout)

		# Initializing with random values (always the same values because the seed)
		numpy.random.seed(43124)
		for i in range(nin):
			self.inputsW[i] = numpy.random.randint(-500,500)/1000.0
		for j in range(nhidden):
			self.hiddensW[j] = numpy.random.randint(-500,500)/1000.0
			self.b0[j] = numpy.random.randint(-500,500)/1000.0
		for k in range(nout):
			self.b1[k] = numpy.random.randint(-500,500)/1000.0
		
		ite = 0
		# train with 500 iterations
		while(ite < 500 ):
			for x,t in trainingSet:
				# Forward Propagation
				self.propagation(x)
				
				# Back Propagation
				for k in range(self.nout):
					self.outputsD[k] = self.outputsO[k]*(1-self.outputsO[k])*(t[k]-self.outputsO[k])		

				for j in range(self.nhidden):
					suma = 0
					for k in range(self.nout):
						suma += self.hiddensW[j][k]*self.outputsD[k]
					self.hiddensD[j] = self.hiddensO[j]*(1-self.hiddensO[j])*suma
					for i in range(self.nin):
						self.inputsW[i][j] += n*self.hiddensD[j]*x[i]
					self.b0[j] += n*self.hiddensD[j]

				for k in range(self.nout):
					for j in range(self.nhidden):
						self.hiddensW[j][k] += n*self.outputsD[k]*self.hiddensO[j]
					self.b1[k] += n*self.outputsD[k]
			ite += 1
			print(ite)
			




		
















