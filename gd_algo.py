import time
import math
import numpy as np
from numpy import linalg as LA
from common import Scanner, evaluate

class GDAlgorithm(object):

	def __init__(self, n=10000, m=10000, alpha=0.0001, k=50, lamb=0.01, threshold=0.001, compare=False):
		self.n = 10000
		self.m = 10000
		self.alpha = alpha
		self.k = k
		self.lamb = lamb
		self.threshold = threshold
		self.compare = compare

	def __init_matrix(self, matrix, test, test_index):
		self.matrix = matrix
		self.test = test
		self.test_index = test_index

		self.u_matrix, self.v_matrix = self.__decompose(matrix, self.k)
		self.uv_matrix = np.dot(self.u_matrix, self.v_matrix.T)
		self.a_matrix = np.int32(self.matrix > 0)
		print "[log] init matrixs done."

	def __decompose(self, matrix, k):
		u_matrix = np.zeros((self.n, k))
		v_matrix = np.zeros((self.m, k))

		for i in xrange(self.n):
			for j in xrange(k):
				if not self.compare:
					u_matrix[i][j] = np.random.rand()/1000#0.000001 * i
					v_matrix[i][j] = np.random.rand()/1000#0.000001 * (self.n-i+1)
				else:
					u_matrix[i][j] = 0.000001 * i
					v_matrix[i][j] = 0.000001 * (self.n-i+1)
		return u_matrix, v_matrix

	def __calc_derivative(self):
		t_matrix = self.a_matrix * (self.uv_matrix - self.matrix)
		derivative_u = np.dot(t_matrix, self.v_matrix) + 2 * self.lamb * self.u_matrix
		derivative_v = np.dot(t_matrix.T, self.u_matrix) + 2 * self.lamb * self.v_matrix
		return derivative_u, derivative_v

	def __calc_target(self):
		sum = 0.5 * math.pow(LA.norm(self.a_matrix * (self.matrix - self.uv_matrix)), 2)
		sum += self.lamb * math.pow(LA.norm(self.u_matrix), 2)
		sum += self.lamb * math.pow(LA.norm(self.v_matrix), 2)
		return sum

	def process(self, train, test, test_index):
		self.__init_matrix(train, test, test_index)

		iteration = 1
		diff = 100
		js = 0
		res = []
		
		while (diff > self.threshold or iteration < 10) and (iteration <= 300):
			istart = time.clock()
		
			u, v = self.__calc_derivative()
			self.u_matrix -= self.alpha * u
			self.v_matrix -= self.alpha * v
			self.uv_matrix = np.dot(self.u_matrix, self.v_matrix.T)

			jsn = self.__calc_target()
			diff = abs(jsn - js) / jsn
			js = jsn
			rmse = evaluate(self.uv_matrix, self.test, self.test_index)
			res.append((iteration, jsn, rmse))
			iend = time.clock()
			print "[log] k = %d lambda= %0.4f" % (self.k, self.lamb)
			print """[log] now %d iterations: \n[log] js = %.4f \n[LOG] diff = %.4f\n[LOG] rmse = %.5f\n[log] this iteration time spent %.2f seconds\n""" \
				% (iteration, jsn, diff, rmse, (iend - istart))
			iteration += 1

		return res
