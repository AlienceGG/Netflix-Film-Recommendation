from __future__ import division
import time
import numpy as np
import numpy.matlib as MB
from common import Scanner

class CFAlgorithm(object):

	def __init__(self, n=10000, m=10000):
		self.n = 10000
		self.m = 10000

	def __init_matrix(self, matrix, train_index):
		self.train_matrix = matrix
		self.train_index = train_index
		print "[log] init input matrix done."

	def calc_sim_matrix(self):
		start = time.clock()
		print "[log] start calculating similary matrix ......"
		norm_matrix = self.train_matrix / MB.repmat(np.sqrt(np.square(self.train_matrix).sum(axis=1)),self.train_matrix.shape[0],1).T
		sim_matrix = np.dot(norm_matrix, norm_matrix.T)
		np.fill_diagonal(sim_matrix, 0)
		end = time.clock()
		print "[log] calculate similary matrix done. time spent %0.2f seconds" % (end-start)
		return sim_matrix

	def calc_predication_matrix(self, sim_matrix):
		print "[log] start calculating train_matrix ......"
		up_matrix = np.dot(sim_matrix, self.train_matrix)
		down_matrix = np.dot(sim_matrix, self.train_index)

		matrix = None
		with np.errstate(divide='ignore', invalid='ignore'):
			matrix = np.true_divide(up_matrix, down_matrix)
			matrix = np.nan_to_num(matrix)
		print "[log] calculate score matrix done."
		return matrix

	def process(self, matrix, train_index):
		start = time.clock()

		self.__init_matrix(matrix, train_index)
		sim_matrix = self.calc_sim_matrix()
		p_matrix = self.calc_predication_matrix(sim_matrix)

		end = time.clock()
		print "[log] process collaborative filtering algorithm done, time spent %0.4f seconds." % (end - start)
		return p_matrix
