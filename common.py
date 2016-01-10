import numpy as np
import time
import math

def evaluate(p_matrix, t_matrix, t_index):
	rmse = np.sqrt(((p_matrix * t_index - t_matrix)**2).sum() / t_index.sum())
	return rmse

class Scanner(object):

	def __init__(self, user_filepath, film_filepath, score_filepath, n=10000, m=10000):
		self.user_filepath = user_filepath
		self.film_filepath = film_filepath
		self.score_filepath = score_filepath
		self.matrix = None
		self.n = n
		self.m = m
		self.user_map = {}

	def __get_users_map(self):
		self.user_map = {}
		user_num = 0
		with open(self.user_filepath) as f:
			for line in f:
				self.user_map[line.rstrip()] = user_num
				user_num += 1
		return self.user_map

	def get_matrix(self):
		user_map = self.__get_users_map()
		matrix = np.zeros((self.n, self.m))
		print "[log] start loading matrix"
		start = time.clock()
		with open(self.score_filepath) as f:
			for line in f:
				items = line.strip().split(" ")
				u_index = user_map[unicode(items[0])]
				f_index = int(items[1]) - 1
				score = int(items[2])
				matrix[u_index][f_index] = score

		end = time.clock()
		print "[log] load matrix done, time spent %.2f seconds" % (end - start)
		self.matrix = matrix
		return matrix

	def get_index(self):
		return np.int32(self.matrix > 0)

