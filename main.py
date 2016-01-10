#coding: utf-8

import sys
import numpy as np
import time
from common import Scanner, evaluate
from cf_algo import CFAlgorithm
from gd_algo import GDAlgorithm

#data root
user_file = "data/users.txt"
film_file = "data/movie_titles.txt"
train_file = "data/netflix_train.txt"
test_file = "data/netflix_test.txt"

#result root
compare_result_file="res/compare_result.txt"
gd_result_file="res/gd_result.txt"

class Processor(object):
	def __init__(self):
		pass

	def process_cf_algo(self):
		start = time.clock()

		train, train_index = self._get_train()
		test, test_index = self._get_test()
	
		cf = CFAlgorithm()
		p_matrix = cf.process(train, train_index)
		rmse = self._evaluate(p_matrix, test, test_index)

		end = time.clock()
		print "[log] process Collaborative Filtering Algorithm done. time spent %0.2f seconds totally, RMSE = %0.4f" % (end-start, rmse)

	def process_gd_algo(self, res_file):
		start = time.clock()
		train, train_index = self._get_train()
		test, test_index = self._get_test()

		gd = GDAlgorithm()
		res = gd.process(train, test, test_index)

		with open(res_file, "w") as f:
			for r in res:
				f.write('{} {} {}\n'.format(r[0], r[1], r[2]))

		end = time.clock()
		print "[log] process Matrix Decompose algorithm done, time spent %0.2f seconds." % (end - start)

	def compare_kr(self, res_file):
		train, train_index = self._get_train()
		test, test_index = self._get_test()

		res_list = []
		ks = [10, 20, 30, 40, 50, 60, 70]
		rs = [0.001, 0.01, 0.025, 0.05, 0.075, 0.10]
		for k in ks:
			r = 0.01
			gd = GDAlgorithm(k=k, lamb=r, compare=True)
			res = gd.process(train, test, test_index)
			res_list.append((k, res[len(res)-1][2]))

		for r in rs:
			k = 50
			gd = GDAlgorithm(k=k, lamb=r, compare=True)
			res = gd.process(train, test, test_index)
			res_list.append((r, res[len(res)-1][2]))

		with open(res_file, "w") as f:
			for r in res_list:
				f.write('{} {}\n'.format(r[0], r[1]))

	def _evaluate(self, p_matrix, test, test_index):
		return evaluate(p_matrix, test, test_index)

	def _get_test(self):
		s = Scanner(user_file, film_file, test_file)
		return s.get_matrix(), s.get_index()

	def _get_train(self):
		s = Scanner(user_file, film_file, train_file)
		return s.get_matrix(), s.get_index()


def command():
	print "1: Collaborative Filtering\n2: Matrix Decomposition with Gradient Descent Algorithm\n3: Comparison with k & lambda\n"
	processor = Processor()
	try:
		index = int(raw_input("input 1,2,3: (eg:2) "))
	except ValueError as e:
		print "illegal"
	else:
		if index == 1:
			processor.process_cf_algo()
		elif index == 2:
			processor.process_gd_algo()
		elif index == 3:
			processor.compare_kr(compare_result_file)
		else:
			print "illegal"

if __name__ == '__main__':
	command()