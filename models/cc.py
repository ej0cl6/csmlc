import numpy as np

class CC:
	def __init__(self, base_learner, params={}):
		self.base_learner = base_learner
		self.params = params

	def fit(self, x_train, y_train):
		self.K = y_train.shape[1]
		self.clfs = [self.base_learner(**self.params) for i in xrange(self.K)]
		for i in xrange(self.K):
			self.clfs[i].fit(np.concatenate((x_train, y_train[:, :i]), axis=1), y_train[:, i])

	def predict(self, x_test):
		p_test = np.zeros((x_test.shape[0], self.K), dtype=int)
		for i in xrange(self.K):
			p_test[:, i] = self.clfs[i].predict(np.concatenate((x_test, p_test[:, :i]), axis=1))
		return p_test
