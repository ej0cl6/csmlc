import numpy as np
from criteria import hamming_loss, rank_loss, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from mdsw import MDSW

class CLEMS:
	def __init__(self, cost, base_learner, params={}):
		self.cost = cost
		self.base_learner = base_learner
		self.params = params
		if self.cost == 'ham':
			self.dis = hamming_loss
		elif self.cost == 'rank':
			self.dis = rank_loss
		elif self.cost == 'f1':
			self.dis = lambda x1, x2: 1.0 - f1_score(x1, x2)
		elif self.cost == 'acc':
			self.dis = lambda x1, x2: 1.0 - accuracy_score(x1, x2)

	def fit(self, x_train, y_train):
		self.K = y_train.shape[1]
		self.z_dim = self.K

		# get unique label vectors
		bb = np.ascontiguousarray(y_train).view(np.dtype((np.void, y_train.dtype.itemsize * y_train.shape[1])))
		_, idx = np.unique(bb, return_index=True)
		self.y_train_uq = y_train[idx]
		num_uq = self.y_train_uq.shape[0]

		self.nn_y_uq = NearestNeighbors(n_neighbors=1)
		self.nn_y_uq.fit(self.y_train_uq)

		# calculate weight
		uq_weight = self.cal_count(y_train)

		# calculate delta matrix
		delta = np.zeros((2*num_uq, 2*num_uq))
		for i in xrange(num_uq):
			for j in xrange(num_uq):
				delta[i, num_uq+j] = np.sqrt(self.dis(self.y_train_uq[None, i], self.y_train_uq[None, j]))
				delta[num_uq+j, i] = delta[i, num_uq+j]

		# calculate MDS embedding
		mds = MDSW(n_components=self.z_dim, n_uq=num_uq, uq_weight=uq_weight, max_iter=300, eps=1e-6, dissimilarity="precomputed", n_init=8, n_jobs=1)
		z_train_uq = mds.fit(delta).embedding_

		self.nn_z_uq = NearestNeighbors(n_neighbors=1)
		self.nn_z_uq.fit(z_train_uq[num_uq:])

		_dis, _idxs = self.nn_y_uq.kneighbors(y_train)
		z_train_uq[_idxs[:, 0]]
		z_train = z_train_uq[_idxs[:, 0]]

		# train regressor
		self.rgrs = [self.base_learner(**self.params) for i in xrange(self.K)]
		for i in xrange(self.z_dim):
			self.rgrs[i].fit(x_train, z_train[:, i])

	def predict(self, x_test):
		z_test = np.zeros((x_test.shape[0],self.z_dim))
		for i in xrange(self.z_dim):
			z_test[:, i] = self.rgrs[i].predict(x_test)

		_dis, _idxs = self.nn_z_uq.kneighbors(z_test)
		p_test = self.y_train_uq[_idxs[:, 0]]
		return p_test

	def cal_count(self, y_train):
		_dis, _idxs = self.nn_y_uq.kneighbors(y_train)
		idxs = _idxs[:, 0]
		uq_, uq_weight = np.unique(idxs, return_counts=True)
		return uq_weight


