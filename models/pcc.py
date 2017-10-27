import numpy as np
from criteria import rank_loss

class PCC:
	def __init__(self, cost, base_learner, params={}, n_sample=100):
		self.cost = cost
		self.base_learner = base_learner
		self.params = params
		self.n_sample = n_sample

	def fit(self, x_train, y_train):
		self.K = y_train.shape[1]
		self.clfs = [self.base_learner(**self.params) for i in xrange(self.K)]
		for i in xrange(self.K):
			self.clfs[i].fit(np.concatenate((x_train, y_train[:, :i]), axis=1), y_train[:, i])

	def predict(self, x_test):
		r_test = self.predict_prob(x_test)
		p_test = np.zeros((x_test.shape[0], self.K), dtype=int)
		for i in xrange(x_test.shape[0]):
			p_test[i, :] = self.predict_one(x_test[i, :], r_test[i, :])
		return p_test

	def predict_prob(self, x_test):
		r_test = np.zeros((x_test.shape[0], self.K))
		for i in xrange(self.K):
			r_test[:, i] = 1.0 - self.clfs[i].predict_proba(np.concatenate((x_test, (r_test[:, :i]>0.5).astype(int)), axis=1))[:, 0]
		return r_test

	def predict_one(self, x, pb):
		if self.cost == 'ham':
			return (pb>0.5).astype(int)
		prob = np.repeat(pb, self.n_sample).reshape((pb.shape[0], self.n_sample)).T
		y_sample = (np.random.random((self.n_sample, self.K))<prob).astype(int)
		if self.cost == 'rank':
			thr = 0.0
			pred = (pb>thr).astype(int)
			p_sample = np.repeat(pred, self.n_sample).reshape((pred.shape[0], self.n_sample)).T
			score = rank_loss(y_sample, p_sample).mean()
			for p in pb:
				pred = (pb>p).astype(int)
				p_sample = np.repeat(pred, self.n_sample).reshape((pred.shape[0], self.n_sample)).T
				score_t = rank_loss(y_sample, p_sample).mean()
				if score_t < score:
					score = score_t
					thr = p
			return (pb>thr).astype(int)
		elif self.cost == 'f1':
			s_idxs = y_sample.sum(axis=1)
			P = np.zeros((self.K, self.K))
			for i in xrange(self.K):
				P[i, :] = y_sample[s_idxs==(i+1), :].sum(axis=0)*1.0/self.n_sample

			W = 1.0 / (np.cumsum(np.ones((self.K, self.K)), axis=1) + np.cumsum(np.ones((self.K, self.K)), axis=0))
			F = P*W
			idxs = (-F).argsort(axis=1)
			H = np.zeros((self.K, self.K), dtype=int)
			for i in xrange(self.K):
				H[i, idxs[i, :i+1]] = 1
			scores = (F*H).sum(axis=1)
			pred = H[scores.argmax(), :]
			# if (s_idxs==0).mean() > 2*scores.max():
			# 	pred = np.zeros((self.K, ), dtype=int)
			return pred

