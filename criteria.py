def hamming_loss(y_test, p_test):
	return 1.0 - (p_test==y_test).mean(axis=1)

def rank_loss(y_test, p_test):
	revloss = 1.0 * ((y_test==1) & (p_test==0)).sum(axis=1) * ((y_test==0) & (p_test==1)).sum(axis=1)
	eq0loss = 0.5 * ((y_test==1) & (p_test==0)).sum(axis=1) * ((y_test==0) & (p_test==0)).sum(axis=1)
	eq1loss = 0.5 * ((y_test==1) & (p_test==1)).sum(axis=1) * ((y_test==0) & (p_test==1)).sum(axis=1)
	return (revloss + eq0loss + eq1loss)

def f1_score(y_test, p_test):
	v1 = 2.0*(p_test*y_test).sum(axis=1)
	v2 = p_test.sum(axis=1) + y_test.sum(axis=1)
	v1[v2<=0] = 1.0
	v1[y_test.sum(axis=1)<=0] = 1.0
	v1[v2>0] /= v2[v2>0]
	return v1

def accuracy_score(y_test, p_test):
	v1 = 1.0 * ((p_test==1) & (y_test==1)).sum(axis=1)
	v2 = 1.0 * ((p_test==1) | (y_test==1)).sum(axis=1) 
	v1[v2<=0] = 1.0
	v1[v2>0] /= v2[v2>0]
	return v1
