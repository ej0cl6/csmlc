import numpy as np
from criteria import hamming_loss, rank_loss, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from models.br import BR
from models.cc import CC
from models.pcc import PCC
from models.cft import CFT
from models.clems import CLEMS

# set random seed
np.random.seed(1)

# load data
x_data = np.loadtxt('scene/data.x', dtype=float)
y_data = np.loadtxt('scene/data.y', dtype=int)

# split data for training and testing
idxs = np.arange(x_data.shape[0])
np.random.shuffle(idxs)
x_train = x_data[:x_data.shape[0]/2]
y_train = y_data[:y_data.shape[0]/2]
x_test = x_data[x_data.shape[0]/2:]
y_test = y_data[x_data.shape[0]/2:]

# algorithms (there is no inference rule for pcc_acc, using pcc_f1 instead)
params = {"n_estimators":100, "max_depth": 10, "max_features": "sqrt", "n_jobs": 10}
alg_br = BR(RandomForestClassifier, params)
alg_cc = CC(RandomForestClassifier, params)
alg_pcc_ham = PCC('ham', RandomForestClassifier, params)
alg_pcc_rank = PCC('rank', RandomForestClassifier, params)
alg_pcc_f1 = PCC('f1', RandomForestClassifier, params)
alg_cft_ham = CFT('ham', RandomForestClassifier, params)
alg_cft_rank = CFT('rank', RandomForestClassifier, params)
alg_cft_f1 = CFT('f1', RandomForestClassifier, params)
alg_cft_acc = CFT('acc', RandomForestClassifier, params)
alg_clems_ham = CLEMS('ham', RandomForestRegressor, params)
alg_clems_rank = CLEMS('rank', RandomForestRegressor, params)
alg_clems_f1 = CLEMS('f1', RandomForestRegressor, params)
alg_clems_acc = CLEMS('acc', RandomForestRegressor, params)

print 'training BR ...'
alg_br.fit(x_train, y_train)
p_br = alg_br.predict(x_test)

print 'training CC ...'
alg_cc.fit(x_train, y_train)
p_cc = alg_cc.predict(x_test)

print 'training PCC_ham ...'
alg_pcc_ham.fit(x_train, y_train)
p_pcc_ham = alg_pcc_ham.predict(x_test)

print 'training PCC_rank ...'
alg_pcc_rank.fit(x_train, y_train)
p_pcc_rank = alg_pcc_rank.predict(x_test)

print 'training PCC_f1 ...'
alg_pcc_f1.fit(x_train, y_train)
p_pcc_f1 = alg_pcc_f1.predict(x_test)

print 'training CFT_ham ...'
alg_cft_ham.fit(x_train, y_train)
p_cft_ham = alg_cft_ham.predict(x_test)

print 'training CFT_rank ...'
alg_cft_rank.fit(x_train, y_train)
p_cft_rank = alg_cft_rank.predict(x_test)

print 'training CFT_f1 ...'
alg_cft_f1.fit(x_train, y_train)
p_cft_f1 = alg_cft_f1.predict(x_test)

print 'training CFT_acc ...'
alg_cft_acc.fit(x_train, y_train)
p_cft_acc = alg_cft_acc.predict(x_test)

print 'training CLEMS_ham ...'
alg_clems_ham.fit(x_train, y_train)
p_clems_ham = alg_clems_ham.predict(x_test)

print 'training CLEMS_rank ...'
alg_clems_rank.fit(x_train, y_train)
p_clems_rank = alg_clems_rank.predict(x_test)

print 'training CLEMS_f1 ...'
alg_clems_f1.fit(x_train, y_train)
p_clems_f1 = alg_clems_f1.predict(x_test)

print 'training CLEMS_acc ...'
alg_clems_acc.fit(x_train, y_train)
p_clems_acc = alg_clems_acc.predict(x_test)

ham_br, rank_br, f1_br, acc_br = hamming_loss(y_test, p_br).mean(), rank_loss(y_test, p_br).mean(), f1_score(y_test, p_br).mean(), accuracy_score(y_test, p_br).mean()
ham_cc, rank_cc, f1_cc, acc_cc = hamming_loss(y_test, p_cc).mean(), rank_loss(y_test, p_cc).mean(), f1_score(y_test, p_cc).mean(), accuracy_score(y_test, p_cc).mean()
ham_pcc, rank_pcc, f1_pcc, acc_pcc = hamming_loss(y_test, p_pcc_ham).mean(), rank_loss(y_test, p_pcc_rank).mean(), f1_score(y_test, p_pcc_f1).mean(), accuracy_score(y_test, p_pcc_f1).mean()
ham_cft, rank_cft, f1_cft, acc_cft = hamming_loss(y_test, p_cft_ham).mean(), rank_loss(y_test, p_cft_rank).mean(), f1_score(y_test, p_cft_f1).mean(), accuracy_score(y_test, p_cft_acc).mean()
ham_clems, rank_clems, f1_clems, acc_clems = hamming_loss(y_test, p_clems_ham).mean(), rank_loss(y_test, p_clems_rank).mean(), f1_score(y_test, p_clems_f1).mean(), accuracy_score(y_test, p_clems_acc).mean()

show_title = 'algorithm  hamming_loss  rank_loss  f1_score  accuracy_score'
show_bar   = '============================================================'
show_br    = '       BR        {:.4f}     {:.4f}    {:.4f}          {:.4f}'.format(ham_br, rank_br, f1_br, acc_br)
show_cc    = '       CC        {:.4f}     {:.4f}    {:.4f}          {:.4f}'.format(ham_cc, rank_cc, f1_cc, acc_cc)
show_pcc   = '      PCC        {:.4f}     {:.4f}    {:.4f}          {:.4f}'.format(ham_pcc, rank_pcc, f1_pcc, acc_pcc)
show_cft   = '      CFT        {:.4f}     {:.4f}    {:.4f}          {:.4f}'.format(ham_cft, rank_cft, f1_cft, acc_cft)
show_clems = '    CLEMS        {:.4f}     {:.4f}    {:.4f}          {:.4f}'.format(ham_clems, rank_clems, f1_clems, acc_clems)

print ''
print show_bar
print show_title
print show_bar
print show_br
print show_cc
print show_pcc
print show_cft
print show_clems
print show_bar
