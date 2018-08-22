import json
import numpy as np
import pandas as pd
import re
import random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
# import matplotlib.pyplot as plt
# import MySQLdb
import gc

df = pd.read_csv("movie.csv")

actor = pd.read_csv("actor_new.csv")
actress = pd.read_csv("actress_new.csv")
mcast = pd.read_csv("male_cast.csv")
fcast = pd.read_csv("female_cast.csv")

mc = mcast.copy()
mc = mc.set_index('movie_id')

fc = fcast.copy()
fc = fc.set_index('movie_id')

names = pd.read_csv("name_all.csv")
names = names.set_index('Id')

df = df.set_index('Id')

df['Cast'] = mc['group_concat(person_id)'] + "," + fc['group_concat(person_id)']

col = df.columns.tolist()
# col = col[:2]+col[4:13]+col[14:17]+col[-1:]+col[2:4]+col[13:14]
col = col[:1]+col[3:12]+col[13:15]+col[-1:]+col[1:3]+col[12:13]+col[15:17]
df = df[col]


#break countries string into array
df[col[1:13]] = df[col[1:13]].replace(np.nan, '', regex=True)
df[col[1:13]] = df[col[1:13]].fillna(value='')
for i in range(1, 13):
	df[col[i]] = df[col[i]].apply(lambda x: x.split(",") if len(x)>0 else [])
	df[col[i]] = df[col[i]].apply(lambda x: [col[i][:3]+e.strip() if len(e.strip())>0 else e.strip() for e in x] if len(x)>0 else [])

counts = actor.person_id.value_counts(sort=True).copy()
fcounts = actress.person_id.value_counts(sort=True).copy()
# db= MySQLdb.connect(host="localhost", user="root", passwd="rootuser", db="imdb")
# cur = db.cursor()


def actorsCount(fname, lname):
	counts = actor.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts > 30 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()

	fcounts = actress.person_id.value_counts(sort=True).copy()
	fnewCount = fcounts.where( fcounts > 4 )
	fnewCount = fnewCount.dropna()	
	p_ids = p_ids + fnewCount.index.tolist()
	counts = pd.concat([newCount, fnewCount])
	cur.execute("Select id, name from name where lower(name) like '%"+fname+"%' and lower(name) like '%"+lname+"%'")
	arr = {}
	for row in cur:
		arr[int(row[0])] =  row[1]
	all_ids = arr.keys()
	intersection = np.intersect1d(all_ids, p_ids)
	user_ids = []
	for i in intersection:
		user_ids.append([i, arr[i].decode('utf8').encode('ascii', errors='ignore'), counts[i]])
	return user_ids

def findActor(fname, lname):
	found = names[names['Name'].str.contains(".*"+lname+".*"+fname+".*", regex=True, case=False)]
	id_found = found.index
	users = []
	c_ind = counts.index
	f_ind = fcounts.index
	for i in id_found:
		if i in c_ind:
			users.append([i, names.loc[i]['Name'].decode('utf8').encode('ascii', errors='ignore'), counts[i]])
		else:
			users.append([i, names.loc[i]['Name'].decode('utf8').encode('ascii', errors='ignore'), fcounts[i]])
			
	return users


def logreg(actor_id, lg=0):
	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress

	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count
	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx+len_inds/2]
	else:
		p_ids = idxs[idx:idx+len_inds]

	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	X = movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())

		
	X_train, X_test, y_train, y_test = train_test_split(X, 1*y, test_size=0.33)
	X_train_tfidf = tfidf.fit_transform(X_train)
	X_test_tfidf = tfidf.transform(X_test)
	# logreg = linear_model.LogisticRegression()
	# logreg = LinearSVC(random_state=0, C=100000, fit_intercept=False, tol=0)
	logreg = linear_model.LogisticRegression(C=200, penalty='l1', fit_intercept=False)
	# logreg = RandomForestClassifier(n_estimators=100, random_state=0)
	# logreg = SVC(C=3200)
	logreg.fit(X_train_tfidf, y_train)
	y_pred = logreg.predict(X_test_tfidf)
	# sum(y_pred)*1.0/sum(y_test)

	yt = np.where(y_test ==1)
	yn = np.where(y_pred ==1)
	yf = np.intersect1d(yt[0], yn[0])
	#movies
	proba = logreg.predict_proba(X_test_tfidf)
	top10_idx = np.argsort(proba[:, 1])[-10:][::-1]
	if lg == 1:
		top10_idx = np.argsort(proba[:, 1])[-20:][::-1]

	indc = np.array(y_test.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values

	y_score = logreg.decision_function(X_test_tfidf)
	fpr, tpr, _ = roc_curve(y_test, y_score)
	roc_auc = auc(fpr, tpr)
	# plt.figure()
	# lw = 2
	# plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Receiver operating characteristic example')
	# plt.legend(loc="lower right")
	# plt.savefig(str(actor_id)+".png")

	return [[len(y_train), len(y_test)], [i.decode('utf8').encode('ascii', errors='ignore') for i in df.loc[inxs].Title.tolist()], proba[top10_idx, 1], y_pred[top10_idx], y_test.values[top10_idx], [sum(y_train), sum(y_test), sum(y_pred), len(yf)]]


def svmc(actor_id, lg=0):
	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress

	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count
	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx+len_inds/2]
	else:
		p_ids = idxs[idx:idx+len_inds]

	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	# movies['features'] = df[col[3:14]+['Cast']].values.tolist()
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	X = movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())

		
	X_train, X_test, y_train, y_test = train_test_split(X, 1*y, test_size=0.33)
	X_train_tfidf = tfidf.fit_transform(X_train)
	X_test_tfidf = tfidf.transform(X_test)

	logreg = SVC(C=3200, probability=True)
	
	logreg.fit(X_train_tfidf, y_train)
	y_pred = logreg.predict(X_test_tfidf)
	

	yt = np.where(y_test ==1)
	yn = np.where(y_pred ==1)
	print yn
	yf = np.intersect1d(yt[0], yn[0])
	#movies
	proba = logreg.predict_proba(X_test_tfidf)
	top10_idx = np.argsort(proba[:, 1])[-10:][::-1]
	if lg == 1:
		top10_idx = np.argsort(proba[:, 1])[-20:][::-1]

	indc = np.array(y_test.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values


	return [[len(y_train), len(y_test)], [i.decode('utf8').encode('ascii', errors='ignore') for i in df.loc[inxs].Title.tolist()], proba[top10_idx, 1], y_pred[top10_idx], y_test.values[top10_idx], [sum(y_train), sum(y_test), sum(y_pred), len(yf)]]


def jaccard_sim(actor_id):
	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress


	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count

	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx] + idxs[idx+1:idx+len_inds/2]
	else:
		p_ids = idxs[idx+1:idx+len_inds]

	col = df.columns.tolist()
	#actor movies

	orig_movie_ids = dataframe.loc[dataframe['person_id'] == actor_id]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	rndom = np.random.choice(orig_movie_ids, int(math.ceil(0.7*len(orig_movie_ids))), replace=False)
	to_add = np.setdiff1d(orig_movie_ids, rndom)

	movies_co = df.loc[df.index.isin(rndom)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()

	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])

	x_0 = movies['features'].tolist()
	x_0 = [i for obj in x_0 for i in obj]

	#job pool
	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy().tolist()
	orig_movie_ids = np.unique(orig_movie_ids).tolist() + to_add.tolist()

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	# tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	# X = [x_0] + movies['features'].tolist()
	# y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())
	# y = y*1

	X = movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())
	y = y*1
	sim = np.zeros(len(X))
	j=0
	for i in X:
		union = np.union1d(x_0, i)
		inter = np.intersect1d(x_0, i)
		if len(union) > 0:
			sim[j] = (len(inter)*1.0)/len(union)
		j = j+1

		# X_tfidf = tfidf.fit_transform(X)

		# feat_names = tfidf.get_feature_names()
		# similarity = np.zeros((len(X)))
		# for i in range(count):
		# 	similarity += cosine_similarity(X_tfidf[0+i:i+1], X_tfidf)[0]

	z = np.where(y == 1)

	v = np.where(sim >0.0007)

	yf = len(np.intersect1d(z[0], v))
	
	y_pred = np.zeros(len(y))
	y_pred[v]  = 1
	top10_idx = np.argsort(sim)[-11:][9::-1]
	top10_idx = [i for i in top10_idx]

	indc = np.array(y.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values


	return [[len(rndom), len(to_add)], [i.decode('utf8').encode('ascii', errors='ignore') for i in df.loc[inxs].Title.tolist()], sim[top10_idx].tolist(), y_pred[top10_idx], y.values[top10_idx], [len(rndom), sum(y), sum(y_pred), yf]]


def cosine_sim(actor_id, lg=0):
	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress

	
	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count

	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx] + idxs[idx+1:idx+len_inds/2]
	else:
		p_ids = idxs[idx+1:idx+len_inds]

	col = df.columns.tolist()
	#actor movies

	orig_movie_ids = dataframe.loc[dataframe['person_id'] == actor_id]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	rndom = np.random.choice(orig_movie_ids, int(math.ceil(0.7*len(orig_movie_ids))), replace=False)
	to_add = np.setdiff1d(orig_movie_ids, rndom)

	movies_co = df.loc[df.index.isin(rndom)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])

	x_0 = movies['features'].tolist()
	x_0 = [i for obj in x_0 for i in obj]

	#job pool
	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy().tolist()
	orig_movie_ids = np.unique(orig_movie_ids).tolist() + to_add.tolist()

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	X = [x_0] + movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())
	y = y*1

	X_tfidf = tfidf.fit_transform(X)

	# feat_names = tfidf.get_feature_names()
	# similarity = np.zeros((len(X)))
	# for i in range(count):
	# 	similarity += cosine_similarity(X_tfidf[0+i:i+1], X_tfidf)[0]

	z = np.where(y == 1)
	sim = cosine_similarity(X_tfidf[0:1], X_tfidf)[0]
	v = np.where(sim >0.05)
	v = np.setdiff1d(v, [0])
	v = [i-1 for i in v]

	yf = len(np.intersect1d(z[0], v))

	y_pred = np.zeros(len(y))
	y_pred[v]  = 1

	top10_idx = np.argsort(sim)[-10:][::-1]
	if lg == 1:
		top10_idx = np.argsort(sim)[-20:][9::-1]

	top10_idx = [i for i in top10_idx]

	indc = np.array(y.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values

	

	return [[len(rndom), len(to_add)], [i.decode('utf8').encode('ascii', errors='ignore') for i in df.loc[inxs].Title.tolist()], sim[top10_idx].tolist(), y_pred[top10_idx], y.values[top10_idx], [len(rndom), sum(y), sum(y_pred), yf]]
	

def pearson_sim(actor_id):
	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress


	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count

	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx] + idxs[idx+1:idx+len_inds/2]
	else:
		p_ids = idxs[idx+1:idx+len_inds]

	col = df.columns.tolist()
	#actor movies

	orig_movie_ids = dataframe.loc[dataframe['person_id'] == actor_id]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	rndom = np.random.choice(orig_movie_ids, int(math.ceil(0.7*len(orig_movie_ids))), replace=False)
	to_add = np.setdiff1d(orig_movie_ids, rndom)

	movies_co = df.loc[df.index.isin(rndom)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()

	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])

	x_0 = movies['features'].tolist()
	x_0 = [i for obj in x_0 for i in obj]

	#job pool
	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy().tolist()
	orig_movie_ids = np.unique(orig_movie_ids).tolist() + to_add.tolist()

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	X = [x_0] + movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())
	y = y*1

	X_tfidf = tfidf.fit_transform(X)

		# feat_names = tfidf.get_feature_names()
		# similarity = np.zeros((len(X)))
		# for i in range(count):
		# 	similarity += cosine_similarity(X_tfidf[0+i:i+1], X_tfidf)[0]

	z = np.where(y == 1)
	sim = np.corrcoef(X_tfidf.A[0:1], X_tfidf.A)[0]
	v = np.where(sim >0.0)
	v = np.setdiff1d(v, [0])
	v = [i-1 for i in v]

	yf = len(np.intersect1d(z[0], v))

	y_pred = np.zeros(len(y))
	y_pred[v]  = 1
	top10_idx = np.argsort(sim)[-11:][9::-1]
	top10_idx = [i-1 for i in top10_idx]

	indc = np.array(y.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values

	return [[len(rndom), len(to_add)], [i.decode('utf8').encode('ascii', errors='ignore') for i in df.loc[inxs].Title.tolist()], sim[top10_idx].tolist(), y_pred[top10_idx], y.values[top10_idx], [sum(rndom), sum(y), sum(y_pred), yf]]
	

def logreg_filer(actor_id):

	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress

	
	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count

	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx+len_inds/2]
	else:
		p_ids = idxs[idx:idx+len_inds]


	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: [i if i != col[12][3]+str(actor_id) else '' for i in x ] if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	# movies['features'] = df[col[3:14]+['Cast']].values.tolist()
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	X = movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())


	X_train, X_test, y_train, y_test = train_test_split(X, 1*y, test_size=0.33)
	X_train_tfidf = tfidf.fit_transform(X_train)
	X_test_tfidf = tfidf.transform(X_test)
	# logreg = linear_model.LogisticRegression()
	# logreg = linear_model.LogisticRegression(C=200, penalty='l1', fit_intercept=False)
	# logreg = RandomForestClassifier(n_estimators=100, random_state=0)
	logreg = SVC(C=3200, probability=True)
	logreg.fit(X_train_tfidf, y_train)
	y_pred = logreg.predict(X_test_tfidf)
	# sum(y_pred)*1.0/sum(y_test)

	yt = np.where(y_test ==1)
	yn = np.where(y_pred ==1)
	yf = np.intersect1d(yt[0], yn[0])
	yf_arr = 0
	if (sum(y_test) >0 ):
		yf_arr = len(yf)*1.0/sum(y_test)


	scores = cross_val_score(logreg, tfidf.fit_transform(X), y*1, cv=5)

	if sum(y_test) > 0:
		y_score = logreg.decision_function(X_test_tfidf)
		fpr, tpr,_ = roc_curve(y_test, y_score)
		roc_auc = roc_auc_score(y_test, y_score)
		avg_prc = average_precision_score(y_test, y_score)
		prec, recall, _ = precision_recall_curve(y_test, y_score)
		y_score = y_score.tolist()
		prec = prec.tolist()
		recall = recall.tolist()
	else:
		y_score = []
		fpr, tpr = 0,0
		roc_auc = 0
		avg_prc = 0
		prec, recall = [],[]

	#movies
	proba = logreg.predict_proba(X_test_tfidf)

	top10_idx = np.argsort(proba[:, 1])[-10:][::-1]
	indc = np.array(y_test.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values

	return [counts[actor_id], inxs.tolist(), yf_arr, np.mean(scores), roc_auc, avg_prc, y_test.tolist(), y_pred.tolist(), y_score, prec, recall]

	


def logreg_lsa_filer(actor_id):
	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress

	
	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count
	
	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx+len_inds/2]
	else:
		p_ids = idxs[idx:idx+len_inds]

	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	X = movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())

	svd = TruncatedSVD(200)
	lsa = make_pipeline(svd, Normalizer(copy=False))


	X_train, X_test, y_train, y_test = train_test_split(X, 1*y, test_size=0.33)
	X_train_tfidf = tfidf.fit_transform(X_train)
	X_test_tfidf = tfidf.transform(X_test)

	feat_names = tfidf.get_feature_names()

	X_train_lsa = lsa.fit_transform(X_train_tfidf)
	explained_variance = svd.explained_variance_ratio_.sum()

	X_test_lsa = lsa.transform(X_test_tfidf)

	# logreg = linear_model.LogisticRegression()
	# logreg = linear_model.LogisticRegression(C=200, penalty='l1', fit_intercept=False)
	# logreg = RandomForestClassifier(n_estimators=100, random_state=0)
	# logreg = RandomForestClassifier(n_estimators=100, random_state=0)
	logreg = SVC(C=3200, probability=True)
	logreg.fit(X_train_lsa, y_train)
	y_pred = logreg.predict(X_test_lsa)
	# sum(y_pred)*1.0/sum(y_test)

	yt = np.where(y_test ==1)
	yn = np.where(y_pred ==1)
	yf = np.intersect1d(yt[0], yn[0])
	yf_arr = 0
	if (sum(y_test) >0 ):
		yf_arr = len(yf)*1.0/sum(y_test)


	scores = cross_val_score(logreg, tfidf.fit_transform(X), y*1, cv=5)

	if sum(y_test) > 0:
		y_score = logreg.decision_function(X_test_lsa)
		fpr, tpr,_ = roc_curve(y_test, y_score)
		roc_auc = roc_auc_score(y_test, y_score)
		avg_prc = average_precision_score(y_test, y_score)
		prec, recall, _ = precision_recall_curve(y_test, y_score)
		y_score = y_score.tolist()
		prec = prec.tolist()
		recall = recall.tolist()
	else:
		y_score = []
		fpr, tpr = 0,0
		roc_auc = 0
		avg_prc = 0
		prec, recall = [],[]

	#movies
	proba = logreg.predict_proba(X_test_lsa)

	top10_idx = np.argsort(proba[:, 1])[-10:][::-1]
	indc = np.array(y_test.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values


	return [counts[actor_id], inxs.tolist(), yf_arr, np.mean(scores), roc_auc, avg_prc, y_test.tolist(), y_pred.tolist(), y_score, prec, recall]




def cosine_sim_filer(actor_id):
	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress

	
	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count

	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx] + idxs[idx+1:idx+len_inds/2]
	else:
		p_ids = idxs[idx+1:idx+len_inds]

	col = df.columns.tolist()
	#actor movies

	orig_movie_ids = dataframe.loc[dataframe['person_id'] == actor_id]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	rndom = np.random.choice(orig_movie_ids, int(math.ceil(0.7*len(orig_movie_ids))), replace=False)
	to_add = np.setdiff1d(orig_movie_ids, rndom)

	movies_co = df.loc[df.index.isin(rndom)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])

	x_0 = movies['features'].tolist()
	x_0 = [i for obj in x_0 for i in obj]

	#job pool
	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy().tolist()
	orig_movie_ids = np.unique(orig_movie_ids).tolist() + to_add.tolist()

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	X = [x_0] + movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())
	y = y*1

	X_tfidf = tfidf.fit_transform(X)

	# feat_names = tfidf.get_feature_names()
	# similarity = np.zeros((len(X)))
	# for i in range(count):
	# 	similarity += cosine_similarity(X_tfidf[0+i:i+1], X_tfidf)[0]

	z = np.where(y == 1)
	sim = cosine_similarity(X_tfidf[0:1], X_tfidf)[0]
	v = np.where(sim >0.05)
	v = np.setdiff1d(v, [0])
	v = [i-1 for i in v]

	yf = len(np.intersect1d(z[0], v))

	y_pred = np.zeros(len(y))
	y_pred[v]  = 1
	top10_idx = np.argsort(sim)[-11:][9::-1]
	top10_idx = [i-1 for i in top10_idx]

	indc = np.array(y.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values

	if sum(y) > 0:
		y_score = sim[1:]
		roc_auc = roc_auc_score(y, y_score)
		avg_prc = average_precision_score(y, y_score)
		prec, recall, _, _ = precision_recall_fscore_support(y, y_pred)
		prec = prec.tolist()
		recall = recall.tolist()
		yf_arr = (1.0*yf)/sum(y)
		y_score = y_score.tolist()
	else:
		y_score = []
		fpr, tpr = 0,0
		roc_auc = 0
		avg_prc = 0
		prec, recall = [],[]
		yf_arr = 0

	return [counts[actor_id], inxs.tolist(), sim[top10_idx].tolist(), yf_arr, roc_auc, avg_prc, y.values.tolist(), y_pred.tolist(), y_score, prec, recall]


def pearson_sim_filer(actor_id):
	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress

	
	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count

	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx] + idxs[idx+1:idx+len_inds/2]
	else:
		p_ids = idxs[idx+1:idx+len_inds]

	col = df.columns.tolist()
	#actor movies

	orig_movie_ids = dataframe.loc[dataframe['person_id'] == actor_id]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	rndom = np.random.choice(orig_movie_ids, int(math.ceil(0.7*len(orig_movie_ids))), replace=False)
	to_add = np.setdiff1d(orig_movie_ids, rndom)

	movies_co = df.loc[df.index.isin(rndom)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])

	x_0 = movies['features'].tolist()
	x_0 = [i for obj in x_0 for i in obj]

	#job pool
	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy().tolist()
	orig_movie_ids = np.unique(orig_movie_ids).tolist() + to_add.tolist()

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	X = [x_0] + movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())
	y = y*1

	X_tfidf = tfidf.fit_transform(X)

	# feat_names = tfidf.get_feature_names()
	# similarity = np.zeros((len(X)))
	# for i in range(count):
	# 	similarity += cosine_similarity(X_tfidf[0+i:i+1], X_tfidf)[0]

	z = np.where(y == 1)
	sim = np.corrcoef(X_tfidf.A[0:1], X_tfidf.A)[0][1:]
	sim = np.nan_to_num(sim)
	v = np.where(sim >0)
	v = np.setdiff1d(v, [0])
	v = [i-1 for i in v]

	yf = len(np.intersect1d(z[0], v))

	y_pred = np.zeros(len(y))
	y_pred[v]  = 1
	top10_idx = np.argsort(sim)[-11:][9::-1]
	top10_idx = [i-1 for i in top10_idx]

	indc = np.array(y.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values

	if sum(y) > 0:
		y_score = sim[1:]
		roc_auc = roc_auc_score(y, y_score)
		avg_prc = average_precision_score(y, y_score)
		prec, recall, _, _ = precision_recall_fscore_support(y, y_pred)
		prec = prec.tolist()
		recall = recall.tolist()
		yf_arr = (1.0*yf)/sum(y)
		y_score = y_score.tolist()
	else:
		y_score = []
		fpr, tpr = 0,0
		roc_auc = 0
		avg_prc = 0
		prec, recall = [],[]
		yf_arr = 0

	return [counts[actor_id], inxs.tolist(), sim[top10_idx].tolist(), yf_arr, roc_auc, avg_prc, y.values.tolist(), y_pred.tolist(), y_score, prec, recall]



def jaccard_sim_filer(actor_id):
	counts = actor.person_id.value_counts(sort=True).copy()
	dataframe = actor
	a = np.where(counts.index == actor_id)
	if len(a[0]) == 0:
		counts = actress.person_id.value_counts(sort=True).copy()
		dataframe = actress


	idx = counts.index.get_loc(actor_id)
	idxs = counts.index.tolist()
	count = counts[actor_id]
	len_inds  = 1000/count

	if idx > len_inds:
		p_ids = idxs[idx-len_inds/2:idx] + idxs[idx+1:idx+len_inds/2]
	else:
		p_ids = idxs[idx+1:idx+len_inds]

	col = df.columns.tolist()
	#actor movies

	orig_movie_ids = dataframe.loc[dataframe['person_id'] == actor_id]['movie_id'].values.copy()
	orig_movie_ids = np.unique(orig_movie_ids)

	rndom = np.random.choice(orig_movie_ids, int(math.ceil(0.7*len(orig_movie_ids))), replace=False)
	to_add = np.setdiff1d(orig_movie_ids, rndom)

	movies_co = df.loc[df.index.isin(rndom)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()

	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])

	x_0 = movies['features'].tolist()
	x_0 = [i for obj in x_0 for i in obj]

	#job pool
	col = df.columns

	orig_movie_ids = dataframe.loc[dataframe['person_id'].isin(p_ids)]['movie_id'].values.copy().tolist()
	orig_movie_ids = np.unique(orig_movie_ids).tolist() + to_add.tolist()

	movies_co = df.loc[df.index.isin(orig_movie_ids)].copy()
	# movies_co[col[12]] = movies_co[col[12]].apply(lambda x: x.remove(col[12][3]+str(actor_id)) if len(x)>0 and col[12][3]+str(actor_id) in x else x )

	movies_co[col[12]] = movies_co[col[12]].apply(lambda x: filter(lambda a: a != col[12][:3]+str(actor_id), x))

	movies = pd.DataFrame()
	col = df.columns.tolist()
	movies['Id'] = movies_co.index
	# df['Cast'] = cast.Cast
	movies['features'] = movies_co[col[2:13]].values.tolist()
	movies['features'] = movies['features'].apply(lambda x: [i for obj in x for i in obj])
	# tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

	# X = [x_0] + movies['features'].tolist()
	# y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())
	# y = y*1

	X = movies['features'].tolist()
	y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())
	y = y*1
	sim = np.zeros(len(X))
	j=0
	for i in X:
		union = np.union1d(x_0, i)
		inter = np.intersect1d(x_0, i)
		if len(union) > 0:
			sim[j] = (len(inter)*1.0)/len(union)
		j = j+1

		# X_tfidf = tfidf.fit_transform(X)

		# feat_names = tfidf.get_feature_names()
		# similarity = np.zeros((len(X)))
		# for i in range(count):
		# 	similarity += cosine_similarity(X_tfidf[0+i:i+1], X_tfidf)[0]

	z = np.where(y == 1)

	v = np.where(sim >0.0007)

	yf = len(np.intersect1d(z[0], v))

	y_pred = np.zeros(len(y))
	y_pred[v]  = 1
	top10_idx = np.argsort(sim)[-11:][9::-1]
	top10_idx = [i for i in top10_idx]

	indc = np.array(y.index.tolist())[top10_idx]
	inxs = movies.iloc[indc].Id.values

	if sum(y) > 0:
		y_score = sim
		roc_auc = roc_auc_score(y, y_score)
		avg_prc = average_precision_score(y, y_score)
		prec, recall, _, _ = precision_recall_fscore_support(y, y_pred)
		prec = prec.tolist()
		recall = recall.tolist()
		yf_arr = (1.0*yf)/sum(y)
		y_score = y_score.tolist()
	else:
		y_score = []
		fpr, tpr = 0,0
		roc_auc = 0
		avg_prc = 0
		prec, recall = [],[]
		yf_arr = 0

	return [counts[actor_id], inxs.tolist(), sim[top10_idx].tolist(), yf_arr, roc_auc, avg_prc, y.values.tolist(), y_pred.tolist(), y_score, prec, recall]



# X = movies['features'].tolist()
# y = movies['Id'].isin(dataframe.loc[dataframe['person_id']==actor_id]['movie_id'].values.tolist())

# X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, 1*y, test_size=0.33)
# X_train = tfidf.fit_transform(X_train_tfidf)

# X_test= tfidf.transform(X_test_tfidf) 


	
	# indc = np.array(y_test.index.tolist())[yf]
	# return [df.loc[indc].Title.tolist(), proba[yf, 1]]


	# accuracy = accuracy_score(y_test, y_pred)
	# prec = precision_score(y_test, y_pred, average='macro')
	# proba = logreg.predict_log_proba(X_test)
	# top10_idx = np.argsort(proba[:, 1])[-10:]
	# top10_val = [proba[i, 1] for i in top10_idx]
	# predicts = [y_pred[i] for i in top10_idx]
	# tests = [y_test.tolist()[i] for i in top10_idx]
	# 	print predicts
	# 	print tests
	# 	return [accuracy, prec]


def results_big_cosine():
	counts = actor.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts > 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	c=0
	res = dict()
	j=0
	for i in p_ids:
		if c != newCount[i]:
			j=0
		if j<10:
			print i
			res[i] = cosine_sim_filer(i)
			gc.collect()
			c = newCount[i]
			j = j+1
		

	with open('cosine_Big-9Aug-Male.txt', 'w') as file:
		file.write(json.dumps(res))

	counts = actress.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts > 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	c=0
	res = dict()
	j=0
	for i in p_ids:
		if c != newCount[i]:
			j=0
		if j<10:
			print i
			res[i] = cosine_sim_filer(i)
			gc.collect()
			c = newCount[i]
			j = j+1

	with open('cosine_Big-9Aug-Female.txt', 'w') as file:
		file.write(json.dumps(res))


def results_big_jaccard():
	counts = actor.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts > 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	c=0
	res = dict()
	
	j=0
	for i in p_ids:
		if c != newCount[i]:
			j=0
		if j<10:
			print i
			res[i] = pearson_sim_filer(i)
			gc.collect()
			c = newCount[i]
			j = j+1

	with open('pearson_Big-10_Aug_Male.txt', 'w') as file:
		file.write(json.dumps(res))

	counts = actress.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts > 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	c=0
	res = dict()
	j=0
	for i in p_ids:
		if c != newCount[i]:
			j=0
		if j<10:
			print i
			res[i] = pearson_sim_filer(i)
			gc.collect()
			c = newCount[i]
			j = j+1
		

	with open('pearson_Big-10Aug_Female.txt', 'w') as file:
		file.write(json.dumps(res))


def results_jaccard():
	counts = actor.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts == 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	res = dict()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 4 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 3 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 2 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 1 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	with open('jaccard-24-Jul_Male.txt', 'w') as file:
		file.write(json.dumps(res))

	counts = actress.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts == 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	res = dict()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 4 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 3 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 2 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 1 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = jaccard_sim_filer(p_ids[i])
		gc.collect()

	with open('jaccard-24-Jul_Female.txt', 'w') as file:
		file.write(json.dumps(res))


def results_cosine():
	counts = actor.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts == 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	res = dict()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 4 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 3 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 2 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 1 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	with open('cosine-22-Jul_Male.txt', 'w') as file:
		file.write(json.dumps(res))

	counts = actress.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts == 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	res = dict()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 4 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 3 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 2 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	newCount = counts.where( counts == 1 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	
	for i in range(100):
		res[p_ids[i]] = cosine_sim(p_ids[i])
		gc.collect()

	with open('cosine-22-Jul_Female.txt', 'w') as file:
		file.write(json.dumps(res))


def lsa_res():
	
	counts = actor.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts > 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	c=0
	res = dict()
	j=0
	for i in p_ids:
		if c != newCount[i]:
			j = 0
		if j<10:
			print i
			res[i] = logreg_lsa_filer(i)
			gc.collect()
			c = newCount[i]
			j = j+1



	with open('resl/SVM-Lsa200-11-Aug_Male.txt', 'w') as file:
		file.write(json.dumps(res))


	counts = actress.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts > 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	c=0
	res = dict()
	j=0
	for i in p_ids:
		if c != newCount[i]:
			j = 0
		if j<10:
			print i
			res[i] = logreg_lsa_filer(i)
			gc.collect()
			c = newCount[i]
			j = j+1



	with open('resl/SVM-Lsa200-11-Aug_Female.txt', 'w') as file:
		file.write(json.dumps(res))

def results():
	counts = actor.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts > 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	c=0
	res = dict()
	j=0
	for i in p_ids:
		if c != newCount[i]:
			j = 0
		if j<10:
			print i
			res[i] = logreg_filer(i)
			gc.collect()
			c = newCount[i]
			j = j+1


	with open('svm(cast)-7Aug_Male.txt', 'w') as file:
		file.write(json.dumps(res))
	
	counts = actress.person_id.value_counts(sort=True).copy()
	newCount = counts.where( counts > 5 )
	newCount = newCount.dropna()	
	p_ids = newCount.index.tolist()
	c=0
	res = dict()
	for i in p_ids:
		if c != newCount[i]:
			j = 0
		if j<10:
			print i
			res[i] = logreg_filer(i)
			gc.collect()
			c = newCount[i]
			j = j+1


	with open('svm(cast)-7Aug_Female.txt', 'w') as file:
		file.write(json.dumps(res))


	# counts = actor.person_id.value_counts(sort=True).copy()
	# newCount = counts.where( counts > 5 )
	# newCount = newCount.dropna()	
	# p_ids = newCount.index.tolist()
	# c=0
	# res = dict()
	# for i in p_ids:
	# 	if c != newCount[i]:
	# 		print i
	# 		res[i] = logreg_lsa_filer(i)
	# 		gc.collect()
	# 		c = newCount[i]


	# with open('svm(3200)(cast)_LSA(50)-19-Jul_Male.txt', 'w') as file:
	# 	file.write(json.dumps(res))

	# counts = actress.person_id.value_counts(sort=True).copy()
	# newCount = counts.where( counts > 5 )
	# newCount = newCount.dropna()	
	# p_ids = newCount.index.tolist()
	# c=0
	# res = dict()
	# for i in p_ids:
	# 	if c != newCount[i]:
	# 		print i
	# 		res[i] = logreg_lsa_filer(i)
	# 		gc.collect()
	# 		c = newCount[i]

	
	# with open('svm(3200)(cast)_LSA(50)-19-Jul_Female.txt', 'w') as file:
	# 	file.write(json.dumps(res))



# def actorsCount(con):
# 	counts = actor.person_id.value_counts(sort=True).copy()
# 	newCount = counts.where( counts > con )
# 	newCount = newCount.dropna()
# 	counts = actress.person_id.value_counts(sort=True).copy()
# 	fnewCount = counts.where( counts > con )	
# 	fnewCount = fnewCount.dropna()
# 	return len(newCount), len(fnewCount)

