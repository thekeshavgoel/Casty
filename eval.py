import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import *



files_male = [ "svm(cast)-7Aug_Male.txt", "logreg(cast)-7Aug_Male.txt", "SVM-Lsa200-11-Aug_Male.txt", "Logreg-Lsa200-11-Aug_Male.txt", "pearson_Big-10_Aug_Male.txt", "jaccard_Big-10_Aug_Male.txt", "cosine_Big-9Aug-Male.txt"]
names_male = [ "SVM - Actors", "Logistic Reg - Actors", "SVM with LSA - Actors", "Logistic Reg with LSA - Actors", "PCC - Actors", "Jaccard Similarity - Actors", "Cosine Similarity - Actors" ]

files_fmale = [ "svm(cast)-7Aug_Female.txt", "logreg(cast)-7Aug_Female.txt", "SVM-Lsa200-11-Aug_Female.txt", "Logreg-Lsa200-11-Aug_Female.txt", "pearson_Big-10Aug_Female.txt", "jaccard_Big-10Aug_Female.txt", "cosine_Big-9Aug-Female.txt"]
names_fmale = [ "SVM - Actresses", "Logistic Reg - Actresses", "SVM with LSA - Actresses", "Logistic Reg with LSA - Actresses", "PCC - Actresses", "Jaccard Similarity - Actresses", "Cosine Similarity - Actresses " ]


files_cosine_f = ["jaccard-24-Jul_Female.txt", "pearson-24-Jul_Female.txt", "cosine-22-Jul_Female.txt"]
names_cosine_f = ["Jaccard Similarity - Actresses", "PCC - Actresses", "Cosine Similarity -  Actresses"]

files_cosine_m = ["jaccard-24-Jul_Male.txt", "pearson-24-Jul_Male.txt", "cosine-22-Jul_Male.txt"]
names_cosine_m = ["Jaccard Similarity - Actors", "PCC - Actors", "Cosine Similarity -  Actors"]

def cosines():
	for j in range(len(files_cosine_f)):
		with open('resl/'+files_cosine_f[j]) as f:	
		    data = json.load(f)

		vals = data.values()

		x = []
		y = []
		for i in range(0, len(vals)):
			x += [vals[i][0]]
			# y_test = vals[i][6]
			# yt = np.where(np.array(y_test) ==1)
			# y_pred = vals[i][7]	
			# if sum(y_pred) > 0:
			# 	yn = np.where(np.array(y_pred) ==1)
			# 	yf = np.intersect1d(yt, yn)
			# 	y += [(1.0*len(yf))/sum(y_pred)]
			# else:
			# 	y += [0]
			y += [vals[i][4]]
			

		plt.figure(num=names_cosine_f[j])       
		plt.plot(x, y, 'go')
		plt.axis([0, 160, 0, 1.1])
		plt.xlabel("Acting credits")
		plt.ylabel("AUC - ROC")
		plt.title(names_cosine_f[j])
		plt.legend(loc='best', ncol=1, fancybox=True, shadow=True)
		plt.tight_layout()
		# plt.show()
		plt.savefig("resl/"+"roc_auc/"+names_cosine_f[j]+".png")


	
	for j in range(len(files_cosine_m)):
		with open('resl/'+files_cosine_m[j]) as f:	
		    data = json.load(f)

		vals = data.values()

		x = []
		y = []
		for i in range(0, len(vals)):
			x += [vals[i][0]]
			# y_test = vals[i][6]
			# yt = np.where(np.array(y_test) ==1)
			# y_pred = vals[i][7]	
			# if sum(y_pred) > 0:
			# 	yn = np.where(np.array(y_pred) ==1)
			# 	yf = np.intersect1d(yt, yn)
			# 	y += [(1.0*len(yf))/sum(y_pred)]
			# else:
			# 	y += [0]
			y += [vals[i][4]]

		plt.figure(num=names_cosine_m[j])       
		plt.plot(x, y, 'go')
		plt.axis([0, 230, 0, 1.1])
		plt.xlabel("Acting credits")
		plt.ylabel("AUC - ROC")
		plt.title(names_cosine_m[j])
		plt.legend(loc='best', ncol=1, fancybox=True, shadow=True)
		plt.tight_layout()
		# plt.show()
		plt.savefig("resl/"+"roc_auc/"+names_cosine_m[j]+".png")

def results():	
	for j in range(len(files_fmale)):
		with open('resl/'+files_fmale[j]) as f:	
		    data = json.load(f)

		print files_fmale[j]
		vals = data.values()

		x = []
		y = []
		z = []
		prec = []
		rec = []
		f1 = []
		for i in range(0, len(vals)):
			x += [vals[i][0]]
			# y_test = vals[i][6]
			# yt = np.where(np.array(y_test) ==1)
			# y_pred = vals[i][7]	
			# if sum(y_pred) > 0:
			# 	yn = np.where(np.array(y_pred) ==1)
			# 	yf = np.intersect1d(yt, yn)
			# 	y += [(1.0*len(yf))/sum(y_pred)]
			# else:
			# 	y += [0]
			y += [vals[i][4]]
			z += [vals[i][5]]
			y_true = vals[i][6]
			y_pred = vals[i][7]
			prec += [precision_score(y_true, y_pred)]
			rec += [recall_score(y_true, y_pred)]
			f1 += [f1_score(y_true, y_pred)]

		print "ROC: ", np.mean(np.array(y))
		print "Prec: ", np.mean(np.array(prec))
		print "Rec: ", np.mean(np.array(rec))
		print "F1: ", np.mean(np.array(f1))
		plt.figure(num=names_fmale[j])       
		plt.plot(x, f1, 'go')
		plt.axis([0, 160, 0, 1.1])
		plt.xlabel("Acting credits")
		plt.ylabel("AUC - ROC")
		plt.title(names_fmale[j])
		plt.legend(loc='best', ncol=1, fancybox=True, shadow=True)
		plt.tight_layout()
		# plt.show()
		plt.savefig("resl/"+"f1/"+names_fmale[j]+".png")


	for j in range(len(files_male)):
		with open("resl/"+files_male[j]) as f:	
		    data = json.load(f)

		vals = data.values()
		print files_male[j]
		x = []
		y = []
		z = []
		prec = []
		rec = []
		f1 = []
		for i in range(0, len(vals)):
			x += [vals[i][0]]
			# y_test = vals[i][6]
			# yt = np.where(np.array(y_test) ==1)
			# y_pred = vals[i][7]	
			# if sum(y_pred) > 0:
			# 	yn = np.where(np.array(y_pred) ==1)
			# 	yf = np.intersect1d(yt, yn)
			# 	y += [(1.0*len(yf))/sum(y_pred)]
			# else:
			# 	y += [0]
			y += [vals[i][4]]
			z += [vals[i][5]]
			y_true = vals[i][6]
			y_pred = vals[i][7]
			prec += [precision_score(y_true, y_pred)]
			rec += [recall_score(y_true, y_pred)]
			f1 += [f1_score(y_true, y_pred)]

		print "ROC: ", np.mean(np.array(y))
		print "Prec: ", np.mean(np.array(prec))
		print "Rec: ", np.mean(np.array(rec))
		print "F1: ", np.mean(np.array(f1))
		plt.figure(num=names_male[j])       
		plt.plot(x, f1, 'go')
		plt.axis([0, 230, 0, 1.1])
		plt.xlabel("Acting credits")
		plt.ylabel("AUC - ROC")
		plt.title(names_male[j])
		plt.legend(loc='best', ncol=1, fancybox=True, shadow=True)
		plt.tight_layout()
		# plt.show()
		plt.savefig("resl/"+"f1/"+names_male[j]+".png")


	for j in range(len(files_cosine_f)):
		with open('resl/'+files_cosine_f[j]) as f:	
		    data = json.load(f)

		vals = data.values()
		print files_cosine_f[j]
		x = []
		y = []
		z = []
		prec = []
		rec = []
		f1 = []
		for i in range(0, len(vals)):
			x += [vals[i][0]]
			# y_test = vals[i][6]
			# yt = np.where(np.array(y_test) ==1)
			# y_pred = vals[i][7]	
			# if sum(y_pred) > 0:
			# 	yn = np.where(np.array(y_pred) ==1)
			# 	yf = np.intersect1d(yt, yn)
			# 	y += [(1.0*len(yf))/sum(y_pred)]
			# else:
			# 	y += [0]
			y += [vals[i][4]]
			z += [vals[i][5]]
			y_true = vals[i][6]
			y_pred = vals[i][7]
			prec += [precision_score(y_true, y_pred)]
			rec += [recall_score(y_true, y_pred)]
			f1 += [f1_score(y_true, y_pred)]

		print "ROC: ", np.mean(np.array(y))
		print "Prec: ", np.mean(np.array(prec))
		print "Rec: ", np.mean(np.array(rec))
		print "F1: ", np.mean(np.array(f1))
		plt.figure(num=names_cosine_f[j])       
		plt.plot(x, f1, 'go')
		plt.axis([0, 6, 0, 1.1])
		plt.xlabel("Acting credits")
		plt.ylabel("AUC - ROC")
		plt.title(names_cosine_f[j])
		plt.legend(loc='best', ncol=1, fancybox=True, shadow=True)
		plt.tight_layout()
		# plt.show()
		plt.savefig("resl/"+"f1/"+names_cosine_f[j]+"_1.png")


	for j in range(len(files_cosine_m)):
		with open('resl/'+files_cosine_m[j]) as f:	
		    data = json.load(f)

		vals = data.values()
		print files_cosine_m[j]
		x = []
		y = []
		z = []
		prec = []
		rec = []
		f1 = []
		for i in range(0, len(vals)):
			x += [vals[i][0]]
			# y_test = vals[i][6]
			# yt = np.where(np.array(y_test) ==1)
			# y_pred = vals[i][7]	
			# if sum(y_pred) > 0:
			# 	yn = np.where(np.array(y_pred) ==1)
			# 	yf = np.intersect1d(yt, yn)
			# 	y += [(1.0*len(yf))/sum(y_pred)]
			# else:
			# 	y += [0]
			y += [vals[i][4]]
			z += [vals[i][5]]
			y_true = vals[i][6]
			y_pred = vals[i][7]
			prec += [precision_score(y_true, y_pred)]
			rec += [recall_score(y_true, y_pred)]
			f1 += [f1_score(y_true, y_pred)]

		print "ROC: ", np.mean(np.array(y))
		print "Prec: ", np.mean(np.array(prec))
		print "Rec: ", np.mean(np.array(rec))
		print "F1: ", np.mean(np.array(f1))
		plt.figure(num=names_cosine_m[j])       
		plt.plot(x, f1, 'go')
		plt.axis([0, 6, 0, 1.1])
		plt.xlabel("Acting credits")
		plt.ylabel("AUC - ROC")
		plt.title(names_cosine_m[j])
		plt.legend(loc='best', ncol=1, fancybox=True, shadow=True)
		plt.tight_layout()
		# plt.show()
		plt.savefig("resl/"+"f1/"+names_cosine_m[j]+"_1.png")

results()