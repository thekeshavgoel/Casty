from flask import *
import casty

app = Flask(__name__)

app.debug = True

@app.route('/', methods=['GET'])
def search():
	return render_template('search.html')

@app.route('/find', methods=['GET'])
def dropdown():
	fname = request.args.get('first_actor_name')
	lname = request.args.get('last_actor_name')
	algo = request.args.get('algo')
	actors = casty.findActor(fname, lname)
	return render_template('select.html', actors=actors, algo=algo)

@app.route('/actor', methods=['GET'])
def actor():
	actor = request.args.get('actor')
	algo = request.args.get('algo')
	name = request.args.get('name')
	if algo == "logreg":
		predicts = casty.logreg(int(actor))
	elif algo == "svm":
		predicts = casty.svmc(int(actor))
	else:
		predicts = casty.cosine_sim(int(actor))
	leng = predicts[0]
	title = predicts[1]
	prob =  predicts[2]
	clasf = predicts[3]
	actual = predicts[4]
	sums = predicts[5]
	# for i in range(0, len(title)):
	# 	print(title[i], prob[i])
	return render_template('actor.html', name=name, actor_id=actor, algo=algo, titles=title, prob=prob, clasf=clasf, actual=actual, length=len(title), lengh=leng, sums = sums)

@app.route('/actor_more', methods=['GET'])
def actor_more():
	actor = request.args.get('actor')
	algo = request.args.get('algo')
	name = request.args.get('name')
	if algo == "logreg":
		predicts = casty.logreg(int(actor), 1)
	elif algo == "svm":
		predicts = casty.svmc(int(actor), 1)
	else:
		predicts = casty.cosine_sim(int(actor), 1)
	leng = predicts[0]
	title = predicts[1]
	prob =  predicts[2]
	clasf = predicts[3]
	actual = predicts[4]
	sums = predicts[5]
	# for i in range(0, len(title)):
	# 	print(title[i], prob[i])
	return render_template('actor.html', name=name, actor_id=actor, algo=algo, titles=title, prob=prob, clasf=clasf, actual=actual, length=len(title), lengh=leng, sums = sums)


if __name__ == "__main__":
    app.run()