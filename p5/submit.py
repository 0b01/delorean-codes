import requests, json

url = "https://captcha.delorean.codes/u/rickyhan/solution"

def post(list_of_solved):
	r = requests.post(url, data=json.dumps({"solutions": list_of_solved}))
	print r.text

data = []
solutions = [i.split(',') for i in open('outfile2', 'r').read().split('\n')]
for sol in solutions:
	data.append({'name': sol[0], 'solution': sol[1]})

# print len(solutions)
post(data)