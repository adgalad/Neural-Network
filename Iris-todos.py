import sys
from nn import NN

nn = NN()

if (len(sys.argv) < 5):
	print("Usage:\n   python Iris-todos.py training_file learning_rate number_hidden test_file")
	quit()

#ARGUMENTS
trainingFile = open(sys.argv[1])
n = float(sys.argv[2])
nhidden = int(sys.argv[3])
testFile = open(sys.argv[4])

#TRAINING SET
trainingSet = []
testSet = []
for line in trainingFile:
	string = line.split(",", 5)
	x = [[float(string[0])/10.0,float(string[1])/10.0,float(string[2])/10.0,float(string[3])/10.0]]
	if string[4].rstrip() == "Iris-setosa":
		x += [[1,0]]
	elif string[4].rstrip() == "Iris-versicolor":
		x += [[0,1]]
	else:
		x += [[1,1]]
	trainingSet += [x]

#TEST SET
for line in testFile:
	string = line.split(",", 5)
	x = [[float(string[0])/10.0,float(string[1])/10.0,float(string[2])/10.0,float(string[3])/10.0]]
	if string[4].rstrip() == "Iris-setosa":
		x += [[1,0]]
	elif string[4].rstrip() == "Iris-versicolor":
		x += [[0,1]]
	else:
		x += [[1,1]]
	testSet += [x]

trainingFile.close()
testFile.close()

#TRAINING
nn.backPropagation(trainingSet, n, 4, nhidden, 2)

#TESTING
p = 0
for x,t in trainingSet:
	
	O = nn.evaluate(x)
	o1 = 0
	o2 = 0
	if (1 - O[0]) < 0.1:
		o1 = 1

	if (1 - O[1]) < 0.1:
		o2 = 1

	if (o1 == t[0] and o2== t[1]):
		p+=1

print(str(p)+" out of "+str(len(testSet))+" tests")