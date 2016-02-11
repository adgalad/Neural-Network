import sys
from nn import NN
import matplotlib.pyplot as plt

nn = NN()

if (len(sys.argv) < 5):
	print("Usage:\n   python Area.py training_file learning_rate number_hidden test_file")
	quit()

#ARGUMENTS
trainingFile = open(sys.argv[1])
n = float(sys.argv[2])
nhidden = int(sys.argv[3])
testFile = open(sys.argv[4])

trainingSet = []
testSet = []

#TRAINING SET
for line in trainingFile:
	string = line.split(" ", 3)
	x = [[float(string[0])/20.0,float(string[1])/20.0]]
	if int(string[2]) == 1 :
		x += [[1]]
	else:
		x += [[0]]
	trainingSet += [x]

#TEST SET
for line in testFile:
	string = line.split(" ", 3)
	x = [[float(string[0])/20.0,float(string[1])/20.0]]
	if int(string[2]) == 1 :
		x += [[1]]
	else:
		x += [[0]]
	testSet += [x]

trainingFile.close()
testFile.close()

#TRAINING
nn.backPropagation(trainingSet, n, 2, nhidden, 1)

#TESTING
p = 0
xr = []
yr = []
xc = []
yc = []
for x,t in testSet:
	O = nn.evaluate(x)
	if((1 - O[0]) < 0.2):
		if (t[0] == 1):				
			p+=1				
		xr += [x[0]]
		yr += [x[1]]
	else:
		if (t[0] == 0):				
			p+=1
		xc += [x[0]]
		yc += [x[1]]
print(p)

#PLOT
# f1, p2 = plt.subplots()
# f2, p2 = plt.subplots()
# p1.plot(nn.errorX, nn.errorY,"-")
# p1.set_title("Error")
# p2.plot(xr, yr, 'ro')
# p2.plot(xc, yc, 'bs')
# p2.set_aspect('equal', adjustable='box')
# p2.set_title(str(p)+" de "+str(len(testSet))+" pruebas exitosas")

plt.plot(xr, yr, 'ro')
plt.plot(xc, yc, 'bs')
plt.gca().set_aspect('equal', adjustable='box')
plt.title(str(p)+" out of "+str(len(testSet))+" tests")

plt.show()



