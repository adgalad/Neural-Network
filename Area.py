import sys
from nn import NN
import matplotlib.pyplot as plt

nn = NN()
trainingSet = []
testSet = []
if len(sys.argv) == 3:
	nn.loadWeights(sys.argv[1])
	testFile = open(sys.argv[2])

	#TEST SET	
	for line in testFile:
		string = line.split(" ", 3)
		x = [[float(string[0])/20.0,float(string[1])/20.0]]
		if int(string[2]) == 1 :
			x += [[1]]
		else:
			x += [[0]]
		testSet += [x]



elif len(sys.argv) == 6:
	#ARGUMENTS
	trainingFile = open(sys.argv[1])
	n = float(sys.argv[2])
	nhidden = int(sys.argv[3])
	testFile = open(sys.argv[4])



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
	nn.saveWeights(sys.argv[5])

else:
	print("Usage:\n   python Area.py training_file learning_rate number_hidden_neurons test_file \n   python Area.py weight_file test_file")
	quit()


#TESTING
p = 0
xr = []
yr = []
xc = []
yc = []
xn1 = []
yn1 = []
xn2 = []
yn2 = []
for x,t in testSet:
	O = nn.evaluate(x)
	if(O[0] > 0.9): ##THRESHOLD 0.2
		if (t[0] == 1):				
			p+=1				
		# else:
		# 	print(t[0],O[0])
		xr += [x[0]]
		yr += [x[1]]
	elif O[0] < 0.1:
		if (t[0] == 0):				
			p+=1
		# else:
		# 	print(t[0],O[0])
		xc += [x[0]]
		yc += [x[1]]
	else:
		if (t[0] == 0):
			xn1 += [x[0]]
			yn1 += [x[1]]
		else:
			xn2 += [x[0]]
			yn2 += [x[1]]
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

plt.plot(xr, yr, 'rs')
plt.plot(xc, yc, 'bs')
plt.plot(xn1, yn1, 'gs')
plt.plot(xn2, yn2, 'ys')

plt.gca().set_aspect('equal', adjustable='box')
plt.title(str(p)+" out of "+str(len(testSet))+" tests")

plt.show()



