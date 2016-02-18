import sys
from nn import NN
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sumaVector(a,b):
	return [x+y for x,y in zip(a,b)]



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



elif len(sys.argv) >= 4:
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

	errorPlot = []
	mediaX = []
	mediaY = []
	#TRAINING
	if len(sys.argv) > 6 and sys.argv[6] == ".":
		nvalues =[0.5, 0.4, 0.3, 0.2, 0.1]
		for i in range(5):
			for j in range(5):
				nn.backPropagation(trainingSet, nvalues[i] , 2, nhidden, 1)
				if (mediaY == []):
					mediaY = numpy.zeros(len(nn.errorX))
				mediaX = nn.errorX
				mediaY = sumaVector(mediaY, nn.errorY)
			mediaY = [y/5 for y in mediaY]
			errorPlot += [[mediaX,mediaY]]
		print(">>"+str(len(errorPlot)))
		i = 0

		for x,y in errorPlot:
			print(">>")
			print (x)
			print (y)
			plt.plot(x, y, '-', label=str(nvalues[i]))
			i+=1
		plt.legend()
		# plt.gca().set_aspect('equal', adjustable='box')
		plt.title("error")
		plt.savefig("./"+str(len(trainingSet))+"_ejemplos_"+str(nhidden)+"neuronas.png")
		quit()
	else:
		nn.backPropagation(trainingSet, n , 2, nhidden, 1)	
		if len(sys.argv) > 5:
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

for x,t in testSet:
	O = nn.evaluate(x)
	if O[0] > 0.5:
		if t[0] == 1: ##THRESHOLD 0.5
			p += 1
		xr += [x[0]]
		yr += [x[1]]
	elif O[0] <= 0.5 :
		if t[0] == 0:				
			p += 1
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

plt.plot(xr, yr, 'rs')
plt.plot(xc, yc, 'bs')

plt.gca().set_aspect('equal', adjustable='box')
plt.title(str(p)+" out of "+str(len(testSet))+" tests")

plt.show()




