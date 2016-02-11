import matplotlib.pyplot as plt
p = []
xr = []
yr = []
xc = []
yc = []
f = open("pruebaArea.txt", "w+")
for i in range(100):
	for j in range(100):
		if pow(i-50,2) + pow(j-50,2) < 1225:
			string = str(i/5.0)+" "+str(j/5.0)+" "+str(-1) 
			f.write(string+"\n")
			print(string)
			xc += [i/100.0]
			yc += [j/100.0]
		else:
			string = str(i/5.0)+" "+str(j/5.0)+" "+str(1) 
			f.write(string+"\n")
			print(string)
			xr += [i/100.0]
			yr += [j/100.0]

plt.plot(xr, yr, 'ro')
plt.plot(xc, yc, 'bs')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

