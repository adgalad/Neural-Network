import matplotlib.pyplot as plt
import random
p = []
xr = []
yr = []
xc = []
yc = []
f = open("pruebaArea2000.txt", "w+")
while (len(xr)<1000) or (len(xc)<1000):
	j = random.randint(0,99)
	k = random.randint(0,99)
	if pow(j-50,2) + pow(k-50,2) < 1225:
		if len(xc) < 1000:
			if (len(xc) != 0):
				string = str(j/5.0)+" "+str(k/5.0)+" "+str(-1) 
				f.write(string+"\n")
				print(string)
				xc += [j/100.0]
				yc += [k/100.0]
			else:
				if not((j/100.0) in xc):
					string = str(j/5.0)+" "+str(k/5.0)+" "+str(-1) 
					f.write(string+"\n")
					print(string)
					xc += [j/100.0]
					yc += [k/100.0]
				else:
					if (yc[xc.index(j/100.0)] != (k/100.0)):
						string = str(j/5.0)+" "+str(k/5.0)+" "+str(-1) 
						f.write(string+"\n")
						print(string)
						xc += [j/100.0]
						yc += [k/100.0]
	else:
		if len(xr) < 1000:
			if (len(xr) != 0):
				string = str(j/5.0)+" "+str(k/5.0)+" "+str(1) 
				f.write(string+"\n")
				print(string)
				xr += [j/100.0]
				yr += [k/100.0]
			else:
				if not((j/100.0) in xr):
					string = str(j/5.0)+" "+str(k/5.0)+" "+str(1) 
					f.write(string+"\n")
					print(string)
					xr += [j/100.0]
					yr += [k/100.0]
				else:
					if (yr[xr.index(j/100.0)] != (k/100.0)):
						string = str(j/5.0)+" "+str(k/5.0)+" "+str(1) 
						f.write(string+"\n")
						print(string)
						xr += [j/100.0]
						yr += [k/100.0]

plt.plot(xr, yr, 'ro')
plt.plot(xc, yc, 'bs')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()