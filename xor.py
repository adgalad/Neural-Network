

from nn import NN

nn = NN()

train = [[[0,0],[0]],[[0,1],[1]],[[1,0],[1]],[[1,1],[0]]]

for x in train:
	print(x)

nn.backPropagation(train,0.5,2,3,1)

print(nn.evaluate([0,0]))
print(nn.evaluate([0,1]))
print(nn.evaluate([1,0]))
print(nn.evaluate([1,1]))