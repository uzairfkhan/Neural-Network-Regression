import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing and Splitting data.
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)         #For Online
#raw_df = pd.read_csv('boston.data', sep="\s+", skiprows=22, header=None)   #For Offline
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
data = np.array(data)

#Setting Seed
np.random.seed(123)

#Making a class for layers - Defining weights and Bias
class Layers():
    def __init__(self,input,output):
        self.weights = np.random.normal(size=(input,output)) #Generating normal random values for weights
        self.bias = np.random.normal(size=output)

#Making a class for NN - Defining neurons in layers
class Neural_Network():
    def __init__(self):
        self.layer1 = Layers(13,3) #Defining the attributes of layer 1
        self.layer2 = Layers(3, 1) #Defining the attributes of layer 2

#Forward Propogation Function
    def Forward_Propogation(self,Z):
        self.layer1_output = Relu(np.dot(Z,self.layer1.weights) + self.layer1.bias)
        self.layer2_output = Relu(np.dot(self.layer1_output,self.layer2.weights) + self.layer2.bias)
        return np.copy(self.layer2_output)

#Back Propogation Function
    def Backpropagation(self,target,pred,Z,lr=0.000000001):
        delta = (pred-target) * (derivative_Relu(self.layer2_output)) #Delta Rule
        l2 = delta * np.reshape(self.layer1_output,(3,1))
        l1 = (delta * derivative_Relu(self.layer1_output)) * (np.reshape(Z,(13, 1)))
        self.layer1.weights = self.layer1.weights + (l1*-lr) #Updating the Values of layer 1 weights
        self.layer2.weights = self.layer2.weights + (l2*-lr) #Updating the Values of layer 2 weights
        #print(nn.layer2.weights)

def Relu(Z):
    return np.maximum(0,Z)
def derivative_Relu(val):
    val[val<=0] = 0
    val[val>0] = 1
    return val
train_x , train_y = data[0:300],target[0:300]
test_x , test_y = data[300:] , target[300:]

nn = Neural_Network()
#Running Epochs for training
epochTrain = 100
errorTrain = []
avgTrain = []
pred_arrTrain = []
target_arrTrain = []
predictionsTrain = 0
for i in range(epochTrain):
    for j in range(len(train_x)):

        predictionsTrain = nn.Forward_Propogation(train_x[j])
        pred_arrTrain.append(predictionsTrain)
        err = (train_y[j] - predictionsTrain) ** 2 / 2
        errorTrain.append(err)
        nn.Backpropagation(train_y[j], predictionsTrain, train_x[j])
        print(f'Epoch={i+1}\nTarget = {train_y[j]}\nError={err}\nPrediction = {predictionsTrain}\n')
        target_arrTrain.append(train_y[j])
    avgTrain.append(sum(errorTrain) / len(errorTrain))
    errorTrain = []

#Running Epochs for testing
epochTest = 100
errorTest = []
avgTest = []
pred_arrTest = []
target_arrTest = []
predictionsTest = 0
for i in range(epochTest):
    for j in range(len(test_x)):

        predictionsTest = nn.Forward_Propogation(test_x[j])
        pred_arrTest.append(predictionsTest)
        err = (test_y[j] - predictionsTest) ** 2 / 2
        errorTest.append(err)
        nn.Backpropagation(test_y[j], predictionsTest, test_x[j])
        print(f'Epoch={i+1}\nTarget = {test_y[j]}\nError={err}\nPrediction = {predictionsTest}\n')
        target_arrTest.append(test_y[j])
    avgTest.append(sum(errorTest) / len(errorTest))
    errorTest = []


#plotting error graph for testing and training Epochs
plt.figure(figsize=(10,6))
#plt.plot(avgTrain)
plt.plot(avgTest)
#plt.legend(['Train','Test'])
plt.show()

#plt.figure(figsize=(10,6))
# plt.plot(pred_arr,target_arr)
# plt.show()


