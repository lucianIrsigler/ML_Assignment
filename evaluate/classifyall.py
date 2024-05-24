import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

finalFile = "predlabels.txt"
testData = "testdata.txt"

def hot_one_encoding(number):
    nums = [0]*4
    nums[number]=1
    return nums

class Model(nn.Module):
    #h1=256,h2=128,out=21
    def __init__(self,in_features=1044,h1=512,h2=128,out=21):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(h1,h2)
        self.dropout2 = nn.Dropout(0.5)
        self.out = nn.Linear(h2,out)

    def forward(self,x):
        x = F.sigmoid(self.fc1(x))
        x = self.dropout1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x
    

with open(testData,"r") as f:
    trainData = [i.split(",") for i in f.readlines()]

pixel_arrays = []
for i in range(len(trainData)):
    pixels = list(map(float,trainData[i]))
    rotation_angle = pixels.pop()
    hotOneEncoding = hot_one_encoding(int(rotation_angle))

    min_val = min(pixels)
    max_val = max(pixels)
    
    for j in range(len(pixels)):
        pixels[j] = (pixels[j]-min_val)/(max_val-min_val)

    for j in hotOneEncoding:
        pixels.append(j)

    pixel_arrays.append(pixels)

inputData = np.array(pixel_arrays)
inputData = torch.FloatTensor(inputData)

model = Model()
model.load_state_dict(torch.load("classify.pt"))

model.eval()
y_pred = model(inputData)

with open(finalFile,"w") as f:
    for i in y_pred:
        f.write(f"{i.argmax().item()}\n")

