import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

finalFile = "predlabels.txt"
testData = "testdata.txt"

class Model(nn.Module):
    def __init__(self,in_features=1041,h1=256,h2=128,out=21,embedding_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out)

    def forward(self,x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.out(x)
        return x

    
with open(testData,"r") as f:
    trainData = [i.split(",") for i in f.readlines()]


pixel_arrays = []
for i in range(len(trainData)):
    pixels = list(map(float,trainData[i][0:1040]))
    rotation_angle = int(float(trainData[i][-1].strip()))

    min_val = min(pixels)
    max_val = max(pixels)
    
    for j in range(1040):
        pixels[j] = (pixels[j]-min_val)/(max_val-min_val)

    pixels.append(rotation_angle)
    pixel_arrays.append(pixels)


inputData = np.array(pixel_arrays)
inputData = torch.FloatTensor(inputData)

model = Model()
model.load_state_dict(torch.load("classify.pt"))

y_pred = model(inputData)

with open(finalFile,"w") as f:
    for i in y_pred:
        f.write(f"{i.argmax().item()}\n")

