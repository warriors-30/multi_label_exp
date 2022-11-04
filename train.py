import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import sklearn.metrics as metrics
import csv


class FashionDataset(Dataset):
    def __init__(self,annotation_path):
    # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.color_labels = []
        self.gender_labels = []
        self.article_labels = []

    # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.color_labels.append(self.attr.color_name_to_id[row['baseColour']])
                self.gender_labels.append(self.attr.gender_name_to_id[row['gender']])
                self.article_labels.append(self.attr.article_name_to_id[row['articleType']])


class custom_resnet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        resnet=models.resnet50(pretrained=True)
        resnet.fc=nn.Linear(in_features=2048,out_features=num_classes,bias=True)
        self.backbone=resnet
        self.sigm=nn.Sigmoid()

    def forward(self,x):
        return self.sigm(self.backbone(x))

def calculate_metrics(pred,target,threshhold=0.5):
    pred=np.array(pred>threshhold,dtype=float)
    return {
        'micro/precision':metrics.precision_score(y_true=target,y_pred=pred,average='micro'),
        'micro/recall':metrics.recall_score(y_true=target,y_pred=pred,average='micro'),
        'micro/f1':metrics.f1_score(y_true=target,y_pred=pred,average='micro')
    }

def train(data_loader,model,loss_fn,optimizer):
    size=len(data_loader.dataset)
    total_loss=0
    model.train()
    for batch,(images,labels) in enumerate(data_loader):
        images,labels=images.to(device),labels.to(device)
        pred=model(images)
        loss=loss_fn(pred,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

        if batch%10 == 0:
            average_loss=total_loss/(batch+1)
            print(f"[{batch*len(images)}/{size}]  loss:{average_loss}")

def test(data_loader,model,loss_fn):
    size=len(data_loader.dataset)
    num_batches=len(data_loader)
    total_loss=0
    model.eval()
    with torch.no_grad():
        model_results=[]
        gt_labels=[]
        for images, labels in data_loader:
            images,labels=images.to(device),labels.to(device)
            pred=model(images)
            model_results.append(pred.cpu().numpy())
            gt_labels.append(labels.cpu().numpy())
            total_loss+=loss_fn(pred,labels).item()
    result=calculate_metrics(np.array(model_results),gt_labels,0.5)
    test_loss=total_loss/num_batches
    print(f"Test error: \n  loss:{test_loss} \n")
    print(f"micro precision:{result['micro/precision']}\nmicro recall:{result['micro/recall']}\n")
    print(f"micro f1:{result['micro/f1']}\n")

if __name__=='__main__':
    model=custom_resnet(27)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    batch_size=32
    epoch=35
    learning_rate=1e-3
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)

    print(f"model structure : {model.__class__}  config_name : {os.path.basename(__file__)}\n-----")
    print(f"optimizer : \n{optimizer.state_dict()}\n-----")

    for i in range(epoch):
        print(f"---epoch{i + 1}----")
        train(train_loader, model, criterion, optimizer)
        if (i != 0 and (i + 1) % 5 == 0):
            test(test_loader, model, criterion)
            # 权重保存路径
            save_path = os.path.join("workdirs/checkpoints", f"epoch_{i + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"save model to {save_path}")

