#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import scipy.io
import glob
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import CSI

def inf_helper(y):
    return np.isinf(y), lambda z: z.nonzero()[0]

def load_data():
    X_train = np.zeros((840,1,342,2000))
    X_val = np.zeros((180,1,342,2000))
    X_test = np.zeros((180,1,342,2000))
    y_train = np.zeros((840,))
    y_val = np.zeros((180,))
    y_test = np.zeros((180,))
    # load train
    n=0
    for i,name in enumerate(['walk_train','run_train','fall_train','clean_train','circle_train','box_train']):
        path = './normal/train/'+name+'/*.mat'
        num=i
        for filepath in glob.iglob(path):
            mat = scipy.io.loadmat(filepath)
            x = mat['CSIamp']
            X_train[num,] = x
            y_train[num] = n
            num += 6
        n += 1
    for s in range(840):
        for i in range(342):
            y = X_train[s,0,i]
            inf, x = inf_helper(y)
            y[inf]= np.interp(x(inf), x(~inf), y[~inf])
    #load val
    n = 0
    for i,name in enumerate(['walk_val','run_val','fall_val','clean_val','circle_val','box_val']):
        path = './normal/val/'+name+'/*.mat'
        num=i
        for filepath in glob.iglob(path):
            mat = scipy.io.loadmat(filepath)
            x = mat['CSIamp']
            X_val[num,] = x
            y_val[num] = n
            num += 6
        n += 1
    for s in range(180):
        for i in range(342):
            y = X_val[s,0,i]
            inf, x = inf_helper(y)
            y[inf]= np.interp(x(inf), x(~inf), y[~inf])
    #load test
    n = 0
    for i,name in enumerate(['walk_test','run_test','fall_test','clean_test','circle_test','box_test']):
        path = './normal/test/'+name+'/*.mat'
        num=i
        for filepath in glob.iglob(path):
            mat = scipy.io.loadmat(filepath)
            x = mat['CSIamp']
            X_test[num,] = x
            y_test[num] = n
            num += 6
        n += 1
    for s in range(180):
        for i in range(342):
            y = X_test[s,0,i]
            inf, x = inf_helper(y)
            y[inf]= np.interp(x(inf), x(~inf), y[~inf])
    
    X = np.concatenate((X_train,X_val,X_test),axis=0)
    X = (X - X.min())/(X.max() - X.min())
    y = np.concatenate((y_train,y_val,y_test),axis=None)
    
    X = X[:,:,:,::4]
    X = X.reshape(1200,3,114,500)
    X_train = X[0:840]
    y_train = y[0:840]
    X_test = X[840:1200]
    y_test = y[840:1200]
    
    return X_train, X_test, y_train, y_test

class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss,self).__init__()
    
    def forward(self,original,recovered):
        loss = (torch.mean(torch.square(original-recovered))/torch.sum(torch.square(original)))
        return loss

def train(model, tensor_loader, num_epochs, batch_size,learning_rate,device):
    model = model.to(device)
    criterion1 = NMSELoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        predict_loss = 0
        quantize_loss = 0
        num_batch = len(tensor_loader.dataset) / batch_size
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            vq_loss, r_x, y_p, perplexity = model(inputs)
            r_x = r_x.to(device)
            y_p = y_p.to(device)
            y_p = y_p.type(torch.FloatTensor)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            loss1 = criterion1(inputs,r_x)
            loss2 = criterion2(y_p,labels)
            loss = loss1 + loss2 + vq_loss
            loss.backward()
            optimizer.step()
            epoch_loss += (loss).item() * inputs.size(0)
            predict_y = torch.argmax(y_p,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/num_batch
        print('Epoch:{}, Train_Accuracy:{:.4f},Train_Loss:{:.14f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
        scheduler.step()
    return

def test(model, test_loader, device):
    criterion1 = NMSELoss()
    # criterion1 = nn.MSELoss()
    for step, data in enumerate(test_loader, start=0):
        model.to(device)
        inputs, labels = data
        inputs = inputs.to(device)
        vq_loss, r_x, y_p, perplexity = model(inputs)
        r_x = r_x.to(device)
        loss = criterion1(inputs,r_x)
        labels = labels.type(torch.LongTensor)
        labels.to(device)
        y_p = y_p.type(torch.FloatTensor)
        y_p.to(device)
        predict_y = torch.argmax(y_p,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        print("step:{},test_accuracy:{:.4f},test_rebuild_loss:{:.18f}".format(step,float(accuracy),float(loss)))
    return

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 90
    batch_size = 128
    learning_rate = 1e-2
    
    X_train, X_test, y_train, y_test = load_data()
    print("shape of X_train is:{}\nshape of X_test is:{}\nshape of y_train is:{}\nshape of y_test is:{}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    train_x = torch.Tensor(X_train)
    train_y = torch.Tensor(y_train)
    train_set = torch.utils.data.TensorDataset(train_x,train_y)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_x = torch.Tensor(X_test)
    test_y = torch.Tensor(y_test)
    test_set = torch.utils.data.TensorDataset(test_x,test_y)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=120,shuffle=True)
    
    embedding_dim = 32
    num_embeddings = 512
    commitment_cost = 1
    model = CSI(num_embeddings, embedding_dim, commitment_cost)
    
    train(
        model = model,
        tensor_loader = train_loader,
        num_epochs = num_epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
        device = device
     )
    test(
        model = model,
        test_loader = test_loader,
        device = device
     )
    return

#execute
main()