# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:16:00 2019

@author: HP
"""

#importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#importing the dataset
movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None, engine='python', encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None, engine='python', encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None, engine='python', encoding='latin-1')

#preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype = 'int')

#Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#converting the data into an array with users in line and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users +1):
        id_movies = data[:,1][data[:,0]== id_users]
        id_ratings = data[:,2][data[:,0]== id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)    

#converting the data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#creating the architecture of the neural network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.fc3(x))
            x = self.fc4(x)
            return x
        
sae = SAE()
criterion = MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay = 0.5)

#Training the SAE
nb_epoch = 200
for epoch in range(1,nb_epoch + 1):
    train_loss = 0
    s = 0. #the no. of users that rated at leasst one movie
    for id_user in range(nb_users):#for loop for all the users bcoz each observation corresponds to a user
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, criterion)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(train_loss/s))

#Testing the SAE
 test_loss = 0
    s = 0. #the no. of users that rated at leasst one movie
    for id_user in range(nb_users):#for loop for all the users bcoz each observation corresponds to a user
        input = Variable(training_set[id_user]).unsqueeze(0)#We will keep this training_set here
        #bcoz right now we are dealing with specific user and we want to take the input corresponding to
        #that user and the input means all the ratings of the movies that user has rated.So that's our
        #input vector and we put this vector into the network called auto encoder
        target =  Variable(test_set[id_user])#we will compare to the test set
        if torch.sum(target.data > 0) > 0:
            output = sae(input)#Here sae will predict ratings for all the movies user hasn't watched yet.Also sae contains our forward function
            #that returns the vector of predicted ratings
            target.require_grad = False#We use this to override the consierations of the gradient with respect to the target
            output[target == 0] = 0#it will only consider non-zero ratings in the test set
            loss = criterion(output, criterion)#this measures the loss
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)#Adjusting factor that will measure the relevantloss on the test set
            #loss.backward(), bcoz back propagation is not needed for test set
            test_loss += np.sqrt(loss.data[0]*mean_corrector)#Here we compute the loss accumulated with the user.We are dealing with right now the loop
            #the loss starts at zero then we come to the error between the predicted rating and the real rating for the first user.So we add this generated loss
            #to this loss here
            s += 1.
    print('test loss: '+str(test_loss/s))

#Continued of for loop:
#Then our autoencoder will look at the ratings of the movies and especially the positive ratings and based
#on these ratings it will predict the ratings of the movie that the user hasn't watched yet.So e.g if in our input
#vector our user give five star ratings to all the action movies that you watched then what we feed this input vector into the network
#Well the neurons corresponding to the specific features related to action movies will be activated with a large weight
#to predict high ratings for the other action movies that the user hasn't watched yet.    

#And then what we will do is we will compare this prdicted ratings to the ratings of the test set bcoz the test set contain the ratings
#that were not part of the train set i.e these action movies that the user hasn't watched yet in the training set
        
