%envi_perc.py
%multilayer perceptron to reconstruct SSTDR waveforms from environment data

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np

# ======= CONSTANTS ==========
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_RATIO = 0.25

# ======== NETWORK DEFINITION ====
class Net(nn.Module):
    def __init(self):
        super(Net, self).__init__()
        
        self.net = nn.sequential(
            nn.Linear(3,16),
            nn.Relu(),
            nn.Linear(16,92),
        )
    
    def forward(self,x):
        results = self.net(x)
        return results

# ======= LOAD DATA ==========
#TODO: data only locally stored on work computer, will locate/run from there
#env_data
#bl_measured

#============ SPLIT DATA ==============
x_full = env_data
border = int(len(x_full)*TEST_RATIO)
x_train = x_full[0:border]
x_test = x_full[border:]

y_full = bl_measured
y_train = y_full[0:border]
y_test = y_full[border:]

N_train - len(x_train)
train_indices = list(range(N_train))

# ========= TRAINING ===========
network = Net().double()
optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
loss_func = nn.MSELoss()

for epoch in range(EPOCHS):
    #shuffle order of training data
    random.shuffle(train_indices)
    x_train_s = [x_train[i] for i in train_indices]
    y_train_s = [y_train[i] for i in train_indices]
    for b in range(int(N_train/BATCH_SIZE)):
        #for each batch
        #get batch of input data
        b_data = x_train_s[b*BATCH_SIZE:min(N_train, (b+1)*BATCH_SIZE)]
        b_x = torch.from_numpy(b_data).view(-1,3).double() #batch_size by 3 tensor
        
        #get batch of desired data
        b_desired = y_train_s[b*BATCH_SIZE:min(N_train, (b+1)*BATCH_SIZE)]
        b_y = torch.from_numpy(b_desired).view(-1,92).double() #batch size by 92 tensor
        
        #predict
        predictions = network(b_x)
        
        #update
        loss = loss_func(predictions, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print learning
        if b%100 == 0:
            print('Epoch: ' + str(epoch) + ', loss: %.4f' % loss.data.numpy())

#======= TESTING ===========
x_test_tensor = torch.from_numpy(x_test).view(-1,3).double()
test_results = network(x_test_tensor)
N_test = len(x_test)
cc = np.zeros(N_test)
for i in range(N_test):
    cc[i] = np.corrcoef(y_test[i], test_results[i])[0,1]
plt.plot(cc)
plt.title("Testing Set Prediction Corr. Coeffs")
plt.ylabel("CC")
plt.xlabel("Sample")
plt.show()