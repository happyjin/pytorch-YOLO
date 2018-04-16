
# coding: utf-8

# In[21]:


import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from load_dataset import *
from YoloLoss import YoloLoss
from network import *



loss_name = 'loss_150iters_S14_1000feas_vot20classes_fixlossbug.h5'
model_name = 'model_150iters_S14_1000feas_vot20classes_fixlossbug.pth'

# ### time start

# In[23]:


start_time = time.time()


# ### gpu usage

# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
use_gpu = torch.cuda.is_available()


# ### dataset and file folder

# In[3]:


file = './routine_generate_vot2017_train/vot2017_train.txt'
img_folder = './routine_generate_vot2017_train/'


# ### set hyperparameters

# In[4]:


learning_rate = 0.0006
img_size = 224
num_epochs = 150
lambda_coord = 5
lambda_noobj = .5
n_batch = 64
S = 14
B = 2
C = 20
n_features = 1000


# ### load pre-trained vgg 16 model

# In[5]:


model = models.vgg16(pretrained=True)  


# In[6]:


model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, n_features),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(),
        nn.Linear(n_features, (B*5+C) * S * S),
        nn.Sigmoid(),
    )


# initialize the weights and biases for the linear block of the model

# In[7]:


for m in model.modules():
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


# In[8]:


print(model)
print('pre-trained vgg16 model has loaded!')
print('')


# utilize gpu to speed up if it is avaliable

# In[9]:


if use_gpu:
    model.cuda()


# ### input pipeline

# In[10]:


train_dataset = VotTrainDataset(img_folder=img_folder, file=file, img_size=img_size, S=S, B=B, C=C, transforms=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True, num_workers=2)


# ### set model into train mode

# In[11]:


model.train()


# ### set loss function and optimizer

# In[12]:


loss_fn = YoloLoss(n_batch, B, C, lambda_coord, lambda_noobj, use_gpu=use_gpu)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)


# ### training

# In[13]:


loss_list = []


# In[14]:

loss_record = []
for epoch in range(num_epochs):
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = model(images)
        loss = loss_fn(pred,target)
        current_loss = loss.data.cpu().numpy()[0]
        loss_list.append(current_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            sys.stdout.write("\r%d/%d batches in %d/%d iteration, current error is %f"                              % (i, len(train_loader), epoch+1, num_epochs, current_loss))
            sys.stdout.flush()
        loss_record.append(current_loss)


# ### save the model parameters

# In[15]:


save_folder = './results/new/'


# In[16]:


loss_list = np.array(loss_list)
dd.io.save(os.path.join(save_folder, loss_name), loss_list)


# In[17]:


print('loss has saved successfully!')


# ### save the model parameters

# set model into eval mode

model.eval()

# In[18]:


torch.save(model.state_dict(),os.path.join(save_folder, model_name))

loss_record = np.array(loss_record)
dd.io.save(os.path.join(save_folder, 'yolo_loss_150epoches_0411.h5'), loss_record)

# In[19]:


print('model has saved successfully!')


# ### time end

# In[24]:


print("\n--- it costs %.4s minutes ---" % ((time.time() - start_time)/60))

