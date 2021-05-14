# -*- coding: utf-8 -*-
"""MNIST_Tanh_Attacks.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ysG7TBoyl1pQ5CIehFMrQvM3Ccam7ML7
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import datasets, transforms

epsilons = [0, .05, .1, .15, .2, .25, .3]
#epsilons = [0, .01, .05, .1]
use_cuda=False

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.tanh(F.max_pool2d(self.conv1(x), 2))
        x = torch.tanh(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.tanh(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('/content/drive/MyDrive/CS726/', train=False, download=True, transform=transforms.Compose([
#            transforms.ToTensor(),
#            ])),
#        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
# model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
#model.eval()

# Load the datasets
trainset = datasets.MNIST(root='/content/drive/MyDrive/CS726/mnist_train', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 16, shuffle = True)

testset = datasets.MNIST(root='/content/drive/MyDrive/CS726/mnist_test', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()]))
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = True)

# Set the optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device),data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

# Save the model
PATH = '/content/drive/MyDrive/CS726/lenet_mnist_tanh.pth'
torch.save(model.state_dict(), PATH)

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (100 * correct / total))

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    #print(image.shape)
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, testloader, eps)
    accuracies.append(acc)
    examples.append(ex)

def rfgsmtest( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    counter = 0
    # Loop over all examples in test set
    for data, target in test_loader:   
        # Set Alpha
        alpha = epsilon/2 

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Perform Random Step
        adv_images = data + alpha*torch.randn_like(data).sign()
        data = torch.clamp(adv_images, min=0, max=1)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack with epsilon-alpha
        perturbed_data = fgsm_attack(data, epsilon-alpha, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = rfgsmtest(model, device, testloader, eps)
    accuracies.append(acc)
    examples.append(ex)

def mifgsmtest( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    counter = 0
    # Loop over all examples in test set
    for data, target in test_loader:
        #print(data)
        # Check for 10000 images
        counter += 1
        if counter == 10000:
          break
        
        # Set the parameters
        steps = 2
        alpha = epsilon/steps
        mu = 1

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        
        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            steps = 0
        g = torch.zeros_like(data).to(device)        
        for step in range(steps):
          # Send the data and label to the device
          data, target = data.to(device), target.to(device)

          # Set requires_grad attribute of tensor. Important for Attack
          data.requires_grad = True
          
          # Forward pass the data through the model
          output = model(data)
          
          # Calculate the loss
          loss = F.nll_loss(output, target)

          # Zero all existing gradients
          model.zero_grad()

          # Calculate gradients of model in backward pass
          loss.backward()

          # Collect datagrad
          data_grad = data.grad.data
          data.requires_grad = False
          
          # Update gradients
          grad_norm = torch.norm(nn.Flatten()(data_grad), p=1, dim=1)
          if step == 0:
            g = data_grad/grad_norm
          else:
            g = mu*g + data_grad/grad_norm

          # Create Adversarial Image
          sign_data_grad = g.sign()
          data = data + alpha*sign_data_grad
          data = torch.clamp(data, 0, 1)

          # Re-classify the perturbed image
          output = model(data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    #final_acc = correct/float(len(test_loader))
    final_acc = correct/float(10000)
    #print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, 10000, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# Run test for each epsilon
for eps in epsilons:
    acc, ex = mifgsmtest(model, device, testloader, eps)
    accuracies.append(acc)
    examples.append(ex)

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()