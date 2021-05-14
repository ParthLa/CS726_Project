import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


args = {
    'train_batch_size' : 128,  # input batch size for training (default: 128)
    'test_batch_size'  : 128,  # input batch size for testing (default: 128)
    'epochs' : 50,             # number of epochs to train  
    
    # 'weight_decay' : 2e-4,
    'lr' : 0.01,                # learning rate
    'momentum' : 0.9,          # SGD momentum
    
    'epsilon' : 0.03,         # perturbation
    'num_steps' : 1,          # perturb number of steps
    'step_size' : 0.01,       # perturb step size

    'beta'   : 1.0,            # regularization (1/lambda) in TRADES
    'seed'   : 1,              # random seed
    'log_interval' : 100,      # how many batches to wait before logging training status

    'save_freq' : 5,           # save frequency
    'model_dir' : './AML'         # directory of model for saving checkpoint

}
kwargs = {'num_workers' : 1, 'pin_memory' : True} if device.type=='cuda' else {}
kwargs

## Set up data set

import torchvision
from torchvision import datasets, transforms

trainset = datasets.MNIST(root=args['model_dir'], train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST(root=args['model_dir'], train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(trainset, 
                                           batch_size = args['train_batch_size'],
                                           shuffle = True, **kwargs)

test_loader  = torch.utils.data.DataLoader(testset, 
                                           batch_size = args['test_batch_size'],
                                           shuffle = False, **kwargs)

## Model

from collections import OrderedDict

class SmallCNN(nn.Module):
    def __init__(self, drop=0.5):
        super(SmallCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


## Modified TRADES loss

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                prev_delta,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'
                ):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + (torch.tile(prev_delta,(x_natural.shape[0],1,1,1))).detach()
    if distance == 'l_inf':
        #for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        new_delta = x_adv - x_natural
        new_delta = torch.sum(new_delta,axis=0)/(new_delta.shape[0]) ## taking average of perturbations
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(mmodeodel(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, new_delta

## Model Training 

def adjust_learning_rate(optimizer, epoch, old_lr):
    lr = old_lr
    if epoch >= 55:
        lr = 75 * 0.1
    if epoch >= 90:
        lr = old_lr * 0.01
    if epoch >= 90:
        lr = old_lr * 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, model, device, data_loader, optimizer, epoch):
    model.train()
    delta = None
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        if delta==None:
          delta = 0.001 * torch.randn(data.shape[1:]).cuda()
          # print(data.shape)
          # print(delta.shape)

        loss, delta = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           prev_delta = delta,
                           step_size=args["step_size"],
                           epsilon=args["epsilon"],
                           perturb_steps=args["num_steps"],
                           beta=args["beta"],
			                     distance='l_inf'
                           )
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))


def evaluate(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0

    with torch.no_grad():

        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print('Evaluation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, len(data_loader.dataset), 100.0 * accuracy))
     
    return loss, accuracy


def main(model, args):
    optimizer = optim.SGD(model.parameters(),
                          lr=args['lr'], momentum=args['momentum'],
                          # weight_decay=args['weight_decay']
                          )
    model_dir = args['model_dir']
    for epoch in range(1, args['epochs'] + 1):
        # adjust learning rate for torch.save(model.state_dict(), PATH)SGD
        adjust_learning_rate(optimizer, epoch, args['lr'])

        # training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        evaluate(model, device, train_loader)
        evaluate(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-SmallCNN-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-SmallCNN-checkpoint_epoch{}.tar'.format(epoch)))



## Model creation and training

model = SmallCNN().to(device)

main(model,args)

## Attack

def attack(model, device, model_path, test_loader, is_random_initial, args, num_steps):
    torch.manual_seed(args['seed'])  
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(model_path))

    epsilon   = args['epsilon']
    # num_steps = args['num_steps']
    step_size = args['step_size']

    natural_err_total = 0
    robust_err_total = 0
    total_samples = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
          data, target = data.to(device), target.to(device)
          # print(data.shape)
          # PGD attack
          X, y = Variable(data, requires_grad=True), Variable(target)

          out = model(X)
          err_natural = (out.data.max(1)[1] != y.data).float().sum()

          X_attack = Variable(X.data, requires_grad=True)
          if is_random_initial:
            random_noise = torch.FloatTensor(*X_attack.shape).uniform_(-epsilon, epsilon).to(device)
            X_attack = Variable(X_attack.data + random_noise, requires_grad=True)
          
          for _ in range(num_steps):
                opt = optim.SGD([X_attack], lr=1e-3)
                opt.zero_grad()

                with torch.enable_grad():
                  loss = nn.CrossEntropyLoss()(model(X_attack), y)
                loss.backward()

                eta = step_size * X_attack.grad.data.sign()
                X_attack = Variable(X_attack.data + eta, requires_grad=True)
                eta = torch.clamp(X_attack.data - X.data, -epsilon, epsilon)
                X_attack = Variable(X.data + eta, requires_grad=True)
                X_attack = Variable(torch.clamp(X_attack, 0, 1.0), requires_grad=True)
          
          err_robust = (model(X_attack).data.max(1)[1]!=y.data).float().sum()
          total_samples+=data.shape[0]
          natural_err_total += err_natural
          robust_err_total += err_robust
          # print progress
          print('BatchID: {}, natural_err: {:.3f}, roubust_err: {:.3f}'.format(
                    batch_idx, err_natural, err_robust))
    print('natural_accuracy: {:.3f}, roubust_accuracy: {:.3f}'.format(100.0*(1-(natural_err_total/total_samples)),100.0*(1-(robust_err_total/total_samples))))
    return natural_err_total, robust_err_total

attack(SmallCNN().to(device),device,'./AML/model-SmallCNN-epoch50.pt',test_loader,True,args,40) ## FGSM-40 attack
