import torch
from tqdm.notebook import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, test_loader, loss_function, optimizer, num_epochs, lr_scheduler=None):
    
#     def loss_fun(output, target):
#         return F.cross_entropy(output, target)
    
    out_dict = {'train_acc': [],
                'test_acc': [],
                'train_loss': [],
                'test_loss': []}
  
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        train_correct = 0
        train_len = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            output = model(data)[:,0]
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            train_loss.append(loss.item())
            predicted = output > 0.5
            train_correct += (target==predicted).sum().cpu().item()
            train_len += data.shape[0]
            
        test_loss = []
        test_correct = 0
        test_len = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.type(torch.FloatTensor).to(device)
            with torch.no_grad():
                output = model(data)[:,0]
            test_loss.append(loss_function(output, target).cpu().item())
            predicted = output > 0.5
            test_correct += (target==predicted).sum().cpu().item()
            test_len += data.shape[0]
            
        out_dict['train_acc'].append(train_correct/train_len)
        out_dict['test_acc'].append(test_correct/test_len)
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
        
    return out_dict


def train_transfer(model, train_loader, test_loader, loss_function, optimizer, num_epochs, model_name, lr_scheduler=None, save_model=False ):
    
#     def loss_fun(output, target):
#         return F.cross_entropy(output, target)
    
    out_dict = {'train_acc': [],
                'test_acc': [],
                'train_loss': [],
                'test_loss': []}
  
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        train_correct = 0
        train_len = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(torch.float32).to(device)
            optimizer.zero_grad()
            output = model(data)[:,0]
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            train_loss.append(loss.item())
            predicted = output > 0.5
            train_correct += (target==predicted).sum().cpu().item()
            train_len += data.shape[0]
            
        test_loss = []
        test_correct = 0
        test_len = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(torch.float32).to(device)
            with torch.no_grad():
                output = model(data)[:,0]
            test_loss.append(loss_function(output, target).cpu().item())
            predicted = output > 0.5
            test_correct += (target==predicted).sum().cpu().item()
            test_len += data.shape[0]

        if save_model and epoch > 0 and test_correct/test_len > max(out_dict['test_acc']):
            torch.save(model, 'models/' + model_name)
            
            
        out_dict['train_acc'].append(train_correct/train_len)
        out_dict['test_acc'].append(test_correct/test_len)
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))

        
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
        
    return out_dict