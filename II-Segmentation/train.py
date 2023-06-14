import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import time
from IPython.display import clear_output
from loss import IoU

def train(model, opt, loss_fn, epochs, train_loader, test_loader, device):
    X_test, Y_test = next(iter(test_loader))
    out_dict = {'train_acc': [],
                'test_acc': [],
                'train_loss': [],
                'test_loss': []}
    
    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        train_loss = []
        train_acc = []
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            print(f"Y_pred shape: {Y_pred.shape}")
            print(f"Y_batch shape: {Y_batch.shape}")
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            train_loss.append(loss.item())
            train_acc.append(IoU(Y_batch, Y_pred).cpu().numpy())
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        toc = time()
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.show()
        out_dict['train_acc'].append(np.mean(train_acc))
        #out_dict['test_acc'].append(test_correct/test_len)
        out_dict['train_loss'].append(np.mean(train_loss))
        #out_dict['test_loss'].append(np.mean(test_loss))

    return out_dict

def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [F.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(Y_pred)