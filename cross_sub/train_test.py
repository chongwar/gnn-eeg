import torch
import numpy as np


def train(model, criterion, optimizer, data_loader, device, train_num, epochs, logged=False):
    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        num_correct = 0
        batch_size = None
        
        for index, data in enumerate(data_loader):
            data = data.to(device)
            y = torch.from_numpy(np.asarray(data.y)).float()
            y = y.to(device)
            batch_size = y.shape[0] if index == 0 else batch_size

            y_pred = model(data)
            _, pred = torch.max(y_pred, 1)
            # print(pred, y)
            num_correct += (pred == y).sum()

            loss = criterion(y_pred, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        batch_num = train_num // batch_size
        _loss = running_loss / (batch_num + 1)
        acc = num_correct.item() / train_num * 100
        
        if logged:
            print(f'Epoch {epoch + 1}/{epochs}\tTrain loss: {_loss:.4f}\t'
                  f'Train acc: {acc:.2f}%')

    # path = f'checkpoint/{model.__class__.__name__}_{epochs}.pth'
    # torch.save(model.state_dict(), path)
    print('Finish training!')


def test(model, criterion, data_loader, device, test_num, log, logged=False):
    model.eval()
    
    running_loss = 0.0
    num_correct = 0
    batch_size = None
    
    for index, data in enumerate(data_loader):
        data = data.to(device)
        y = torch.from_numpy(np.asarray(data.y)).float()
        y = y.to(device)
        batch_size = y.shape[0] if index == 0 else batch_size

        y_pred = model(data)
        _, pred = torch.max(y_pred, 1)
        num_correct += (pred == y).sum()

        loss = criterion(y_pred, y.long())
        running_loss += float(loss.item())

    batch_num = test_num // batch_size
    _loss = running_loss / (batch_num + 1)
    acc = num_correct.item() / test_num * 100
    print(f'Test loss: {_loss:.4f}\tTest acc: {acc:.2f}%')

    if logged:
        log.append(f'{acc:.2f}\t\n')
        with open('../result/gnn_20201208_cross_sub.txt', 'a') as f:
            f.writelines(log)
