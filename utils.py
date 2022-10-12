import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import time
from copy import deepcopy


def class_weight(target):
    class_sample_count = np.array([len(np.where(target.values == t)[0]) for t in np.unique(target.values)])

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target.values])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler, weight


def create_dataloader(X_raw, Y_raw, target_name, batch_size=16, test_size=0.2, shuffle=True):
    xtrain, xtest, ytrain, ytest = train_test_split(X_raw, Y_raw[target_name], test_size=test_size, shuffle=shuffle)

    train_in = torch.Tensor(xtrain)
    train_out = torch.tensor(ytrain.to_numpy()).type(torch.LongTensor)
    train_data = TensorDataset(train_in, train_out)

    test_in = torch.Tensor(xtest)
    test_out = torch.tensor(ytest.to_numpy()).type(torch.LongTensor)
    test_data = TensorDataset(test_in, test_out)

    sampler, weight = class_weight(ytrain)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=1)

    return train_loader, test_loader, weight


def train_model(model, criterion, optimizer, dataloader, test_dataloader, scheduler, label_name, num_epochs=25,
                device="cpu", mode='STD_RemoveMid_120_5lag_current', verbose=True):
    since = time.time()

    data_key = ['train', 'val']
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    history = {'acc': [], 'loss': []}

    for epoch in range(num_epochs):
        if verbose:
            print('-' * 10)
            print('Epoch {}|{}'.format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in data_key:
            if phase != 'val':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            label = []
            result = []

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase != 'val'):
                    outputs = model(inputs).to(device)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase != 'val':
                        loss.backward()
                        optimizer.step()

                # batch size * Mean(Loss)
                running_loss += loss.item() * inputs.size(0)
                label += labels.detach().cpu().numpy().tolist()
                result += preds.detach().cpu().numpy().tolist()
                running_corrects += torch.sum(preds == labels.data)
            if phase != 'val':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            epoch_f1 = f1_score(label, result, average='macro')

            history['acc'].append(epoch_acc.detach().cpu().numpy())
            history['loss'].append(epoch_loss)

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_f1))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = deepcopy(model.state_dict())
            if phase == 'val' and epoch_f1 > best_f1:
                best_acc = epoch_acc
                best_f1 = epoch_f1
                best_model_wts = deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0:
            _ = testing(model, test_dataloader, cls_name=label_name, device=device, verbose=verbose)
            torch.save(model, 'checkpoints/' + mode + '_' + str(epoch) + '.pkl')

    time_elapsed = time.time() - since
    print('-' * 10)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Train Acc: {:4f}'.format(best_acc))
    print('Best Train f1: {:4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, 'checkpoints/' + mode + '_' + 'best2.pkl')
    _, _, cls_report = testing(model, test_dataloader, cls_name=label_name, device=device)
    return model, cls_report, history


def testing(model, test_dataloader, cls_name, device='cuda', verbose=True):
    model.eval()

    results = []
    target = []
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.detach().cpu().numpy()
            results.append(preds)
            target.append(labels.detach().cpu().numpy())
    results = np.vstack(results).reshape(-1)
    target = np.vstack(target).reshape(-1)
    if verbose:
        print(classification_report(target, results, target_names=cls_name))

    return results, target, classification_report(target, results, target_names=cls_name)
