import pickle
import time

from Models import LatentNet
from Utils import load_data,accuracy
import torch.nn.functional as F

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

# with open('train_data.pickle', 'rb') as f:
#     X_, y_, train_mask_, test_mask_, weight_ = pickle.load(f)  # Load the data

feats, adj_mat, labels, idx_train, idx_val, labels_train, labels_test = load_data(True)
# features, adj, labels, idx_train, idx_val, labels_train, labels_test = load_data2()
epochs = 1000
fast_mode = True
patience = 10000

model = LatentNet(num_classes=int(labels.max())+1)
# optimizer = optim.Adam(model.parameters(),lr=1,weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

lr_scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=50, verbose=True)

features, adj, labels = Variable(torch.tensor(feats)), Variable(adj_mat), Variable(torch.tensor(labels))
model.double()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

cg = {'train': idx_train,
      'validation': idx_val,
      'x': feats,
      'adj': adj_mat
      }

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fast_mode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    lr_scheduler.step(loss_train)

    return acc_val


# Train model
t_total = time.time()
acc_values = []
bad_counter = 0
best = 0
best_epoch = 0
best_model_state_dict = None

for epoch in range(epochs):
    acc_values.append(train(epoch))

    if acc_values[-1] > best:
        best_model_state_dict = model.state_dict()
        best = acc_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == patience:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))




#
# # number of epochs to train the model
# n_epochs = 1000
#
# best_acc = 0
# best_model = None
# best_data = None
#
# for fold in range(3,10):
#     print('\n\n\n-----------fold {} ------------\n'.format(fold))
#
#     # initialize the NN
#     # model = LatentNet(3).double()
#     model = LatentNet(3).double()
#     print(model)
#
#     # specify loss function
#     criterion = nn.CrossEntropyLoss()
#
#     # specify optimizer
#     optimizer = torch.optim.SGD(model.parameters(), lr=1)
#
#     lr_scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=50, verbose=True)
#
#     model.train()  # prep model for training
#
#     X_Train_ = []
#     y_train_ = []
#     X_Val_ = []
#     y_val_ = []
#     for i in range(len(train_mask_[:, fold])):
#         if train_mask_[i, fold] == 1:
#             X_Train_.append(X_[i, :, fold])
#             y_train_.append(y_[i, :, fold])
#         else:
#             X_Val_.append(X_[i, :, fold])
#             y_val_.append(y_[i, :, fold])
#     X_Train = torch.tensor(np.array(X_Train_)).double()
#     y_train = torch.tensor(np.argmax(np.array(y_train_), axis=1)).long()
#     X_Val = torch.tensor(np.array(X_Val_)).double()
#     y_val = torch.tensor(np.argmax(np.array(y_val_), axis=1)).long()
#
#     train_mask = torch.tensor(train_mask_[:, fold] / np.mean(train_mask_[:, fold]))
#     test_mask = torch.tensor(test_mask_[:, fold] / np.mean(test_mask_[:, fold]))
#     weight = np.squeeze(weight_[:, fold])
#     train_mask = train_mask * weight
#     test_mask = test_mask * weight
#
#     # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('Start Training')
#
#     for epoch in range(n_epochs):
#         # monitor training loss
#         ###################
#         # train the model #
#         ###################
#         # clear the gradients of all optimized variables
#         optimizer.zero_grad()
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output_train = model(X_Train)
#         y_hat_train = torch.argmax(output_train, dim=1)
#         #     print('#####\n', output, '\n######\n')
#         #     print(y_hat)
#
#         train_correct = 0
#
#         for i in range(len(y_train)):
#             if y_hat_train[i] == y_train[i]:
#                 train_correct += 1
#             # elif test_mask_[i, fold] == 1 and y_hat[i] == y[i]:
#             #     val_correct += 1
#
#         loss = criterion(output_train, y_train)
#         train_loss = torch.mean(loss * train_mask)
#         # backward pass: compute gradient of the loss with respect to model parameters
#         train_loss.backward()
#         # perform a single optimization step (parameter update)
#         optimizer.step()
#
#         output_val = model(X_Val)
#         y_hat_val = torch.argmax(output_val, dim=1)
#         if epoch % 100 == 0:
#             print(y_hat_val)
#
#         val_correct = 0
#
#         for i in range(len(y_val)):
#             if y_hat_val[i] == y_val[i]:
#                 val_correct += 1
#
#         if best_acc < val_correct / np.sum(test_mask_[:, fold]):
#             best_model = model.state_dict()
#             best_acc = val_correct / np.sum(test_mask_[:, fold])
#             best_data = [X_Train, y_train, X_Val, y_val, train_mask, test_mask]
#
#         loss = criterion(output_val, y_val)
#         validation_loss = torch.mean(loss * test_mask)
#
#         lr_scheduler.step(train_loss)
#         # update running training loss
#
#         # print training statistics
#         # calculate average loss over an epoch
#         print(
#             'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation '
#             'Accuracy: {:.6f}'.format(
#                 epoch + 1,
#                 train_loss,
#                 validation_loss,
#                 train_correct / np.sum(train_mask_[:, fold]),
#                 val_correct / np.sum(test_mask_[:, fold])
#             ))
#
#     print('Best model has ' + str(best_acc) + ' accuracy')
#
# print('Final Best model has ' + str(best_acc) + ' accuracy')
# torch.save(best_model, 'Saved_Models/mod_0.txt')
# with open('Saved_Models/data.pickle', 'wb') as handle:
#     pickle.dump(best_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
