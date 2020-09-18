import pickle
import time
import numpy as np
from Models import FeatExplainer, LatentNet
import torch

model = LatentNet(3).double()
model.load_state_dict(torch.load('Saved_Models/mod_0.txt'))
with open('Saved_Models/data.pickle', 'rb') as handle:
    arr = pickle.load(handle)
x_train, y_train, x_val, y_val = arr[0], arr[1], arr[2], arr[3]
model.eval()
explainer = FeatExplainer(
    x=x_train,
    model=model,
    label=y_train
)
explainer.train()
begin_time = time.time()
for epoch in range(100):
    explainer.zero_grad()
    explainer.optimizer.zero_grad()
    y_pred = explainer()
    loss = explainer.loss(y_pred, torch.argmax(model(x_train), dim=1), y_train)
    loss.backward()

    explainer.optimizer.step()
    if explainer.scheduler is not None:
        explainer.scheduler.step()

    # output_val = model(x_val)
    # y_hat_val = torch.argmax(output_val, dim=1)
    # val_correct = 0

    # for i in range(len(y_val)):
    #     if y_hat_val[i] == y_val[i]:
    #         val_correct += 1

    print(
        "epoch: ",
        epoch,
        "; loss: ",
        loss
        # "; acc: ",
        # val_correct / len(y_val),
    )

print(torch.where(explainer.feat_mask_c0<0.1),torch.where(explainer.feat_mask_c0<0.1)[0].size())
print(torch.where(explainer.feat_mask_c0>0.95),torch.where(explainer.feat_mask_c0>0.95)[0].size())
print(torch.where(explainer.feat_mask_c1<0.1),torch.where(explainer.feat_mask_c1<0.1)[0].size())
print(torch.where(explainer.feat_mask_c1>0.95),torch.where(explainer.feat_mask_c1>0.95)[0].size())
print(torch.where(explainer.feat_mask_c2<0.1),torch.where(explainer.feat_mask_c2<0.1)[0].size())
print(torch.where(explainer.feat_mask_c2>0.95),torch.where(explainer.feat_mask_c2>0.95)[0].size())
