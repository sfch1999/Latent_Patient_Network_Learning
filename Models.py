import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEBUG = False


class LatentNet(nn.Module):
    def __init__(self, num_classes):
        super(LatentNet, self).__init__()
        self.fc1 = nn.Linear(342, 128)
        # self.bn1=nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(128, 64)
        self.temp = torch.empty(1)
        torch.nn.init.trunc_normal_(self.temp, std=1e-1)
        self.temp = nn.Parameter(self.temp[0] + 1)
        self.theta = torch.empty(1)
        torch.nn.init.trunc_normal_(self.theta, std=1e-1)
        self.theta = nn.Parameter(self.theta[0] + 1)
        self.sig = nn.Sigmoid()
        self.fc3 = nn.Linear(342, 32)
        # self.fc4 = nn.Linear(32, 16)
        # self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(32, num_classes)
        self.sm = nn.Softmax(dim=1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)
        # torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        net = self.fc1(x)

        if DEBUG:
            print('fc1', net)

        net = self.fc2(net)

        if DEBUG:
            print('fc2', net)

        A = -1 * self.pairwise_distances(net)

        if DEBUG:
            print(A)
            print('bef sig', self.temp * A + self.theta)

        A = self.sig(self.temp * A + self.theta)

        if DEBUG:
            print(A)

        A = A * (1 - torch.eye(A.shape[-1])) + torch.eye(A.shape[-1])

        if DEBUG:
            print(A)

        A = torch.true_divide(A, torch.sum(A, dim=1)[:, None])

        if DEBUG:
            print(A)

        x = A.matmul(x)

        if DEBUG:
            print('after matmul', x)
        x = self.fc3(x)
        # x = A.matmul(x)
        # x = self.fc4(x)
        # x = self.fc5(x)
        x = F.log_softmax(self.fc6(x))
        # x = self.sm(self.fc6(x))

        return x

    @staticmethod
    def pairwise_distances(x, y=None):
        """
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        return torch.clamp(dist, 0.0, np.inf)



class FeatExplainer(nn.Module):
    def __init__(
            self,
            x,
            model,
            label,
            use_sigmoid=True,
    ):
        super(FeatExplainer, self).__init__()
        self.x = x
        self.model = model
        self.label = label
        self.use_sigmoid = use_sigmoid

        self.feat_mask_c0 = self.construct_feat_mask(x.size()[1], init_strategy="constant")
        self.feat_mask_c1 = self.construct_feat_mask(x.size()[1], init_strategy="constant")
        self.feat_mask_c2 = self.construct_feat_mask(x.size()[1], init_strategy="constant")

        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, [self.feat_mask_c0, self.feat_mask_c1, self.feat_mask_c2]), lr=0.1,
            momentum=0.95)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.8)
        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    @staticmethod
    def construct_feat_mask(feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def forward(self, marginalize=False):
        x = self.x
        tot_mask = self.build_tot_feat_mask()

        feat_mask = (
            torch.sigmoid(tot_mask)
            if self.use_sigmoid
            else tot_mask
        )
        if marginalize:
            std_tensor = torch.ones_like(x, dtype=torch.float) / 2
            mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
            z = torch.normal(mean=mean_tensor, std=std_tensor)
            x = x + z * (1 - feat_mask)
        else:
            x = x * feat_mask

        ypred = self.model(x)
        return ypred

    def loss(self, pred, pred_label, gt_label):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        tot_mask = self.build_tot_feat_mask()

        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            gt_label_node = gt_label  # TODO By Soroosh
            logit = pred[np.arange(self.x.size()[0]), gt_label_node]
            pred_loss = torch.sum(-torch.log(logit))
        # size
        feat_mask = (
            torch.sigmoid(tot_mask) if self.use_sigmoid else tot_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.sum(torch.mean(feat_mask))
        # print(feat_mask)
        # entropy
        feat_mask_ent = - feat_mask \
                        * torch.log(feat_mask) \
                        - (1 - feat_mask) \
                        * torch.log(1 - feat_mask)

        # feat_mask_ent[feat_mask_ent.isnan()]=0
        # feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.sum(torch.mean(feat_mask_ent))

        loss = pred_loss + feat_size_loss
        # print(feat_mask_ent.isnan().any())
        # print(feat_mask_ent.isnan().any())
        # print(feat_mask_ent)
        return loss

    def build_tot_feat_mask(self):
        tot_mask = torch.FloatTensor(self.x.size())
        for i in range(len(self.label)):
            if self.label[i] == 0:
                tot_mask[i, :] = self.feat_mask_c0
            if self.label[i] == 1:
                tot_mask[i, :] = self.feat_mask_c1
            if self.label[i] == 2:
                tot_mask[i, :] = self.feat_mask_c2
        return tot_mask
