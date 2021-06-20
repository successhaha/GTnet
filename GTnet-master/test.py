import collections
import pickle
import random
import numpy as np
from sklearn.metrics.pairwise import distance_metrics
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA
from tqdm.notebook import tqdm

use_gpu = torch.cuda.is_available()


# ========================================
#      loading datas

def scaleEachUnitaryDatas(datas):

    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms


def QRreduction(datas):
    ndatas = torch.qr(datas.permute(0, 2, 1)).R
    ndatas = ndatas.permute(0, 2, 1)

    return ndatas


def graph_embedding(datas,k,kappa,alpha):
    n, m, z = datas.shape
    F1 = torch.zeros(n, m, m)
    ndatas_qr = torch.qr(datas.permute(0, 2, 1)).R
    ndatas_qr = ndatas_qr.permute(0, 2, 1)
    ndatas_qr.cuda()
    for i in range(n):
        metric = distance_metrics()['cosine']
        S = 1 - metric(ndatas_qr[i, :, :], ndatas_qr[i, :, :])
        S = torch.tensor(S)
        S = S - torch.eye(S.shape[0])

        if k>0:
            topk, indices = torch.topk(S, k)
            mask = torch.zeros_like(S)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask+torch.t(mask))>0).type(torch.float32)
            S    = S*mask

        D       = S.sum(0)
        Dnorm   = torch.diag(torch.pow(D, -0.5))
        E   = torch.matmul(Dnorm, torch.matmul(S, Dnorm))
        E = alpha * torch.eye(E.shape[0]) + E
        E = torch.matrix_power(E, kappa)
        E = E.cuda()
        print("___________________feature", E.shape, ndatas_qr[i, :, :].shape)
        F1[i, :, :] = E
    F1 = torch.nn.functional.softmax(F1, dim=2)
    return F1

def ET(datas):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]
    return datas

class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              

class distribution_Gaussian(Model):
    def __init__(self, n_ways, lam):
        super(distribution_Gaussian, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam

    def clone(self):
        other = distribution_Gaussian(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self):
        self.mus = ndatas.reshape(n_runs, n_shot+n_queries, n_ways, n_nfeat)[:, :n_shot, ].mean(1)
    def updateFromEstimate(self, estimate, alpha):   
        
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)
    def OPT(self, M, r, c, epsilon=1e-6):

        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)

    
    def getProbas(self):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * n_queries
        p_xj_test, _ = self.OPT(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test
        p_xj[:, :n_lsamples].fill_(0)
        p_xj[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)
        return p_xj

    def estimateFromMask(self, mask):

        emus = mask.permute(0, 2, 1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))
        return emus

          
# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):
        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
    
    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:, n_lsamples:].mean(1)

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, epochInfo=None):
     
        p_xj = model.getProbas()
        self.probas = p_xj
        
        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))
        
        m_estimates = model.estimateFromMask(self.probas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, n_epochs=20):
        
        self.probas = model.getProbas()
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f}  lr_m: {:0.3f}".format(epoch, self.alpha))
            self.performEpoch(model, epochInfo=(epoch, n_epochs))
            if (self.progressBar): pb.update()
        
        # get final accuracy and return it
        op_xj = model.getProbas()
        acc = self.getAccuracy(op_xj)
        return acc
    

if __name__ == '__main__':
    n_shot = 5
    n_ways = 5
    n_queries = 15
    n_runs = 10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    k=4
    kappa = 1
    alpha = 0.75
    import FSLTask
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries}
    FSLTask.loadDataSet("dataset")
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot+n_queries, 5).clone().view(n_runs, n_samples)
    beta = 0.5
    ndatas[:, ] = torch.pow(ndatas[:, ]+1e-6, beta)
    ndatas = graph_embedding(ndatas, k=k, kappa=kappa, alpha=alpha)
    n_nfeat = ndatas.size(2)

    ndatas = scaleEachUnitaryDatas(ndatas)  #

    ndatas = ET(ndatas)
    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()

    lam = 10
    model =distribution_Gaussian(n_ways, lam)
    model.initFromLabelledDatas()
    
    alpha = 0.2
    optim = MAP(alpha)

    optim.verbose=False
    optim.progressBar=True

    acc_test = optim.loop(model, n_epochs=20)
    
    print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100*x for x in acc_test)))
    
    

