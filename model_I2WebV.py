# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import network
import torch.nn.functional as F
from torch.autograd import Function
import random

source_loss_func = nn.CrossEntropyLoss()
distribute_loss_func = nn.KLDivLoss(reduction='batchmean')


def hloss(x):
    x1 = F.softmax(x, dim=1)
    x2 = F.log_softmax(x, dim=1)
    b = x1 * x2
    b = (-1.0 * b.sum()) / x1.size(0)
    return b

def entropy(x):
    b = x.log()*x
    b = (-1.0 * b.sum()) / x.size(0)
    return b

def cosin_struc_func(x):
    cos_dis = nn.CosineSimilarity(dim=2, eps=1e-6)
    x_1 = x.view(1, x.shape[0], x.shape[1]).expand(x.shape[0], -1, -1)
    x_2 = x.view(x.shape[0], 1, x.shape[1]).expand(-1, x.shape[0], -1)
    return cos_dis(x_1, x_2)

def euclidean_struc_func(x):
    x_1 = x.view(1, x.shape[0], x.shape[1])
    x_2 = x.view(1, x.shape[0], x.shape[1])
    return torch.cdist(x_1, x_2).squeeze()

def product_struc_func(x):
    x_1 = x.view(1, x.shape[0], 1, x.shape[1]).expand(x.shape[0], -1, -1, -1)
    x_2 = x.view(x.shape[0], 1, x.shape[1], 1).expand(-1, x.shape[0], -1, -1)
    return torch.matmul(x_1, x_2)

class GRL(Function):
    def __init__(self, max_entropy_weight):
        self.max_entropy_weight = max_entropy_weight

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        grad_input = grad_output.neg() * 0.1
        return grad_input


class CDZS(nn.Module):
    def __init__(self, embedding_dim, embeddings, args):
        super(CDZS, self).__init__()
        self.CNN = network.ResNet50Fc()

        for param in self.CNN.parameters():
            param.requires_grad = False
        for param in self.CNN.layer4.parameters():
            param.requires_grad = True
        for param in self.CNN.layer3.parameters():
            param.requires_grad = True
        self.embeddings = embeddings
        self.embeddings.requires_grad = False
        self.softmax = nn.Softmax(dim=1)
        self.entropy_weight = args.entropy_weight
        self.count = 0
        self.grl = GRL.apply
        self.args = args
        self.temperature = args.temperature
        self.distribute_weight = args.distribute_weight
        self.fc_w = args.fc_w
        self.mse_loss = nn.MSELoss()
        self.distribute_bound = args.distribute_bound
        self.warehouse = None
        self.pointer = 0
        self.max_warehoure = args.max_warehoure
        self.use_warehouse = args.use_warehouse
        self.structure_type = args.consistency_type
        self.struc_weight = args.consistency_weight
        self.L2_LOSS = torch.nn.MSELoss()
        structure_loss = {'l2': self.l2_loss,
                          'entropy': self.cross_entropy_rank,
                          'pair': self.pairwise_rank_loss,
                          'consistency': self.consistency_loss}
        self.structure_func = euclidean_struc_func
        self.structure_loss = structure_loss[self.structure_type]
        with torch.no_grad():
            if self.args.dataset == 'I2WebV':
                self.struc = self.structure_func(self.embeddings[:5000])
                self.class_num = 5000
            elif self.args.dataset == 'I2AwA':
                self.struc = self.structure_func(self.embeddings[127:177])
                self.class_num = 50
        self.rank = [i for i in range(self.class_num)]

        if self.args.dataset == 'I2WebV':
            class_num = 100
        else:
            class_num = self.class_num
        self.pair1 = torch.zeros(class_num, int(class_num * (class_num + 1) / 2)).to(args.device)
        self.pair2 = torch.zeros(class_num, int(class_num * (class_num + 1) / 2)).to(args.device)
        k = 0

        for i in range(class_num):
            for j in range(class_num - i):
                self.pair1[i, k] = 1
                self.pair2[j, k] = 1
                k += 1
        assert k == int(class_num * (class_num + 1) / 2)


    def bl(self, unseen, c):
        return torch.mean(unseen) + (c**2)/torch.mean(unseen)

    def l2_loss(self, embedding, prototype):
        embedding = embedding / torch.mean(embedding)
        prototype = prototype / torch.mean(prototype)
        return torch.mean((embedding - prototype) ** 2)

    def consistency_loss(self, embedding, prototype):
        return torch.tensor(0, dtype='torch.float32', device=self.args.device)

    def cross_entropy_rank(self, embedding, prototype):
        embedding = embedding / torch.mean(embedding)
        embedding = (embedding.max().clone().detach() - embedding)
        prototype = prototype / torch.mean(prototype)
        prototype = (prototype.max().clone().detach() - prototype)
        p_embedding = F.log_softmax(prototype, dim=1)
        e_embedding = F.softmax(embedding, dim=1)
        loss_func = nn.KLDivLoss(reduction='sum')
        return loss_func(p_embedding, e_embedding)

    def pairwise_rank_loss(self, embedding, prototype):
        num_rank = len(self.rank)
        sample = random.sample(self.rank, num_rank)
        embedding = embedding[sample, sample]
        prototype = prototype[sample, sample]
        embedding1 = torch.matmul(embedding, self.pair1)
        embedding2 = torch.matmul(embedding, self.pair2)
        embedding_dis = embedding1 - embedding2
        prototype1 = torch.matmul(prototype, self.pair1)
        prototype2 = torch.matmul(prototype, self.pair2)
        prototype_dis = prototype1 - prototype2
        loss = torch.mean(torch.relu(prototype_dis[embedding_dis < 0]))
        return loss

    def forward(self, x, y=None, seen_label=None, is_train='True'):
        x = self.CNN(x)
        x = F.normalize(x, p=2.0, dim=1)/self.temperature
        if self.args.dataset == 'I2WebV':
            embeddings = self.Embedding(self.embeddings)[:5000]
        elif self.args.dataset == 'I2AwA':
            embeddings = self.Embedding(self.embeddings)[127:177]
        embeddings = F.normalize(embeddings, p=2.0, dim=1)

        if is_train == 'True':
            p = torch.mm(x, embeddings.T)
            struc_e = self.structure_func(embeddings)
            struc_loss = self.structure_loss(self.struc, struc_e)
            source_loss = source_loss_func(p, y)
            loss = source_loss + self.struc_weight*struc_loss
            return loss, source_loss.item(), struc_loss.item()
        elif is_train == 'Validation':
            p = torch.mm(x, embeddings.T)
            return self.softmax(p)
        elif is_train == 'Target':
            p = torch.mm(x, embeddings.clone().detach().T)
            distribute_loss = torch.tensor(0)
            distribute_loss1 = torch.tensor(0)

            if self.warehouse is None:
                self.warehouse = x.detach().clone()
            elif self.warehouse.shape[0] < self.max_warehoure:
                self.warehouse = torch.cat((self.warehouse, x.detach().clone()), dim=0)
            else:
                self.warehouse[self.pointer:self.pointer+x.shape[0],:]=x.detach().clone()
                self.pointer += x.shape[0]
                if self.pointer+x.shape[0] > self.max_warehoure:
                    self.pointer = 0
                p_2 = torch.mm(self.warehouse, embeddings.T)
                p_1 = self.softmax(p_2)
                p_1 = torch.mean(p_1, dim=0, keepdim=True)
                distribute_loss = -1*min(entropy(p_1), self.distribute_bound)
                p_2 = torch.mm(x, embeddings.T)

                p_3 = torch.sum(self.softmax(p_2)[:, seen_label], dim=1)
                if self.args.dataset == 'I2WebV':
                    distribute_loss1 = self.bl(p_3, 0.82)
                elif self.args.dataset == 'I2AwA':
                    distribute_loss1 = self.bl(p_3, self.args.bias_weight)






            return self.entropy_weight * hloss(p) + self.distribute_weight * distribute_loss + self.args.fc_w*distribute_loss1, hloss(p).item(), distribute_loss.item(), distribute_loss1.item()
