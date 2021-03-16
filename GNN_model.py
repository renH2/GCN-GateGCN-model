import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import utils_load as ut

'''
MLP
GCN
'''

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, useDropout=True, keepProb=0.5, useBatchNorm=True,
                 activateFunc="relu"):
        '''
        :param num_layers: 网络层数
        :param input_dim: 输入维数
        :param hidden_dim: 隐藏层维数
        :param output_dim: 输出维数
        :param useDropout:是否使用dropout
        :param keepProb: dropout率
        :param useBatchNorm:是否使用batch_normalized
        :param activateFunc:激活函数（支持relu和tanh）
        '''
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        self.useDropout = useDropout
        self.keepProb = keepProb
        self.useBatchNorm = useBatchNorm
        self.activateFunc = activateFunc
        self.num_layers = num_layers
        # 是否使用batch_normalized
        if self.useBatchNorm:
            self.bns = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("层数最少为 1！")
        # 若只有一层，则维数为input_dim x output_dim
        if num_layers == 1:
            self.linears.append(nn.Linear(input_dim, output_dim))
        # 若有多层，则第一层维数为input_dim x hidden_dim，中间层为hidden_dim x hidden_dim,最后一层维数为hidden_dim x hidden_dim
        # 如果使用批处理，则除了第一层，均在线性层后使用batch_normalized
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            if useBatchNorm:
                for layer in range(num_layers - 1):
                    self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, input):
        output = input
        length = len(self.linears)
        for i in range(length):
            output = self.linears[i](output)
            if i != length - 1:
                if self.useBatchNorm:
                    output = self.bns[i](output)
                if self.activateFunc == "relu":
                    output = F.relu(output)
                elif self.activateFunc == "tanh":
                    output = F.tanh(output)
                if self.useDropout:
                    output = F.dropout(output, p=self.keepProb, training=self.training)
        return output


class GCN_Layer(nn.Module):
    def __init__(self, A, input_dim, output_dim):
        '''
        :param A:邻接矩阵
        :param input_dim:输入维数
        :param output_dim:输出维数
        '''
        super(GCN_Layer, self).__init__()
        # GCN公式 H(l+1)=activate(D^-0.5 x (A+I) x D ^-0.5 x H(l) x W)
        self.A_hat = A + torch.eye(A.size(0))
        self.D = torch.diag(torch.sum(A, 1))
        self.D = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        self.W = nn.Parameter(torch.rand(input_dim, output_dim, requires_grad=True))
        init.kaiming_uniform_(self.W)

    def forward(self, input):
        output = torch.relu(torch.mm(torch.mm(self.A_hat, input), self.W))
        return output


# 两层GCN模型
class GCN_Module(nn.Module):
    def __init__(self, A, nfeat, nhid, nout):
        super(GCN_Module, self).__init__()
        self.conv1 = GCN_Layer(A, nfeat, nhid)
        self.conv2 = GCN_Layer(A, nhid, nout)

    def forward(self, X):
        H = self.conv1(X)
        H2 = self.conv2(H)
        return H2


# 多层GCN模型
class GCN_Multi(nn.Module):
    def __init__(self, A, nfeat, nhid, nout, num_layers=2):
        '''
        :param A: 邻接矩阵
        :param nfeat: 输入feature的维数
        :param nhid: 隐藏层维数
        :param nout: 输出层维数
        :param num_layers: GCN层数
        '''
        super(GCN_Multi, self).__init__()
        self.mul = nn.ModuleList()
        self.num_layers = num_layers
        if num_layers == 1:
            raise ValueError("GCN最少为两层!")
        if num_layers == 2:
            self.mul.append(GCN_Layer(A, nfeat, nhid))
            self.mul.append(GCN_Layer(A, nhid, nout))
        else:
            self.mul.append(GCN_Layer(A, nfeat, nhid))
            for i in range(num_layers - 2):
                self.mul.append(GCN_Layer(A, nhid, nhid))
            self.mul.append(GCN_Layer(A, nhid, nout))

    def forward(self, X):
        length = len(self.mul)
        H = X
        for i in range(length):
            H_next = self.mul[i](H)
            H = H_next
        return H


# GatedGCN
class GatedGraphConvolution(nn.Module):

    def __init__(self, adj, in_features, out_features, useDropout=True, keepProb=0.5, useBatchNorm=True):
        super(GatedGraphConvolution, self).__init__()
        self.mlp = MLP(2, in_features, out_features, out_features, useDropout, keepProb, useBatchNorm)
        self.useDropout = useDropout
        self.keepProb = keepProb
        self.useBatchNorm = useBatchNorm
        self.W1 = torch.nn.Linear(out_features, out_features, bias=True)
        self.U1 = torch.nn.Linear(out_features, out_features, bias=True)
        self.W2 = torch.nn.Linear(out_features, out_features, bias=True)
        self.U2 = torch.nn.Linear(out_features, out_features, bias=True)
        self.W3 = torch.nn.Linear(out_features, out_features, bias=True)
        self.U3 = torch.nn.Linear(out_features, out_features, bias=True)
        self.adj = adj

        if self.useBatchNorm:
            self.bn = nn.BatchNorm1d(in_features)

    def forward(self, input):
        output = torch.spmm(self.adj, input)
        ul = F.relu(torch.add(self.W1(output), self.U1(input)))
        rl = F.relu(torch.add(self.W2(output), self.U2(input)))
        fl = F.tanh(torch.add(self.W3(output), self.U3(torch.mul(rl, input))))
        output = torch.mul(ul, fl) + torch.mul((1 - ul), input)
        if self.useBatchNorm:
            output = self.bn(output)
        output = F.relu(output)
        if self.useDropout:
            output = F.dropout(output, p=self.keepProb, training=self.training)
        output = self.mlp(output)
        return output


# 另一种版本别人实现的GCN,效果不太行
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self,adj, input_dim,hidden_dim,output_dim):
        super(GcnNet, self).__init__()
        self.adj=adj
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, feature):
        h = F.relu(self.gcn1(self.adj, feature))
        logits = self.gcn2(self.adj, h)
        return logits


def accurancy(mask, model, H, L):
    '''
    :param mask:掩码(tensor)
    :param model: 图模型
    :param H:feature 矩阵(tensor)
    :param L:标签(tensor)
    :return:准确率
    '''
    model.eval()
    with torch.no_grad():
        logits = model(H)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1).indices
        accuarcy = torch.eq(predict_y, L[mask].squeeze()).float().mean(0)
    return accuarcy


if __name__ == "__main__":

    #选择数据集
    #dataset="citeseer"
    dataset="cora"

    f, a, l, train_idx, valid_idx, test_idx = ut.load_data(dataset+".content", dataset+".cites")
    # 超参数
    learning_rate = 0.1
    weight_decay = 5e-4
    hidden_dim = 16
    output_dim = ut.get_nclass(l)
    epochs = 200

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 邻接矩阵
    A = torch.from_numpy(a).float().to(device)
    # 特征矩阵
    f_x = f / f.sum(1, keepdims=True)
    H = torch.from_numpy(f_x).float().to(device)
    # 标签one-hot矩阵
    L = torch.LongTensor(l).to(device)
    # 掩码向量
    tensor_train_mask = torch.from_numpy(train_idx).to(device)
    tensor_val_mask = torch.from_numpy(valid_idx).to(device)
    tensor_test_mask = torch.from_numpy(test_idx).to(device)

    model = GCN_Module(A, f.shape[1], hidden_dim, output_dim)
    #model = GcnNet(A,f.shape[1],hidden_dim,output_dim)
    #model=MLP(2,f.shape[1],hidden_dim,output_dim)
    #model=MLP(3,f.shape[1],hidden_dim,output_dim)
    #model=GCN_Multi(A, f.shape[1], hidden_dim, output_dim,3)
    #model = GatedGraphConvolution(A, f.shape[1], output_dim)

    # CrossEntropyLoss使用的不是one-hot码
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_history = []
    acc_history = []
    # 如果存在BN标准化，需要使用train初始
    model.train()

    for epoch in range(epochs):
        '''
        经典流程
        optimizer.zero_grad()             ## 梯度清零
        preds = model(inputs)             ## inference
        loss = criterion(preds, targets)  ## 求解loss
        loss.backward()                   ## 反向传播求解梯度
        optimizer.step()                  ## 更新权重参数
        '''
        optimizer.zero_grad()
        predict = model(H)
        train_predict = predict[tensor_train_mask]
        train_label = L[tensor_train_mask].squeeze()
        # 计算损失函数
        loss = criterion(train_predict, train_label)
        # 反向传播计算参数的梯度
        loss.backward()
        # 使用优化方法进行梯度更新
        optimizer.step()

        model.eval()
        # 默认后面计算不纳入梯度下降
        with torch.no_grad():
            valid_predict = predict[tensor_val_mask]
            # 获得最大值的对应下标
            valid_predict_value = valid_predict.max(1).indices
            # 获得准确率
            val_acc = torch.eq(valid_predict_value, L[tensor_val_mask].squeeze()).float().mean(0)

            train_predict = predict[tensor_train_mask]
            # 获得dim=1维,最大值的索引
            train_predict_value = train_predict.max(1).indices
            # 获得准确率
            train_acc = torch.eq(train_predict_value, L[tensor_train_mask].squeeze()).float().mean(0)
            print("Epochs: {:3d} loss {:3f} train_acc {:3f} valid_acc {:3f}".format(epoch, loss.item(), train_acc,
                                                                                      val_acc))

        loss_history.append(loss.item())
        acc_history.append(val_acc)

    print("Test accurancy :{:3f}".format(accurancy(tensor_test_mask, model, H, L)))
    ut.plot_loss_acc(loss_history, acc_history)
