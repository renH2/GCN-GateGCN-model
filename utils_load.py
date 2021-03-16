import numpy as np
import matplotlib.pyplot as plt
import torch
#from GNN_model import GCN_Layer,GCN_Module

def load_data(feature_src, adj_src, directed=False):
    '''
    :param feature_src:构造特征矩阵源文件
    :param adj_src: 构造邻接矩阵源文件
    :return:特征矩阵，邻接矩阵、label、训练节点下标、验证节点下标、测试节点下标
    '''
    name_idx = {}
    label_idx = {}
    counter = 0
    counter1 = 0
    f_matrix = []
    label = []
    first_line = True

    # 获取行数和列数
    n = get_line_num(feature_src)
    m = 0

    with open(feature_src) as fp:
        for i, line in enumerate(fp):
            data_split = line.strip().split()
            if len(data_split) == 2 and first_line:
                f_matrix = np.zeros([int(data_split[0]), int(data_split[1]) - 2])
                label = np.zeros([int(data_split[0]), 1])
                first_line = False
                continue
            elif first_line:
                f_matrix = np.zeros([n, len(data_split) - 2])
                m = len(data_split) - 2
                label = np.zeros([n, 1])
                first_line = False

            if data_split[0] not in name_idx:
                name_idx[data_split[0]] = counter
                counter = counter + 1

            transform_idx = name_idx[data_split[0]]

            if data_split[-1] not in label_idx:
                label_idx[data_split[-1]] = counter1
                counter1 = counter1 + 1

            label[transform_idx][0] = label_idx[data_split[-1]]

            for j in range(len(data_split) - 2):
                f_matrix[transform_idx][j] = data_split[j + 1]

    '''
    构造邻接矩阵
    '''
    adj_matrix = np.zeros([counter, counter])
    with open(adj_src) as fp:
        for i, line in enumerate(fp):
            data_split = line.strip().split()
            first_node = name_idx[data_split[0]]
            second_node = name_idx[data_split[1]]
            adj_matrix[first_node][second_node] = 1
            '''
            判断是否有向图
            '''
            if directed == False:
                adj_matrix[second_node][first_node] = 1

    adj_temp=adj_matrix.sum(1)
    for i,key in enumerate(adj_temp):
        if key==0:
            adj_matrix[i][i]=1

    train_idx,valid_idx,test_idx=dataset_split(n)
    train_mask = np.zeros(n, dtype=np.bool)
    valid_mask = np.zeros(n, dtype=np.bool)
    test_mask = np.zeros(n, dtype=np.bool)
    train_mask[train_idx]=True
    valid_mask[valid_idx]=True
    test_mask[test_idx]=True
    return f_matrix, adj_matrix, label,train_mask,valid_mask,test_mask


def label_onehot(label):
    '''
    :param label:label (ndarray),大小为 nclass*1
    :return: onehot张量(tensor)
    '''
    # 获得class数量
    count = 0
    cnt = []
    for i, v in enumerate(label):
        if v not in cnt:
            cnt.append(v)
            count = count + 1

    temp = torch.LongTensor(label)
    result = torch.zeros(temp.size(0), count)
    return result.scatter_(1, temp, 1)

def get_nclass(label):
    # 获得class数量
    count = 0
    cnt = []
    for i, v in enumerate(label):
        if v not in cnt:
            cnt.append(v)
            count = count + 1
    return count

# 划分数据集(按照0.6,0.2,0.2的比例划分)
def dataset_split(num_nodes):
    random_idx = np.random.permutation(num_nodes)
    train, val,test= random_idx[:int(num_nodes / 10 * 6)], random_idx[int(num_nodes / 10 * 6)+1:int(num_nodes / 10 * 9)],random_idx[int(num_nodes / 10 * 9)+1:]
    return train, val,test


# 获得一个文件的行数
def get_line_num(name):
    with open(name) as f:
        return len(f.readlines())

#绘制损失曲线和准确率曲线
def plot_loss_acc(loss_history,acc_history):
    '''
    :param loss_history:
    :param acc_history:
    :return:绘制曲线
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(acc_history)), acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('Val_Acc')
    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


if __name__ == "__main__":
    #f, a, l, train_idx, valid_idx, test_idx = load_data("cora.content", "cora.cites")
    f, a, l ,train_idx,valid_idx,test_idx= load_data("citeseer.content", "citeseer.cites")
    print(get_nclass(l))
