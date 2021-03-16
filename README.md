utils_load：处理数据所需函数

GNN_model：目前只有MLP,两种GCN(GCN_Module、GcnNet),多层GCN (GatedGNN还有点问题）

---

**运行方法**：运行GNN_model.py即可

---

**修改数据集**：（目前可选cora和citeseer，pubmed找不到...）

<img src="C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20201118214633671.png" alt="image-20201118214633671" style="zoom:67%;" />

**加载数据集**：

![image-20201118214748795](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20201118214748795.png)

f为feature matrix，维数为[node_num, feature_num]

a为邻接矩阵，l为label矩阵

train_idx，valid_idx，test_idx为训练、验证、测试的掩码

**超参数修改**：

<img src="C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20201118215052633.png" alt="image-20201118215052633" style="zoom:67%;" />

**模型选择：**

<img src="C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20201118215215313.png" alt="image-20201118215215313" style="zoom:67%;" />