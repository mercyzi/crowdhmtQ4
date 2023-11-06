import torch
from utils import *
from gnn import *
import os
import pickle
from torch_geometric.data import DataLoader

# sample 360 45 
# node   7.1 6.3
train_data = load_json('/home/zxh/clinical/src/data/train_data.json')
dev_data = load_json('/home/zxh/clinical/src/data/dev_data.json')

pre_train_saved = '/home/zxh/clinical/src/data/pre_train_data.json'
pre_dev_saved = '/home/zxh/clinical/src/data/pre_dev_data.json'

dev_acc_saved = '/home/zxh/clinical/src/model/temp/dev_acc_17.p'

if os.path.exists(pre_train_saved):
    with open(pre_train_saved, 'r') as file:
        train_ds = json.load(file)
else:
    train_ds = preprocess(train_data)
    with open(pre_train_saved, 'w') as json_file:
        json.dump(train_ds, json_file)

if os.path.exists(pre_dev_saved):
    with open(pre_dev_saved, 'r') as file:
        dev_ds = json.load(file)
else:
    dev_ds = preprocess(dev_data)
    with open(pre_dev_saved, 'w') as json_file:
        json.dump(dev_ds, json_file)

# g_saved = "/home/zxh/clinical/src/model/temp/data.pt"
# if os.path.exists(g_saved):
#     graph = torch.load(g_saved)
# else:
#     graph = create_graph([train_ds, dev_ds])
#     torch.save(graph, g_saved)
# data = graph

trian_ds = create_graph(train_ds)

dev_ds = create_graph(dev_ds)

train_loader = DataLoader(trian_ds, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=len(dev_ds), shuffle=False)


# 图神经网络
model = Model(16, 64, 768).to(device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

# 创建损失函数
criterion = nn.MSELoss()

# import matplotlib.pyplot as plt
# dev_acc1 = pickle.load(file=open("/home/zxh/clinical/src/model/temp/dev_acc_10.p", "rb"))
# dev_acc2 = pickle.load(file=open("/home/zxh/clinical/src/model/temp/dev_acc_15.p", "rb"))
# dev_acc3 = pickle.load(file=open("/home/zxh/clinical/src/model/temp/dev_acc_16.p", "rb"))
# dev_acc4 = pickle.load(file=open("/home/zxh/clinical/src/model/temp/dev_acc_17.p", "rb"))

# plt.plot(range(len(dev_acc1)), gaussian_filter1d(dev_acc1, sigma=5), color = 'r')
# plt.plot(range(len(dev_acc2)), gaussian_filter1d(dev_acc2, sigma=5), color = 'b')
# plt.plot(range(len(dev_acc3)), gaussian_filter1d(dev_acc3, sigma=5), color = 'green')
# plt.plot(range(len(dev_acc4)), gaussian_filter1d(dev_acc4, sigma=5), color = 'orange')

# plt.xlabel('iter')
# plt.ylabel('Dev Acc')

# plt.savefig('acc.png',dpi=400)

# assert 0

dev_acc = []
# 训练
for epoch in range(100000):
    model.train()
    total_loss =  0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_mask = sample_balance_mask(batch)
        predictions = model(batch)
        
        loss = criterion(predictions[batch_mask], batch.ground_label[batch_mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    if epoch % 10 ==0:
        print(f"Epoch: {epoch:03d}, Loss: {total_loss :.4f}")

    model.eval()
    with torch.no_grad():
        correct = 0
        for batch in train_loader:
            batch = batch.to(device)
            predictions = model(batch)
            train_pred = predictions
            train_pred_idx = []
            train_GT_idx = []
            # 获取原索引值为1的位置
            indices = torch.nonzero(batch.ground_label == 1).squeeze()  
            node_s, node_e = 0, 0
            for i, t in enumerate(indices):
                node_e = indices[i]
                max_v = 0
                max_idx = node_s
                train_GT_idx.append(batch.edge_label_index[1][node_e].item())
                for idx in range(node_s, node_e+1):
                    if train_pred[idx] > max_v:
                        max_idx = idx
                        max_v = train_pred[idx]

                node_s = indices[i] + 1
                train_pred_idx.append(batch.edge_label_index[1][max_idx].item())
            train_pred_idx = torch.LongTensor(train_pred_idx)
            train_GT_idx = torch.LongTensor(train_GT_idx)
            
            correct += train_pred_idx.eq(train_GT_idx).sum().item()
        Acc = correct / len(trian_ds)
        if epoch % 10 ==0:
            print(f"Train Acc: {Acc :.4f}")

    model.eval()
    with torch.no_grad():
        correct = 0
        for batch in dev_loader:
            batch = batch.to(device)
            predictions = model(batch)
            train_pred = predictions
            train_pred_idx = []
            train_GT_idx = []
            # 获取原索引值为1的位置
            indices = torch.nonzero(batch.ground_label == 1).squeeze()  
            node_s, node_e = 0, 0
            for i, t in enumerate(indices):
                node_e = indices[i]
                max_v = 0
                max_idx = node_s
                train_GT_idx.append(batch.edge_label_index[1][node_e].item())
                for idx in range(node_s, node_e+1):
                    if train_pred[idx] > max_v:
                        max_idx = idx
                        max_v = train_pred[idx]

                node_s = indices[i] + 1
                train_pred_idx.append(batch.edge_label_index[1][max_idx].item())
            train_pred_idx = torch.LongTensor(train_pred_idx)
            train_GT_idx = torch.LongTensor(train_GT_idx)
            
            correct += train_pred_idx.eq(train_GT_idx).sum().item()
        Acc = correct / len(dev_ds)
        if epoch % 10 ==0:
            dev_acc.append(Acc)
            print(f"Dev Acc: {Acc :.4f}")
        if epoch % 500 == 1:
            
            pickle.dump(file=open(dev_acc_saved, "wb"), obj=dev_acc)
        
assert 0
