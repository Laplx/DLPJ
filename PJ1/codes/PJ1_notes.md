1. MLP 原始 **MODEL**

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])

optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)

scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)

loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

![image-20250430114558801](C:\Users\Laplace\AppData\Roaming\Typora\typora-user-images\image-20250430114558801.png)

epoch: 4, iteration: 1500
[Train] loss: 2.5873784189867557, score: 0.875
[Dev] loss: 3.851870612027071, score: 0.8052
best accuracy performence has been updated: 0.80090 --> 0.80560

2. Q1 更改 nhidden: 1024

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], **1024**, 10], 'ReLU', [1e-4, 1e-4])

optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)

scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)

loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

![image-20250430131846414](./../AppData/Roaming/Typora/typora-user-images/image-20250430131846414.png)

epoch: 4, iteration: 1500
[Train] loss: 4.8604874421764634, score: 0.78125
[Dev] loss: 3.368807692862332, score: 0.8342
best accuracy performence has been updated: 0.83000 --> 0.83360

3. Q1 加隐层

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], **512, 256**, 10], 'ReLU', [**5e-4, 5e-4, 5e-4**])

optimizer = nn.optimizer.SGD(init_lr=0.1, model=linear_model)

scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[**300, 900, 1800**], gamma=**0.3**)

loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

![image-20250430140402549](./../AppData/Roaming/Typora/typora-user-images/image-20250430140402549.png)

epoch: 4, iteration: 1500
[Train] loss: 2.1586735246819178, score: 0.90625
[Dev] loss: 2.5193894594750774, score: 0.8865
best accuracy performence has been updated: 0.88530 --> 0.88690

---

ever

![image-20250430171113258](./../AppData/Roaming/Typora/typora-user-images/image-20250430171113258.png)

4. Q2 改用动量 SGD **MODEL**

batch_size=**64** in RunnerM

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 512, 256, 10], 'ReLU', [5e-4, 5e-4, 5e-4])

optimizer = nn.optimizer.MomentGD(init_lr=0.1, model=linear_model, mu=0.9)

scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[300, 900, 1800], gamma=0.3)

loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

![image-20250430143500747](./../AppData/Roaming/Typora/typora-user-images/image-20250430143500747.png)

epoch: 4, iteration: 700
[Train] loss: 1.7268473106006428, score: 0.90625
[Dev] loss: 1.9065082148794459, score: 0.8859
best accuracy performence has been updated: 0.88460 --> 0.88560

---

batch_size=**32** in RunnerM 恶劣情况

两张图

5. Q3 L2 的等价，说明即可

6. Q3 Dropout

3 中改为 _Dropout

![image-20250430173651224](./../AppData/Roaming/Typora/typora-user-images/image-20250430173651224.png)

epoch: 4, iteration: 1500
[Train] loss: 2.878231366242557, score: 0.875
[Dev] loss: 2.5266903356960224, score: 0.8872
best accuracy performence has been updated: 0.88540 --> 0.88700

7. Q3 早停或 scheduler加快调控 **X**

8. Q4 已实现，说明即可
9. Q5 CNN（L2）

linear_model = nn.models.Model_CNN(size_list=[1, 16, 256, 10],lambda_list=[5e-4, 5e-4, 5e-4, 5e-4],kernel_size=[3, 3])

optimizer = nn.optimizer.SGD(init_lr=0.1,model=linear_model)

scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[800, 2400, 4000],gamma=0.3)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn,  batch_size=64, scheduler=scheduler)

![image-20250430180720917](./../AppData/Roaming/Typora/typora-user-images/image-20250430180720917.png)

epoch: 4, iteration: 300
[Train] loss: 3.4385684799182097, score: 0.109375
[Dev] loss: 3.514404609098541, score: 0.1159

---

损失函数先增高后走低

10. Q5 Bottleneck **X** 已实现

11. Q6 增强（MLP）

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 512, 256, 10], 'ReLU', [5e-4, 5e-4, 5e-4])

optimizer = nn.optimizer.SGD(init_lr=0.1, model=linear_model, mu=0.9)

scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[300, 900, 1800], gamma=0.3)

loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64, scheduler=scheduler)

![image-20250430153618532](./../AppData/Roaming/Typora/typora-user-images/image-20250430153618532.png)

early stop

epoch: 2, iteration: 900
[Train] loss: 2.2432868075449512, score: 0.703125
[Dev] loss: 1.7925035625743577, score: 0.7722

---

删去MomentGD

![image-20250430161935792](./../AppData/Roaming/Typora/typora-user-images/image-20250430161935792.png)

epoch: 4, iteration: 2300
[Train] loss: 10.415179997511956, score: 0.546875
[Dev] loss: 7.26817721611492, score: 0.6793

12. Q7 可视化（CNN）