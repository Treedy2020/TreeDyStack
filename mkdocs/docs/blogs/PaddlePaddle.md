---
title: 两个分类任务快速入门PaddlePaddle
---
最近百度的LLM文心一言的表现被许多网友嘲笑，但上个月入门PaddlePaddle以来，我对百度是由黑转粉。Paddle作为比PyTorch更早开源的深度学习框架，很多国内的同学对它的映像并不深或者并不好。出于希望对开源框架推广的原因，百度的AI Studio目前只需要完成这篇文章中的两个分类任务其中之一(注册[AI Studio](https://aistudio.baidu.com/aistudio/)并在[AI学习地图](https://aistudio.baidu.com/aistudio/learnmap?_origin=newbie)界面选择二者之一即可开启任务)：   

- [新浪新闻文本标题分类](https://aistudio.baidu.com/aistudio/competition/detail/10/0/introduction)  
- [猫十二分类体验赛](https://aistudio.baidu.com/aistudio/competition/detail/136/0/introduction)

即可以获得100小时的32G V100 的GPU使用时长，同时每周还有4小时的4块V100（128G）显存的使用时长，每天登录运行即送12小时的V100（16G）的使用时长。百度并不是从今年开始这样干，而是从很久以前就开始这样送了，AI Studio和目前许多致力于把显卡的租用价格炒高的云计算平台不同，这个平台甚至**不能充钱**。Paddle时常被诟病非常的像Torch，一部分API的使用是十分像的，其实我觉得对于广大缺算力的硕士和本科生这甚至是福音，因为原则上你可以用Paddle提供的算力和数据集资源熟悉Numpy，Pandas这些深度学习算法工程师必备的Python库，也可以通过熟悉Paddle来熟悉Torch，AI Studio可以为资金并不富裕的同学提供宝贵的算力，特别是在LLM逐渐占据我们的视野的今天，有一个国产的框架可以无偿为你提供128G的显存，无论是对做分布式模型的训练的学习，还是玩更大的模型这都是很诱人的。相较之百度网盘这种不充钱就强制限制网速的产品，简直不像是同一家公司。

## 任务背景介绍
新浪新闻标题分类和猫十二分类是NLP和CV的典型分类任务，非常适合作为新手和PaddlePaddle框架的入门项目。前者的目的是为一个中文的新闻标题在政治、教育、财经...等14个类别上进行分类，而猫十二分类项目则希望为一张图片中的猫打上它的具体所属小类的标签。在这篇文章中，我们将统一采用对预训练模型在分类任务上进行微调的方式为这两个任务提供Baseline，具体的赛事和数据描述见：  
- [新浪新闻文本标题分类](https://aistudio.baidu.com/aistudio/competition/detail/10/0/introduction)  
- [猫十二分类体验赛](https://aistudio.baidu.com/aistudio/competition/detail/136/0/introduction)

## 导入库并预定义一些常数
首先导入一些需要的包
=== "新浪新闻标题分类"
    ```py
    import os
    import time
    import paddle
    import pandas as pd
    import numpy as np
    import paddlenlp
    import paddle.nn.functional as F 
    from tqdm import tqdm
    from collections import defaultdict
    from functools import partial
    from paddle.io import Dataset, DataLoader
    from paddlenlp.transformers import BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
    from sklearn.model_selection import train_test_split
    ```
=== "猫十二分类"
    ```
    import os
    import time
    import paddle
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from paddle.io import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    ```

预先定义一些需要用到的常数，
=== "新浪新闻标题分类"
    ```py
    EPOCHS = 3 # 总的训练轮次
    LEARNING_RATE = 5e-5    # 全局学习率
    MODEL_NAME = "hfl/rbt4" # 预训练模型名称
    SAVE_PATH = './' + MODEL_NAME.split('/')[-1]    #  检查点保存地址
    BATCH_SIZE = 1024   # 批数量
    SAVE_FREQUENCE = 100    # 保存步长
    LOG_FREQUENCE = 20  # 训练日志打印步长
    TOTAL_SIZE = 83599  # 猫十二项目最终需要用于测试的数据量
    NUM_WORKERS = 4 # 数据加载器的线程数量
    TEST_SIZE = 0.20    # 验证数据集从训练数据中划分的比例
    RANDOM_STATE = 1024 # 随机种子（sklearn.model_selection的train_test_split方法中使用）
    BASE_LINE = 0.90 # 开始保存检查点的验证集Base line
    MAX_SEQ_LEN = 48 # 统计验证标题数据的最长长度为48，sequence的长度也只需要设置为48即可
    ```
=== "猫十二分类"
    ```py
    EPOCHS = 20 # 总的训练轮次
    LEARNING_RATE = 1e-4    # 全局学习率
    SAVE_FREQUENCE = 10 # 保存步长
    BATCH_SIZE=128  # 批数量
    LOG_FREQUENCE = 10  # 训练日志打印步长
    TEST_SIZE = 0.20    # 验证数据集从训练数据中划分的比例
    RANDOM_STATE = 1024 # 随机种子（sklearn.model_selection的train_test_split方法中使用）
    IMG_SIZE = (224, 224)   # 图像转为tensor后的后两维大小
    BASE_LINE = 0.90 # 开始保存检查点的验证集Base line
    ```


## 数据读取
=== "新浪新闻标题训练数据读取"
    ```py
    title_with_labels = []
    with open('./data/Train.txt', 'r') as f:
        for line in f.readlines():
            label, _, title = line.strip('\n').split('\t')
            title_with_labels.append((title, int(label)))
    ```
=== "猫十二训练数据读取"
    ```py
    image_data = []
    with open('./data/cat_12/train_list.txt', 'r') as f:
        for line in f.readlines():
            img_path, label = line.strip('\n').split('\t')
            image_data.append((img_path, label))
    ```

## 训练和验证数据的划分
使用`sklearn.model_selection`中的`train_test_split`方法即可，两个数据集均采用0.8/0.2（test_size=0.2）的比例分别作为训练集和验证集。

=== "新浪新闻标题数据集划分"
    ```py
    train_titles, val_titles = train_test_split(title_with_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    ```
=== "猫十二数据集划分"
    ```py
    train_imgs, val_imgs = train_test_split(image_data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    ```

## 构建数据集类
此处新浪新闻标题分类的数据集`TextDataset`类在初始化时，接收一个由预训练模型相对应的分词器tokenizer，负责将对应的输入的标题转化为对应的`tokens`，而猫十二分类器中则简单的定义`image_to_feature_vector`函数得到对应图像的特征。
=== "新浪新闻标题分类"
    ```py
    class TextDataset(Dataset):
        def __init__(self, data,  tokenizer, max_seq_length=MAX_SEQ_LEN, isTest=False):
            super(TextDataset, self).__init__()
            self.data = data
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length
            self.isTest = isTest

        def __getitem__(self, index):
            if  not self.isTest:
                text, label = self.data[index][0], self.data[index][1]
                encoded = self.tokenizer.encode(text, max_seq_len=self.max_seq_length, pad_to_max_seq_len=True)
                input_ids, token_type_ids  = encoded['input_ids'], encoded['token_type_ids']
                return tuple([np.array(x, dtype='int64') for x in [input_ids, token_type_ids, [label]]])
            else:
                title = self.data[index]
                encoded = self.tokenizer.encode(title, max_seq_len=self.max_seq_length, pad_to_max_seq_len=True)
                input_ids, token_type_ids  = encoded['input_ids'], encoded['token_type_ids']
                return tuple([np.array(x, dtype='int64') for x in [input_ids, token_type_ids]])

        def __len__(self):
            return len(self.data)
    ```

=== "猫十二分类"
    ```py
    # 定义一个函数来将图片转换成特征
    def image_to_feature_vector(image_path):
        # 打开图片并将其大小重置为 IMG_SIZE
        image = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
        # 将图像转换为 NumPy 数组
        image_array = np.array(image)
        mean, std = np.mean(image_array), np.std(image_array)
        image_array = (image_array - mean)/std

        return paddle.transpose(paddle.to_tensor(image_array, dtype='float32'), [2, 1, 0])

    class ImageDataSet(Dataset):
        def __init__(self, image_data, isTest=False):
            super(ImageDataSet, self).__init__()
            self.image_data = image_data
            self.isTest=isTest

        def __len__(self):
            return len(self.image_data)
            
        def __getitem__(self, index):
            if not self.isTest:
                img_path, label = self.image_data[index]
                return image_to_feature_vector(os.path.join('./data/cat_12', img_path)), paddle.to_tensor([int(label)], dtype='int64')
            else:
                img_path = self.image_data[index]
                return image_to_feature_vector(os.path.join('./data/cat_12', img_path))
    ```

## 采用API接口获取预训练模型

=== "新浪新闻标题分类"
    ```py
    # 需要获取模型和对应的分词器，num_classes参数对应了14种新闻标题的类别
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=14)
    # 将模型转为静态图模型，加快训练
    model = paddle.jit.to_static(model) 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ```

=== "猫十二分类"
    ```py
    # 使用预训练过后的resnet50，num_classes参数对应12种猫的种类
    model = paddle.vision.models.resnet50(pretrained=True, num_classes=12)
    
    # 将模型转为静态图模型，加快训练
    model = paddle.jit.to_static(model)
    ```

## 数据加载器

=== "新浪新闻标题分类"
    ```py
    # 获得训练和验证数据集
    train_dataset, val_dataset = TextDataset(data=train_titles, tokenizer=tokenizer), TextDataset(data=val_titles, tokenizer=tokenizer)

    # 获得采样器
    train_batch_sampler = BatchSampler(train_dataset,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            )

    val_batch_sampler = BatchSampler(val_dataset,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            )

    # 获得数据加载器
    train_data_loader = DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            return_list=True,
                                            num_workers=NUM_WORKERS)
    val_data_loader = DataLoader(dataset=val_dataset,
                                            batch_sampler=val_batch_sampler,
                                            return_list=True,
                                            num_workers=NUM_WORKERS)
    ```

=== "猫十二分类"
    ```py
    # 获得训练和验证数据集
    train_dataset, val_dataset = ImageDataSet(image_data=train_imgs), ImageDataSet(image_data=val_imgs)

    # 获得采样器
    train_batch_sampler = BatchSampler(train_dataset,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            )

    val_batch_sampler = BatchSampler(val_dataset,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            )

    # 获得数据加载器
    train_data_loader = DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            return_list=True,
                                            num_workers=NUM_WORKERS)
    val_data_loader = DataLoader(dataset=val_dataset,
                                            batch_sampler=val_batch_sampler,
                                            return_list=True,
                                            num_workers=NUM_WORKERS)
    ```

## 定义评估函数
在训练的过程中，需要一个评估函数用在验证数据集上，用于帮助我们判断模型的训练情况，这里均定义`evaluate()`，计算并返回两个分类任务在各自的验证集上的整体Acc并返回，两个`evaluate`函数也仅有batch中的参数处理有细微不同:
=== "新浪新闻标题分类"
    ```py
    def evaluate(model, criterion, metric, data_loader):
        model.eval()
        metric.reset()
        losses = []
        for batch in tqdm(data_loader):
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            losses.append(loss.numpy())
            correct = metric.compute(logits, labels)
            metric.update(correct)
        accu = metric.accumulate()
        print("eval loss: %.5f, accu: %.7f" % (np.mean(losses), accu))
        model.train()
        metric.reset()
        return accu
    ```

=== "猫十二分类"
    ```py
    def evaluate(model, criterion, metric, data_loader):
        model.eval()
        metric.reset()
        losses = []
        for batch in tqdm(data_loader):
            input_ids, labels = batch
            logits = model(input_ids)
            loss = criterion(logits, labels)
            losses.append(loss.numpy())
            correct = metric.compute(logits, labels)
            metric.update(correct)
        accu = metric.accumulate()
        print("eval loss: %.5f, accu: %.7f" % (np.mean(losses), accu))
        model.train()
        metric.reset()
        return accu
    ```

## 训练过程

=== "新浪新闻标题分类"
    ```py
    # 定义优化器、损失函数和Acc计算器
    optimizer = paddle.optimizer.Adam(learning_rate=LEARNING_RATE,
                                parameters=model.parameters(),
                                )
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    # 调整至训练模式
    model.train() 
    best_acc = BASE_LINE # 模型开始保存的BaseLine

    for epoch in range(EPOCHS):
        print(f"epoch: {epoch + 1}, {time.ctime()}")
        start_t = time.time()
        metric.reset()
        for ind, item in enumerate(train_data_loader()):
            if ind and (not ind%SAVE_FREQUENCE):
                accu = evaluate(model, criterion, metric, val_data_loader)
                if accu > best_acc:
                    best_acc = accu
                    print('\t Best Acc: {:.6f}'.format(accu))
                    model.save_pretrained(SAVE_PATH)
                    tokenizer.save_pretrained(SAVE_PATH)
            input_ids, token_type_ids, labels = item
            logits = model(input_ids, token_type_ids)
            print(logits, labels)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)

            correct = metric.compute(probs, labels)
            batch_acc = metric.update(correct)
            acc = metric.accumulate()
            
            loss.backward()
            ave_t = (time.time() - start_t)/(ind + 1)
            extra_h = ave_t*(len(train_data_loader) - ind + 1)/3600
            if ind and (not ind%LOG_FREQUENCE):
                print(f'\t step:{ind}/{len(train_data_loader)},', 'average time: {:.4f},'.format(ave_t), 'loss: {:.6f}'.format(loss.numpy()[0]), 'Batch Acc:{:.9f}, Acc:{:.9f}'.format(batch_acc, acc))

            optimizer.step()
            optimizer.clear_grad()
    ```

=== "猫十二分类"
    ```py
    optimizer = paddle.optimizer.Adam(learning_rate=LEARNING_RATE,
                            parameters=model.parameters(),
                            )
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    # 调整至训练模式
    model.train() 
    best_acc = BASE_LINE # 模型开始保存的BaseLine

    for epoch in range(EPOCHS):
        print(f"epoch: {epoch + 1}, {time.ctime()}")
        start_t = time.time()
        metric.reset()
        for ind, item in enumerate(train_dataloader):
            if ind and (not ind%SAVE_FREQUENCE):
                accu = evaluate(model, criterion, metric, val_dataloader)
                if accu > best_acc:
                    best_acc = accu
                    print('\t Best Acc: {:.6f}'.format(accu))
                
            input_ids,  labels = item
            logits = model(input_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)

            correct = metric.compute(probs, labels)
            batch_acc = metric.update(correct)
            acc = metric.accumulate()
            
            loss.backward()
            ave_t = (time.time() - start_t)/(ind + 1)
            extra_h = ave_t*(len(train_dataloader) - ind + 1)/3600
            if ind and (not ind%LOG_FREQUENCE):
                print(f'\t step:{ind}/{len(train_dataloader)},', 'average time: {:.4f},'.format(ave_t), 'loss: {:.6f}'.format(loss.numpy()[0]), 'Batch Acc:{:.9f}, Acc:{:.9f}'.format(batch_acc, acc))

            optimizer.step()
            optimizer.clear_grad()
    ```

## 推理预测

### 测试数据预处理
=== "新浪新闻标题分类"
    ```py
    test_title = []
    with open('./data/Test.txt', 'r') as f:
        for line in f.readlines():
            test_title.append(line.strip('\n'))

    test_dataset = TextDataset(data=test_title, tokenizer=tokenizer, isTest=True)
    test_batch_sampler = BatchSampler(test_dataset,
                                      shuffle=False, # 注意此处不应该将测试数据集打乱
                                      batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_sampler=test_batch_sampler,
                                  num_workers=4)
    ```

=== "猫十二分类"
    ```py
    img_names = os.listdir('./data/cat_12/cat_12_test')
    test_data = [os.path.join('./data/cat_12/cat_12_test', name) for name in img_names]

    test_dataset = ImageDataSet(image_data=test_data, isTest=True)
    test_batch_sampler = BatchSampler(test_dataset,
                                      shuffle=False, # 注意此处不应该将测试数据集打乱
                                      batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_sampler=test_batch_sampler,
                                  num_workers=4)
    ```

### 推理

=== "新浪新闻标题分类"
    ```py
    model.eval()
    res = []
    for input_ids, token_type_ids in tqdm(test_data_loader):
        logits = model(input_ids, token_type_ids)
        curr_ind = paddle.argmax(logits, axis=1)
        res += curr_ind.numpy().tolist()
    ```

=== "猫十二分类"
    ```py
    model.eval()
    res = []
    for img_tensor in tqdm(test_data_loader):
        logits = model(img_tensor)
        curr_ind = paddle.argmax(logits, axis=1)
        res += curr_ind.numpy().tolist()
    ```

### 记录结果

=== "新浪新闻标题分类"
    ```py
    class_lis = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']
    label_dict = {ind: content for ind, content in enumerate(class_lis)}
    assert len(res) == TOTAL_SIZE, '最终输出的list长度不正确，需要检查test_data是否合理划分'
    with open('./result.txt', 'w') as f:
        print('推理样例：')
        for i in range(TOTAL_SIZE):
            text = label_dict[res[i]] + '\n'
            if not i%100:
                print('\t', label_dict[res[i]] + '\t' + test_title[i])
            f.write(text)
    ```

=== "猫十二分类"
    ```py
    res_pd = pd.DataFrame({'name':img_names, 'label':res})
    res_pd.to_csv('./result.csv', header=None, index=False)
    ```

## 最终结果
在对应的赛事提交入口提交最终的保存的result.txt和result.csv即可，由于这个Demo中的例子都采用了相对比较小的模型，可以把batch_size设置的很大，训练也很快，可以得到相对不错的Base Line，本人使用`ernie-3.0-base-zh`和`resnet_50`在对应的任务下得到的分数分别是88.2分（2020年3月排行榜42位）和92.5分（2023年5月排行榜135位）。

你也可以简单的通过更换更大的预训练模型来达到更好的效果，对于新浪新闻标题分类而言，只需要更换前述代码中的MODEL_NAME即可，Paddlenlp提供了[预训练模型库和模型适合的任务索引](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html)，只要找到适用于Sequence Classification均可使用。对于猫十二分类，则可以在`paddle.vision.models`的[官方API](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/Overview_cn.html)中寻找想要尝试的其他模型。

本方法的缺陷是：

- 参数的设置大多是随意的，有很大的调参空间；

- 数据的预处理步骤较为简单。对于新浪新闻标题项目，数据的量是足够多的，但样本可能存在小幅度的不平衡；对于猫十二分类项目，数据的量并不大，可以考虑进行一定的数据增强对原数据进一步的利用；

- 模型的训练并不快。由于训练时采用了单精度的动转静模型训练方式，训练的速度比动态图更快，但推荐采用官方的API教程[自动混合精度训练（AMP)](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/amp_cn.html)修改少量代码转为fp16训练，推理和训练的速度都会快很多。

## 参考资料
1. [个人新浪新闻标题分类项目主页](https://aistudio.baidu.com/aistudio/projectdetail/6021258)
2. [个人猫十二分类项目主页](https://aistudio.baidu.com/aistudio/projectdetail/6086926)
3. [个人新浪新闻标题Github主页](https://github.com/Treedy2020/NewsTitles)
1. [新浪新闻标题分类优秀参考项目](https://aistudio.baidu.com/aistudio/projectdetail/3502908)
4. [猫十二官方优秀Baseline项目1](https://aistudio.baidu.com/aistudio/projectdetail/3461935?channelType=0&channel=0)
5. [猫十二官方优秀Baseline项目2](https://aistudio.baidu.com/aistudio/projectdetail/3906013)



