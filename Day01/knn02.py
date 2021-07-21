# 约会网站判定
import pandas as pd
import matplotlib.pyplot as plt



# 导入数据集
datingTest = pd.read_table("datingTestSet.txt", header=None)
# print(datingTest.head())
# print(f"导入数据规格{datingTest.shape}")
# print(f"导入数据信息{datingTest.info}")

# 分析数据
# 把不同的数据用不同的颜色进行区分
Colors = []
for i in range(datingTest.shape[0]):
    m = datingTest.iloc[i, -1]      # 通过切片取标签
    # print(f"m={m}")
    if m == "didntLike":
        Colors.append("black")
    elif m == "smallDoses":
        Colors.append("orange")
    else:
        Colors.append("red")

# 绘制两两特征上的散点图
plt.rcParams["font.sans-serif"] = ['Simhei']    # 将字体设置为黑体

# 设置画布
pl = plt.figure(figsize=(12, 8))

# 添加子画布
fig1 = pl.add_subplot(221)  # 221表示将画布分为两行两列，且当前子画布为第一个
plt.scatter(datingTest.iloc[:, 1], datingTest.iloc[:, 2], marker=".", c = Colors)
plt.xlabel("玩游戏所占时间比")
plt.ylabel("每周消费冰激凌公升数")

fig2 = pl.add_subplot(222)
plt.scatter(datingTest.iloc[:, 0], datingTest.iloc[:, 1], marker=".", c = Colors)
plt.xlabel("每年飞行里程")
plt.ylabel("玩游戏所占时间比")

fig3 = pl.add_subplot(223)
plt.scatter(datingTest.iloc[:, 0], datingTest.iloc[:, 2], marker=".", c = Colors)
plt.xlabel("每年飞行里程")
plt.ylabel("每周消费冰激凌公升数")

plt.show()

# 数据归一化
# 使用欧几里得度量
# 由于发现某一项特征的重要性不够，可以使用数据归一化处理
# 此处使用0-1标准化
def minmax(dataset):
    minDf = dataset.min()
    maxDf = dataset.max()
    normSet = (dataset - minDf)/(maxDf-minDf)
    return normSet

datingT = pd.concat([minmax(datingTest.iloc[:, : 3]), datingTest.iloc[:,3]], axis=1)
# print(datingT.head(10))

"""
函数功能：切分训练集和测试集
参数说明：
    dataset：原始数据集
    rate：训练集所占比例
返回：
    切分好的训练集和数据集
"""
def randSplit(dataset, rate=0.9):
    n = dataset.shape[0]        # 取数据集行数
    m = int(n*rate)     # 必须保证随机性
    train = dataset.iloc[:m, :]
    test = dataset.iloc[m:, :]
    test.index = range(test.shape[0])
    train.index = range(train.shape[0])
    return train, test


train, test = randSplit(datingT)
# print(test)



# 定义分类器
def datingClass(train, test, k):
    n = train.shape[1] - 1  # 除标签外的列数
    m = test.shape[0]
    result = []
    for i in range(m):
        # 欧几里得度量
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) ** 2).sum(1)) ** 0.5)
        dist_1 = pd.DataFrame({"dist":dist,
                               "labels":(train.iloc[:, n])})
        dr = dist_1.sort_values(by="dist")[ : k]
        re = dr.loc[:, "labels"].value_counts()
        result.append(re.index[0])
        # 通过pd.Series函数将预测结果转化成可加入test数据集的格式，并设置列名为predict
    result = pd.Series(result)
    test.loc[:, "predict"] = result
    print(test)
    acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()
    print(f"模型预测的准确度为{acc}")

    return test

datingClass(train, test, 5)