# 模型评估与改进

_______

[toc]

## 模型评估

### 交叉验证

交叉验证是一种评估泛化性能的统计学方法，在交叉验证过程中数据被多次划分，分别对模型进行训练。

最常用的交叉验证法是k折交叉验证。

sklearn中利用`model_selection`模块中的`cross_val_score`函数来实现交叉验证方法。

```python
# 默认情况下，执行3折交叉验证
scores = cross_val_score(model, X, Y, cv = 3)
# 通常使用交叉验证评分的均值作为精度的表现形式。
scores.mean()

```

若折与折之间精度由较大变化，说明模型要么强烈依赖于将某个折作为训练数据，要么是因为数据集过小。

对数据进行多次划分，为我们提供了模型对训练集选择的敏感性信息，score的最大最小值表示了模型应用于新数据在最优和最坏情况下的可能表现。

交叉验证的目的只是评估算法在特定数据集上训练后的泛化性能好坏，不能用来构建可应用新数据的模型。

通过==交叉验证分离器==作为cv参数，可以对数据划分的过程作更加精细的控制。

```python
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle = True, random_rate=42)# shuffle通过将数据打乱来代替分层，以打乱样本按标签的排序
scores = cross_val_score(X, Y, kfold)

```



### 留一法交叉验证

留一法交叉验证相当于每折只包含单个样本的k折交叉验证，对于每次划分，选择当个数据点作为测试集，这种交叉验证方法对小型数据集可以给出更好的估计结果。

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()
scores = cross_val_score(X, Y, kfold)

```

### 打乱划分交叉验证

在打乱划分交叉验证中，每次划分为训练集取样`train_size`, 测试集取样`test_size`个不相交点。

```python
from sklearn.model_selection import ShuffleSplit, cross_val_score

shuffle_splite = ShuffleSplit(test_size = .2, train_size=0.8, n_split=10)
scores = cross_val_score(X, Y, shuffle_splite)

```

打乱划分交叉检验可以在训练集和测试集大小之外独立控制迭代次数，还允许每次迭代中只使用部分数据。

### 分组交叉验证

当数据中分组高度相关是，分组交叉验证是适用的。

```python
from sklearn.model_selection import GroupKFold, cross_val_score

shuffle_splite = ShuffleSplit(groups=groups, n_split=10)
scores = cross_val_score(X, Y, shuffle_splite)

```







## 模型调参

### 网格搜索

