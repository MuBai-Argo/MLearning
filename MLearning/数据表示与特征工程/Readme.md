# 数据表示与特征工程

****

[toc]

## 数据表示与特征工程

对某个特定应用来说，如何找到最佳数据表示，这个问题被称为特征工程。

### 分类变量

分类特征来自一系列可能的取值，表示的是定性属性。

#### One-hot编码

到目前为止，表示分类变量最常用大方法就是使用one-hot比那吗或者N取一编码。将一个分类变量替换成一个或多个新特征。

### 分箱

数据表示的最佳方法不仅取决于数据的语义，还取决于所使用的模型种类。线性模型与基于树的模型是两种成员很多同事又非常常用的模型，再处理不同的特征表示时就具有非常不同的性质。

有一种方法可以让线性模型在连续数据上变得更加强大，就是使用特征分箱（也成为离散化），即将其划分为多个特征。

我们假设将特征的输入范围划分为固定个数的箱子。

```python
bins = np.linspace(-n, n, 11)	# 创建10个箱子

```

随后我们记录每个数据点所属的箱子。

```python
which_bin = np.digitize(X, bins=bins)	# 返回X中数据在bins中的位置

```

将数据集中但各输入特征变换为一个分类特征，用于表示数据点所在的箱子。利用sklearn模块中的preprocessing模块的OneHotEncoder（目前只适用于值为整数的分类变量）将这个离散特征变换为One-Hot编码。

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned=encoder.transform(which_bin)

```

由于一共指定了10个箱子，所以变换后X——binned数据集现在包含10个特征。

随后可以在one-hot编码后的数据上构建新的线性模型或者决策树模型。

```python
line_binned=encoder.transform(np.digitize(line, bins=bins))
reg=LinearRegression().fit(X_binned, y)

```

对于每个箱子，线性模型与树形模型都预测出一个常数值，因为每个箱子内的特征是不变的，所以对于一个箱子内的所有点，任何模型都会预测出相同的值。对于特定的数据集，如果有充分的理由使用线性模型（如数据集很大、维度很高）但有些特征和输出的关系是非线性的，那么分箱是提高建模能力的好办法。

### 交互特征与多项式特征

想要丰富的特征表示，特别是对于线性模型来说，一种方法是添加原始数据的交互特征和多项式特征。

在分箱的基础上，为线性模型添加原始特征。

则每个箱子都会学习到一个偏移值和斜率。我们希望每个箱子都有一个不同的斜率，则我们可以添加交互特征或乘积特征。即箱子指示符和原始特征的乘积。

```python
X_product=np.hstack([X_binned, X*X_binned])
reg = LinearRegression().fit(X_product, y)

```

使用分箱是拓展连续特征的一种方法，另一种方法是使用原始特征的多项式。

对于给定的特征X我们可以考虑其n次方，这里用preprocessing模块中的PolynomialFeatures实现

```python
from sklearn.preprocessing import PolynomialFeatures
# 多项式的次数为10故生成了10个特征
poly = PloynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = ploy.transform(X)

```

可以通过调用get_feature_names方法来获取特征的语义，从而给出每个特征的指数。

```python
ploy.get_feature_names()
```

将多项式特征与线性回归模型一起使用，可以得到多项式回归模型

```python
reg = LinearRegression().fit(X_ploy, y)
```

## 自动化特征选择

在添加新特征或处理一般的高维数据集时，最好将特征的数量减少到只包含最有用的那些特征，并删除其余特征，这样会得到泛化能力更好、更简单的模型。

有三种基本策略对特征的作用进行判断。

1. 单变量统计
2. 基于模型的选择
3. 迭代选择

以上都属于监督方法。

### 单变量统计

在单变量统计中，我们计算每个特征和目标值之间的关系是否存在统计显著性，然后选择具有==最高置信度的特征==，对于分类问题，这也被称为==方差分析（ANOVA）==。这些测试的一个关键性质就是他们是单变量的。即他们只单独考虑每个特征。

因此，如果一个特征只有在与另一个特征合并时才具有信息量，那么这个特征将被舍弃。

单变量测试的计算速度通常很快，且不需要构建模型。他们完全独立于你可能想要在特征选择后应用的模型。

在sklearn中进行单变量特征选择，你需要选择一项测试。

对分类问题通常是`f_classif(默认值)`对回归问题通常是`f_regression`然后基于测试中确定的p值来选择一种舍弃特征的方式。

所有设其参数的方法都用阈值来舍弃所有p值过大的特征（这意味着他们不可能与目标值相关）。

计算阈值的方法各有不同，最简单的方式是`SelecKBest`和`SelectPercentile`。

`SelecKBest`选择固定数量的k个特征，`SelectPercentile`选择固定的百分比的特征。

```python
select=SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected=select.transform(X_train)
X_test_selected=select.transform(X_test)
# 将去除参数情况进行可视化
mask=select.get_support()
print(mask)
plt.matshow(mask.reshape(1, -1), cmap="grey r")
plt.xlabel("Sample index")
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
print(logistic.score(X_test, y_test))
logistic.fit(X_train_selected, y_train)
print(logistic.score(X_test_selected, y_test))
      
```

如果因为特征量太大以至于无法构建模型或者怀疑许多特征完全没有信息量的情况下，单变量特征选择是非常有效的。

### 基于模型的特征选择

基于模型的特征选择使用一个监督机器学习模型来判断每个特征的重要性，并且保留最重要特征。

用于特征选择的监督模型不需要与用于最终监督建模的模型相同。特征选择模型需要为每个特征提供某种重要性度量，一边用这个度量对特征进行排序。

决策树和基于决策树模型提供了feature_importances属性可以直接编码每个特征的重要性。线性模型系数的绝对值也可以用于表示特征的重要性。

要想使用基于模型的特征选择我们需要使用`SelectFromModel`变换器

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select=SelectFromModel(
    RandomForestClassifier(n_estimators=100, 
                           random_state=42),
    threshold="median")
```

SelectFromModel类选出重要性度量大于给定阈值的所有特征。

### 迭代特征选择

在迭代特征选择中将会构建一系列模型，每个模型都是用不同数量的特征。有两个基本方法：

1. 开始时没有特征，然后逐个添加特征，知道满足某个终止条件
2. 或者从所有特征开始，然后逐个删除特征，知道满足某个终止条件。

所有这些方法的计算成本都比之前讨论过的方法更高。

其中一种特殊方法是==递归特征消除（RFE）==，它从所有特征开始构建模型，并根据模型设其最不重要的特征，然后使用剩下的特征构建新模型，直到仅剩下预设数量的特征。为了使该方法运行，用于选择的模型需要提供某种确定特征重要性的方法。

```python
from sklean.feature_selection import RFE
select=RFE(
    RandomForestClassifier(n_estimators=100,
                           random_statse=42),
          n_feature_to_select=40)
select.fit(X_train, y_train)
mask=select.get_support()
plt.matshow(mask.reshape(1, -1), cmap="gray r")
plt.xlabel("Sample index")
X_train_ref=select.transform(X_train)
X_test_ref=select.transform(X_test)
score=LogisticRegression().fit(X_test_ref, y_test)
# 也可以利用RFE内使用的模型进行预测，这仅使用被选中的特征集
select.score(X_test, y_test)

```

在不确定使用那些特征作为机器学习算法输入时，自动化特征选择可能特别有用。它有助于减少所需要的特征数量，加快预测速度，或允许可解释性更强的模型。在大多数现实性情况下，使用特征选择不太可能大幅提升性能，但仍是非常有价值的。











