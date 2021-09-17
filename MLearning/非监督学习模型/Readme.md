# 非监督学习

_________

[toc]

## 数据集变换

数据集的无监督变换是创建数据新的表示的算法。与数据的原始表示相比，新的表示可能更容易被人或者其他机器学习算法所理解。

无监督变换的一个常见应用是降维，它接受包含许多特征的数据的高位表示，并找到表示该数据的一种新方法。

## 预处理与缩放

`StandardScaler数据预处理方法`确保每个特征的平均值为0，方差为1，使所有特征都位于同一量级。但这种缩放不能保证特征仍和特定的最大值和最小值。

`RobustScaler数据预处理方法`的工作原理与StandardScaler类似，确保每个特征的统计属性都位于同一范围。但RobustScaler使用的是中位数和四分位数，而不是平均值和方差。这样RobustScaler会忽略与其他点有很大不同的数据点（异常值）。

`MinMaxScaler`移动数据使所有特征都刚好位于0到1之间，对于二维数据集来说，所有的数据都包含在X轴0到1Y轴0到1组成的矩阵中。

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# 然后使用fit方法拟合缩放器，只需要提供X_train而不用y_train(因为是无监督学习)
scaler.fit(X_train)
# 对缩放器进行应用
X_train_scaled = scaler.transform(X_train)

```



`Normalizer`用一种完全不同的缩放方法对所有数据点进行缩放，是的特征向量的欧氏距离为1.换言之，他及那个一个数据点投放到一个半径为1的球上。这意味着每个数据点的缩放比例是不同的。如果只有数据的方向重要而特征向量的长度无关紧要是通常采用这种归一化方式。

为了让监督模型能够在测试集上运行，对训练集和测试集应用完全相同的变换时很重要的，即不管训练集还是验证集都是用利用训练集拟合的缩放器进行缩放，而绝不能单独对测试集进行缩放。

在尝试变换训练集时，可以利用`fit_transform`来替代先拟合后变换的过程，通常来说，这会使计算更高效。

## 降维、特征提取和流形学习

### 主成分分析

主成分分析是一种旋转数据集的方法，旋转后的特征在统计上不相关。在做完这种旋转后，通常根据新特征对解释数据的重要性来选择其子集。

PCA最常见的应用就是将高维数据集可视化。利用PCA，我们可以获取主要的相互作用，并得到稍完整的图像。

在对数据进行预处理后应用主成分分析,为了降低数据的维度，我们需要在创建PCA对象是指定想要保留的主成分个数。默认情况下，PCA经旋转数据，但保留所有的主成分。

```python
from sklearn.decomposition import PCA

pca = PCA(n_component = 2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)	# 将数据变换到前两个主成分方向上

```

PCA是一种无监督方法，在寻找旋转方向是没有用到仍和类别星系，他只是观察数据中的相关性。

在拟合过程中，主成分被保存在PCA对象的components属性中。

components中每一行对应一个主成分，他们按照重要性排序。

#### 特征提取

PCA可以应用与特征提取，找到一种数据表示比给定的原始数据更适用于分析。

启用PCA的白化选项，可以将主成分缩放到相同的尺度，比那花后的结果与使用StandardScaler相同。

```python
pca = PCA(n_comoinents = 100, whiten=True, random_state = 0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

```



### 非负矩阵分解

==非负矩阵分解NMF==是一种无监督学习算法，用于提取有用的特征，工作原理类似于PCA也可以用于降维，与PCA相似，我们试图及那个每个数据点写成一些分量的加权求和。在NMF中我们希望分量与系数都为非负数，因此只能用于每个特征都是非负的数据。

NMF使用了随机初始化，根据随机种子的不同可能产生不痛的结果，在相对简单的情况下，所有数据都可以被完美的解释，那么随机性的影响很小，在更加复杂的情况下，影响可能很大。

```python
from sklearn.decomposition import NMF
nfm = NMF(c_components = 15, random_state = 0)
nmf.fit(X_train)
X_train_nmf = nmf.trainsform(X_train)
X_test_nmf = nmf.trianform(X_test)

```

### s-SNE进行流形学习

流形学习算法主要用于可视化，因此很少用于生成2个以上的新特征。

流形学习算法只能用于变换训练集数据，对于探索性数据分析很有用，但是假如最终目标是监督学习则很少使用。背后的思想史找到数据的一个二位表示，尽可能地保持数据点之间的距离，首先给出每个数据点的随机二维表示，然后尝试让原始特征空间中距离较近的点更加靠近，原始特征空间总距离较远的点更加远离。

```python
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
digits_tsne.fit_transform(digits.data)

```



## 聚类分析

### K-means

K均值聚类是最简单也最常用的聚类算法之一。试图代表数据特定区域的簇中心。算法交替执行以下两个步骤：

1. 将每个数据点分配给最近的簇中心
2. 将每个簇中心设置为所分配的数据点的平均值

当簇的分配不再变化时，算法结束。

```python
from sklearn.cluster import KMeans
kmeans = Kmeans(n_clusters=3)
kmeans.fit(X)

```











