# 构建原始数据集
# 先用字典形式然后再转成DataFrame
import pandas as pd

rowdata = {"电影名称": ["无问西东","后来的我们","前任3","红海行动","唐人街探案","战狼2"],
           "打斗镜头": [1, 5, 12, 108, 112, 115],
           "接吻镜头": [101, 89, 87, 5, 9, 8],
           "电影类型": ["爱情片", "爱情片", "爱情片", "动作片", "动作片", "动作片"]
           }
# pandas 的DataFrame方法可以利用字典进行初始化为DataFrame格式
movie_data = pd.DataFrame(rowdata)
# print(movie_data)

# 计算新数据点与训练集数据的距离
new_data = [24, 67]
dist = list((((movie_data.iloc[:6, 1:3] - new_data)**2).sum(1))**0.5)
# print(dist)

# 对距离进行升序排序
# 设k=4
k = 4
dist_1 = pd.DataFrame({"dist":dist, "label":(movie_data.iloc[:6, 3])})
dr = dist_1.sort_values(by = "dist")[:k]
# print(dr)

# 确定出现的频率
re = dr.loc[:, 'label'].value_counts()
result = []
result.append(re.index[0])
print(result)



# 封装为函数
""""
函数功能：knn分类器
参数说明：
    inX为新数据坐标
    dataset为旧数据集
    k为超参数
返回：
    result为结果集
"""

def classify01(inX, dataset, k):
    dist = list((((dataset.iloc[:6, 1:3] - inX) ** 2).sum(1)) ** 0.5)
    dist_1 = pd.DataFrame({"dist": dist, "label": (dataset.iloc[:6, 3])})
    dr = dist_1.sort_values(by="dist")[:k]
    re = dr.loc[:, 'label'].value_counts()
    result = []
    result.append(re.index[0])
    print(result)

    return result

classify01([31, 24], movie_data, 4)