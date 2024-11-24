from sklearn import datasets
df=datasets.load_iris()
x=df["data"]
y=df["target"]
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=10000)
print(lr.fit(x,y))
print(lr.score(x,y))
