
import pickle
from sklearn import datasets
import joblib
import numpy as np
import pandas as pd
class Perceptron():
    
    def __init__(self,eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                #print(xi, target)
                update = self.eta*(target-self.predict(xi))
                #print(update)
                self.w_[1:] += update*xi
                self.w_[0] += update
                #print(self.w_)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X)>=0.0,1,-1)


iris = datasets.load_iris()

X, y = iris.data, iris.target

df = pd.DataFrame(X)
df['y'] = y
df= df[df['y'] != 2]

df.replace(
      to_replace=(0, 1)
    , value = (-1,1), inplace = True
    )

X = np.array(df.iloc[:,:4], dtype=np.float32)
y = np.array(df.iloc[:,-1], dtype=np.float32)



clf = Perceptron()
clf.fit(X, y)


saved_model = pickle.dumps(clf)


joblib.dump(clf, 'model.pkl')