import numpy as np
import scipy.stats as stats
from sklearn import datasets
import matplotlib.pyplot as plt


class knnClassifier():

    def __init__(self, k):
        self.k = k

    def fit(self, tr_feat, tr_tar):
        self.tr_feat = tr_feat
        self.tr_tar = tr_tar

    def predict(self, te_feat):
        distance = np.sum(self.tr_feat ** 2, axis=1).T - 2*np.dot(te_feat, self.tr_feat.T)
        ind = np.argsort(distance)[:self.k]
        target_k_nearest = self.tr_tar[ind][:self.k]
        predict = stats.mode(target_k_nearest)[0][0]
        return predict
        
def main():
    np.random.seed(0)
    data = datasets.load_iris()
    feature = data.data
    target = data.target
    k = []
    ap = []
    for i in range(30):
        i = i + 1
        correct = 0
        for j in range(feature.shape[0]):
            tr_feat = np.delete(feature, j, 0)
            te_feat = feature[j]
            tr_tar = np.delete(target, j)
            te_tar = target[j]
            mlc = knnClassifier(i)
            mlc.fit(tr_feat, tr_tar)
            te_predict = mlc.predict(te_feat)
            if (te_predict == te_tar):
                correct += 1    
        ap.append(float(correct) / feature.shape[0])
        k.append(i)
        #print('ap: %f'%ap)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('identification rate')
    ax.set_xlabel('k')
    ax.set_ylabel('ap')
    plt.scatter(k,ap,c='blue',label='knn',alpha = 0.3, s = 10)
    savename = 'graph/knn.png'
    plt.savefig(savename)
    plt.close()

    
if __name__ == '__main__':
    main()
