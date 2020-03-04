import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class k_means():

    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.rep_point = np.zeros((k,data.shape[1]))
        self.result = np.zeros(data.shape[0])
        random = np.random.randint(0,data.shape[0]-1,k)
        for i in range(k):
            self.rep_point[i] = data[random[i]]

    def clustering(self):
        delta = 1.0
        while delta > 0.01:
            dist = np.sum(self.rep_point ** 2, axis=1, keepdims=True).T - 2 * np.dot(self.data, self.rep_point.T)
            self.result = np.argsort(dist)[:,0].T
            old_rep_point = self.rep_point.copy()
            for i in range(self.k):
                int = np.where(self.result == i)
                tem = self.data[int]
                self.rep_point[i] = np.mean(tem, axis=0)
            delta = np.sum(np.sum((self.rep_point - old_rep_point) ** 2, axis=1))
        return self.result, self.rep_point

def main():
    data = datasets.load_iris().data
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    for k in range(2,6):
        np.random.seed(0)
        clustering_machine = k_means(k,data)
        [result,rep_point] = clustering_machine.clustering()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('k-means(k={})'.format(k))
        ax.set_xlabel('Sepal(length+width)')
        ax.set_ylabel('Petal(length+width)')
        for i in range(k):
            int = np.where(result==i)
            tem = data[int].T
            plt.scatter(rep_point[i][0]+rep_point[i][1],rep_point[i][2]+rep_point[i][3],c="k",s=300,marker="*")
            plt.scatter(tem[0]+tem[1],tem[2]+tem[3],c=colorlist[i], label='label{}'.format(i))
        savename = 'graph/k-means(k={})'.format(k)
        plt.savefig(savename)
        plt.close()
          


if __name__ == '__main__':
    main()
                    
                    
                
                
