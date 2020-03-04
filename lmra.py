#from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random



#column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']

def make_datasets(path):
    data = pd.read_csv(path + '/auto-mpg.csv')
    datasets = data[['horsepower','weight','mpg']].values
    #datasets = data[['cylinders','displacement','mpg']].values
    error_data_index = np.where(np.isnan(datasets))
    print(error_data_index)
    test = np.arange(3).reshape(1,3)
    train = np.arange(3).reshape(1,3)
    
    for i in range(datasets.shape[0]):
        if random.randint(0,9) == 0:
            test = np.vstack((test,datasets[i]))
        else:
            train = np.vstack((train,datasets[i]))
    error_data_index_test = np.where(np.isnan(test))
    error_data_index_train = np.where(np.isnan(train))
    error_data_index_test = np.append(error_data_index_test,np.zeros((1,3)))
    error_data_index_train = np.append(error_data_index_train,np.zeros((1,3)))
    for index in error_data_index_test:
        test = np.delete(test, index, 0)
    for index in error_data_index_train:
        train = np.delete(train, index, 0)
    return test,train            

    
def main():
    w = np.array([np.nan,np.nan])
    while np.isnan(w[0]) == True:
        [test,train] = make_datasets('datasets')
        #print(train)
        #print(test)
        x = train[:,0:2]
        t = train[:,2]
        #print(x)
        #print(t)
        w = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),t)
        print(w)
    x_test = test[:,0:2]
    t_test = test[:,2]
    horsepower = x[:,0]
    weight = x[:,1]
    X,Y = np.meshgrid(horsepower,weight)
    Z = X*w[0]+Y*w[1]
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.set_xlabel("Cylinders")
    ax.set_xlabel("Horsepower")
    #ax.set_ylabel("Displacement")
    ax.set_ylabel("Weight")
    ax.set_zlabel("W*X")
    ax.plot_wireframe(X,Y,Z)
    ax.scatter(horsepower,weight,t,c='skyblue', label='train')
    ax.scatter(x_test[:,0],x_test[:,1],t_test,c='red', label='test')
    plt.show()
    
    

if __name__ == '__main__':
    main()
