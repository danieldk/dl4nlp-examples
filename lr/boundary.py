import matplotlib.pyplot as plt
import numpy as np

# From https://github.com/dennybritz/nn-from-scratch
#
# Modified by Daniël de Kok
def plot_decision_boundary(sub,f,X,Y,batch_size,x_range=None, y_range=None):
    padding=0.15
    res=0.01

    #max and min values of x and y of the dataset
    if x_range == None:
        x_min, x_max=X[:,0].min(), X[:,0].max()
    else:
        x_min, x_max = x_range
    if y_range == None:
        y_min,y_max=X[:,1].min(), X[:,1].max()
    else:
        y_min, y_max = y_range

    #range of x's and y's
    x_range=x_max-x_min
    y_range=y_max-y_min

    #add padding to the ranges
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding

    #create a meshgrid of points with the above ranges
    xx,yy=np.meshgrid(np.arange(x_min,x_max,res),np.arange(y_min,y_max,res))

    #use model to predict class at each point on the grid
    #ravel turns the 2d arrays into vectors
    #c_ concatenates the vectors to create one long vector on which to perform prediction
    #finally the vector of prediction is reshaped to the original data shape.


    n_instances = xx.size
    n_batches = n_instances // batch_size
    if n_instances % batch_size != 0:
        n_batches += 1

    batches = np.zeros((n_batches * batch_size, 2))
    batches[:xx.size,0] = xx.ravel()
    batches[:yy.size,1] = yy.ravel()
    batches = batches.reshape((n_batches, batch_size, 2))

    Z = np.zeros(n_batches * batch_size)
    for i in range(n_batches):
        idx = i * batch_size
        Z[idx:idx+batch_size] = f(batches[i])

    Z = Z[:n_instances]

    Z = Z.reshape(xx.shape)

    #plot the contours on the grid
    cs = sub.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    #plot the original data and labels
    sub.scatter(X[:,0], X[:,1], s=35, c=Y, cmap=plt.cm.Spectral)
    
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    
    #plot the contours on the grid
    cs = sub.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    
    #plot the original data and labels
    sub.scatter(X[:,0], X[:,1], s=35, c=Y, cmap=plt.cm.Spectral)
