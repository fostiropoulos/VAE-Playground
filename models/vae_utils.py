
import matplotlib.pyplot as plt
import random
import numpy as np

def plot_reconstruction(rec_imgs,results,per_row=10):
    num_imgs=rec_imgs.shape[0]
    overflow=1 if num_imgs%per_row!=0 else 0
    f,a=plt.subplots((num_imgs//per_row+overflow)*2,per_row,figsize=(20,4))
    for i in range(num_imgs):
        img_dim=rec_imgs.shape[-2]
        if rec_imgs.shape[-1]==1:
            reshape_dims=rec_imgs.shape[1:-1]
        else:
            reshape_dims=rec_imgs.shape[1:]
        a[i//per_row*2][i%per_row].imshow(rec_imgs[i%per_row].reshape(reshape_dims))
        a[i//per_row*2+1][i%per_row].imshow(results[i%per_row].reshape(reshape_dims))
    plt.show()

def shuffle_X_y(X,y):
    indices=np.arange(X.shape[0])
    random.shuffle(indices)
    X=X[indices]
    if y is not None:
        y=y[indices]
    return X,y

def plot_loss(loss_monitor,loss_name="", title=""):
    fig=plt.figure(figsize=(10,5))
    for i in range(loss_monitor.shape[-2]):

        plt.plot(loss_monitor[:,i,1].astype(float),label="%s %s"%(loss_name,loss_monitor[0,i,0]))
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.show()
