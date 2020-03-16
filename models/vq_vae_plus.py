import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical

from vqvae.models.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from vqvae.models.vq_vae import VQVAE


class Mine(tf.keras.Model):
    def __init__(self):
        super(Mine, self).__init__()
        
        H=1280

        # propagate the forward pass
        self.dense=[]
        for l in range(4):
            self.dense.append(tf.keras.layers.Dense(H, activation='relu'))
            self.dense.append(tf.keras.layers.Dropout(0.3))
        self.logs=tf.keras.layers.Dense(1)
        
    def call(self, x):
        out=x
        for l in self.dense:
            out=l(out)
        
        return self.logs(out)

class VQVAEPlus(VQVAE):

    def __init__(self,image_size,channels,D,K,L,commitment_beta=1,lr=0.002,c=1, num_convs=4,num_fc=2, mine=True):
        self.mine=mine
        super().__init__(image_size,channels, D,K,L,commitment_beta,lr,c, num_convs,num_fc)


    def _loss_init(self,inputs,outputs):

        x_sample=tf.reshape(self.encodings[:,:,:,:1],(-1,2))
        y_sample=tf.reshape(self.encodings[:,:,:,1:],(-1,2))
        x_sample1, x_sample2 = tf.split(x_sample, 2)
        y_sample1, y_sample2 = tf.split(y_sample, 2)
        joint_sample = tf.concat([x_sample1, y_sample1], axis=1)
        marg_sample =  tf.concat([x_sample2, y_sample1], axis=1)

        model = Mine()
        self.joint = model(joint_sample)
        self.marginal=model(marg_sample)
        
        
        mine=-tf.reduce_mean(self.joint) +tf.math.log(tf.reduce_mean(tf.exp(self.marginal)))
        self.mine=mine if self.mine else tf.constant(0.)
        """
        VAE-LOSS
        """

        reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)
        self.reconstr_loss=tf.reduce_mean(reconstr_loss)

        """
        VQ-LOSS
        """
        self.loss=self.mine*0.1 + self.reconstr_loss +   self.vq_loss + self.commitment_loss * self.commitment_beta 

        self.losses={}
        self.losses["total loss"]=self.loss
        self.losses["VQ"]=self.vq_loss
        self.losses["Commitment"]=self.commitment_loss
        self.losses["reconstruction"]=self.reconstr_loss
        for i,perplexity in enumerate(self.perplexity):
            self.losses["perplexity C_%d"%i]=perplexity
        self.losses["MINE"]=self.mine


        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        tf.summary.scalar("MINE", self.mine)
        tf.summary.scalar("total_loss", self.loss)
