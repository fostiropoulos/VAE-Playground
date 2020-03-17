import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical

from vqvae.models.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from vqvae.models.vq_vae import VQVAE
from tqdm import tqdm

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

    def __init__(self,image_size,channels,D,K,L,commitment_beta=0.25,lr=0.002,c=1, num_convs=4,num_fc=2, use_mine=True):
        self.use_mine=use_mine
        super().__init__(image_size,channels, D,K,L,commitment_beta,lr,c, num_convs,num_fc)


    def _loss_init(self,inputs,outputs):
        if self.use_mine:
            #
            x_sample=tf.reshape(self.z[:,:,:,:self.D],(-1,2))
            y_sample=tf.reshape(self.z[:,:,:,self.D:],(-1,2))
            x_sample1, x_sample2 = tf.split(x_sample, 2)
            y_sample1, y_sample2 = tf.split(y_sample, 2)
            joint_sample = tf.concat([x_sample1, y_sample1], axis=1)
            marg_sample =  tf.concat([x_sample2, y_sample1], axis=1)
            labels = tf.concat([tf.ones(joint_sample.shape[0]), tf.zeros(marg_sample.shape[0])])

            self.mine_model = Mine()
            model=self.mine_model
            logits = model(tf.concat([joint_sample,marg_sample])
            #self.marginal=model(marg_sample)
            # log_mean_Exp
            self.mine_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits) #tf.clip_by_value(-tf.reduce_mean(self.joint) +tf.math.log(tf.reduce_mean(tf.exp(self.marginal))), 0, 1000000)
            opt=tf.train.AdamOptimizer(learning_rate=0.01)
            gvs = opt.compute_gradients(self.mine,var_list=model.trainable_variables)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.mine_train = opt.apply_gradients(capped_gvs)
        
        else:
            self.mine=tf.constant(0.)
        """
        VAE-LOSS
        """
        
        reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)
        self.reconstr_loss=tf.reduce_mean(reconstr_loss)

        """
        VQ-LOSS
        """
        self.loss=self.mine*1 + self.reconstr_loss +   self.vq_loss + self.commitment_loss * self.commitment_beta 

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

    def partial_fit(self,X,X_test=None, batch_size=64):

        np.random.shuffle(X)
        num_batches=X.shape[0]//batch_size
        train_out=[]
        with tqdm(range(num_batches)) as t:
            for i in t:
                X_batch=X[i*batch_size:(i+1)*batch_size]
                X_images=self.read_batch(X_batch)
                if self.use_mine:
                    self.mine_model.trainable=False
                loss,_=self.sess.run([self.loss]+[self.train],feed_dict=self.get_feed_dict(X_images))
                
                losses=self.sess.run( list(self.losses.values()),
                                        feed_dict=self.get_feed_dict(X_images))
                


                if self.use_mine:
                    self.mine_model.trainable=True
                    _=self.sess.run([self.mine_train],feed_dict=self.get_feed_dict(X_images))
                desc=""
                loss_monitor=list(zip(self.losses.keys(),losses))
                for name,val in loss_monitor:
                    desc+=("%s: %.2f\t"%(name,val))
                t.set_description(desc)
                train_out.append(loss_monitor)

        if(X_test is not None):
            
            np.random.shuffle(X_test)
            X_images=self.read_batch(X_test[:batch_size])
            losses=self.sess.run( list(self.losses.values()),
                                    feed_dict=self.get_feed_dict(X_images))
            test_out=list(zip(self.losses.keys(),losses))
        else:
            test_out=[.0]*len(self.losses)

        return train_out, [test_out]
