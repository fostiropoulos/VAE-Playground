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
            x_sample=tf.reshape(self.vq_inputs[:,:,:,:1],(-1,1))
            y_sample=tf.reshape(self.vq_inputs[:,:,:,1:],(-1,1))
            x_sample1, x_sample2 = tf.split(x_sample, 2)
            y_sample1, y_sample2 = tf.split(y_sample, 2)
            x_sample1, x_sample2 = tf.split(x_sample, 2)
            y_sample1, y_sample2 = tf.split(y_sample, 2)
            joint_sample = tf.concat([x_sample1, y_sample1], axis=1)
            marg_sample =  tf.concat([x_sample1, y_sample2], axis=1)
            ones=tf.ones((tf.shape(joint_sample)[0],1))
            zeros=tf.zeros((tf.shape(marg_sample)[0],1))
            labels = tf.concat([ones, zeros],axis=0)
            print(labels.shape)

            self.mine_model = Mine()
            model=self.mine_model
            self.mine_logits = model(tf.concat([joint_sample,marg_sample],axis=0))
            
            #self.marginal=model(marg_sample)
            # log_mean_Exp
            z=tf.split(self.mine_logits,2,axis=0)[0]
            self.mine=tf.clip_by_value(tf.reduce_mean(tf.math.log((tf.math.sigmoid(z))/(1-tf.math.sigmoid(z)))),-1,10000)
            self.mine_loss=tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=self.mine_logits))#,-10000,10000) 
            #tf.clip_by_value(-tf.reduce_mean(self.joint) +tf.math.log(tf.reduce_mean(tf.exp(self.marginal))), 0, 1000000)
            opt=tf.train.AdamOptimizer(learning_rate=0.001)
            
            gvs = opt.compute_gradients(self.mine_loss,var_list=model.trainable_variables)
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
        self.loss=self.mine*1 + self.reconstr_loss + self.commitment_loss * self.commitment_beta   + self.vq_loss 

        self.losses={}
        self.losses["total loss"]=self.loss
        self.losses["VQ"]=self.vq_loss
        self.losses["Commitment"]=self.commitment_loss
        self.losses["reconstruction"]=self.reconstr_loss
        #for i,perplexity in enumerate(self.perplexity):
        #    self.losses["perplexity C_%d"%i]=perplexity
        self.losses["MINE"]=self.mine


        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        tf.summary.scalar("MINE", self.mine)
        tf.summary.scalar("total_loss", self.loss)

    def plot_codebooks(self,test_data,title=""):
        from matplotlib.pyplot import figure
        import matplotlib.pyplot as plt
        from matplotlib import cm

        figure( figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')

        tables=[]
        L=self.L
        for i in range(L):
            tables.append(self.sess.run(self.sess.graph.get_tensor_by_name("vq/lookup_table_%d:0"%(i))))
        codebooks=np.transpose(np.array(tables),[0,2,1])
        for i in range(L):
            x,y=codebooks[i,:,0],codebooks[i,:,1]
            """
            points=np.array(list(zip(x,y)))
            vor = Voronoi(points)
            regions, vertices = voronoi_finite_polygons_2d(vor)
            # colorize
            for region in regions:
                polygon = vertices[region]
                plt.fill(*zip(*polygon), alpha=0.4)

            plt.plot(points[:,0], points[:,1], 'ko')
            plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
            plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
            """
            #plt.show()
            plt.scatter(x,y,label='Codebook Centroids %d'%i,s=10)

        vq_inputs=self.sess.run([self.vq_inputs], feed_dict={self.X:test_data})[0]
        #x,y=z_e.reshape(-1,2)[:,0],z_e.reshape(-1,2)[:,1]
        viridis = cm.get_cmap('viridis', 10)
        #i=vq_inputs[0]
        #print(i.shape)
        #x,y=i[:,:,0],i[:,:,1]
        #plt.scatter(x,y,c='orange', label="Test Sample input to VQ",s=.1)
        #i=vq_inputs[1]
        i=vq_inputs
        x,y=i[:,:,:,0],i[:,:,:,1]
        plt.scatter(x,y,c='blue', label="Test Sample input to VQ",s=.1)
        plt.xlabel("centroid d=0")
        plt.ylabel("centroid d=1")
        plt.title(title)
        plt.legend()
        plt.show()

    def partial_fit(self,X,X_test=None, batch_size=64):

        np.random.shuffle(X)
        num_batches=X.shape[0]//batch_size
        train_out=[]

        np.random.shuffle(X_test)
        X_images=self.read_batch(X_test[:batch_size])
        #self.plot_codebooks(X_images)
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
            #self.plot_codebooks(X_images)
        else:
            test_out=[.0]*len(self.losses)

        return train_out, [test_out]
