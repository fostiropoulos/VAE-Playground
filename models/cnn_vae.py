import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as numpy
from vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

class cnnVAE:

    def __init__(self,image_size,channels,z_dim,lr=0.0002,c=0.2, num_convs=2,num_fc=2):

        self.num_hiddens=256
        self.num_res_hiddens=32
        self.embedding_dim=z_dim
        self.lr=lr
        self.image_size=image_size
        self.channels=channels
        # kl coefficient 
        self.c=c
        self.num_convs=num_convs
        self.num_fc=num_fc 
        self.X=None
        self.mu=None
        self.sigma=None
        self.losses=[]
        self.fc=[]
        self.conv_layers=[]
        self.dec_fc=[]
        self.deconv_layers=[]
        self.display_layer=None
        self.inputs=None
        self.outputs=None
        self.summary_op=None

        self.build_model()
        self.saver = tf.train.Saver()
        self.start_session()


    def start_session(self):
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
        self.summary_op=tf.summary.merge_all()

    def _loss_init(self,inputs,outputs):
        # applies sigmoid inside the function. We must provide logits and labels ONLY
        reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)
        # get the mean for each sample for the reconstruction error
        self.reconstr_loss=tf.reduce_mean(reconstr_loss)
        self.latent_loss= -tf.reduce_mean(0.5 * (1 + self.sigma - self.mu**2 - tf.exp(self.sigma)))

        self.loss=self.reconstr_loss+ self.latent_loss * self.c
        self.losses={}
        self.losses["total loss"]=self.loss
        self.losses["KL"]=self.latent_loss
        self.losses["reconstruction"]=self.reconstr_loss
        tf.summary.scalar("total_loss", self.loss)
        tf.summary.scalar("latent_loss", self.latent_loss)
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)

    def _encoder_init(self,x):

        with tf.variable_scope("encoder"):
            num_hiddens=self.num_hiddens
            num_res_hiddens=self.num_res_hiddens
            embedding_dim=self.embedding_dim
        

            i=0
            conv=(tf.keras.layers.Conv2D(num_hiddens//2,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(x)
            i+=1
            conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            i+=1
            conv=(tf.keras.layers.Conv2D(num_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)
            for j in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_enc_conv_%d"%j))(conv)
                self.conv_layers.append(first_res)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_enc_conv_%d"%j))(first_res)
                self.conv_layers.append(second_res)
                conv+=second_res # resnet v1
            conv=tf.nn.relu(conv)
            conv=(tf.keras.layers.Conv2D(embedding_dim,1,strides=(1,1),padding="same", activation=None, name="to_vq"))(conv)
            #self.orig_shape=(-1,conv.shape[1],conv.shape[2],conv.shape[3])
            #self.orig_shape=conv.shape
            #print(self.orig_shape)
            
            
        return conv#tf.reshape(conv,(-1,np.prod(conv.shape[1:])))

    def _decoder_init(self,x):
        with tf.variable_scope("decoder"):
            num_hiddens=self.num_hiddens
            num_res_hiddens=self.num_res_hiddens
            embedding_dim=self.embedding_dim
            print(x.shape)
            deconv=x#tf.reshape(x,self.orig_shape)
            

            conv=(tf.keras.layers.Conv2D(num_hiddens,3,strides=(1,1),padding="same", activation=None, name="dec_conv_0"))(deconv)

            for j in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="dec_res_conv_%d"%j))(conv)
                self.conv_layers.append(first_res)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="dec_res_conv_2_%d"%j))(first_res)
                self.conv_layers.append(second_res)
                conv+=tf.nn.relu(second_res) # resnet v1
            i=0
            deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i+1)) (conv)
            i+=1
            last_layer = tf.keras.layers.Conv2DTranspose( self.channels*256, 4, strides=(2, 2), padding="same", activation=None, name="dec_deconv_%d"%(i+1)) (deconv)
            
        return tf.reshape(last_layer,[-1,self.image_size,self.image_size,self.channels,256])

    def sample(mu,sigma):
        eps_shape = tf.shape(mu)
        eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
        z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(sigma)), eps), name="z")
        return z

    def _z_init(self,x):

        self.mu=(tf.keras.layers.Conv2D(self.embedding_dim,3,strides=(1,1),padding="same", activation=None, name="dec_conv_0"))(x)
        self.sigma=(tf.keras.layers.Conv2D(self.embedding_dim,3,strides=(1,1),padding="same", activation=None, name="dec_conv_0"))(x)
        return cnnVAE.sample(self.mu,self.sigma)

    def build_model(self):

        tf.reset_default_graph()

        #images
        self.X=tf.placeholder(tf.int32,[None,self.image_size,self.image_size,self.channels], name="x_input")
        enc_out=self._encoder_init(tf.cast(self.X,tf.float32)/255-0.5)
        self.z=self._z_init(enc_out)
        self.outputs=self._decoder_init(self.z)
        
        #display layer ONLY
        self.display_layer=tf.cast(tf.math.argmax(tf.nn.softmax(self.outputs,name="output"),axis=-1),tf.int32)
        

        hstack=tf.cast(tf.concat(([self.display_layer,self.X]),axis=1),tf.float32)
        tf.summary.image("reconstruction",hstack)

        # flatten the inputs
        inputs=tf.reshape(self.X,(-1,self.channels*self.image_size**2), name="inputs")

        # flatten the outputs
        outputs=tf.reshape(self.outputs,(-1,self.channels*self.image_size**2,256),name="outputs")
        self._loss_init(inputs,outputs)
        self._train_init()

    def _train_init(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train = [self.optimizer.minimize(self.loss)]

    def sample(mu,sigma):
        eps_shape = tf.shape(mu)
        eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
        z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(sigma)), eps), name="z")
        return z

    def reconstruct(self,rec_imgs):
        return self.sess.run(self.display_layer,feed_dict={self.X:rec_imgs})

    def plot_reconstruction(self,rec_imgs):
        plot_reconstruction(rec_imgs,self.reconstruct(rec_imgs))


    def get_feed_dict(self,X_batch):
        feed_dict={self.X:X_batch}
        return feed_dict

    def read_batch(self,paths):
        
        imgs=np.zeros([len(paths),self.image_size,self.image_size,self.channels],dtype=np.int32)

        mode="RGB" if(self.channels==3) else "L"
        for i,img in enumerate(paths):
            _img=Image.open(img).convert(mode)
            imgs[i]=np.array(_img.resize((self.image_size,self.image_size))).reshape(self.image_size,self.image_size,self.channels)

        return imgs

    def partial_fit(self,X,X_test=None, batch_size=64):

        np.random.shuffle(X)
        num_batches=X.shape[0]//batch_size
        train_out=[]
        with tqdm(range(num_batches)) as t:
            for i in t:
                X_batch=X[i*batch_size:(i+1)*batch_size]
                X_images=self.read_batch(X_batch)
                loss,_=self.sess.run([self.loss]+[self.train],feed_dict=self.get_feed_dict(X_images))
                
                losses=self.sess.run( list(self.losses.values()),
                                        feed_dict=self.get_feed_dict(X_images))
                loss_monitor=list(zip(self.losses.keys(),losses))
                desc=""
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


    def fit(self,X,X_test=None,epochs=20,batch_size=64, plot=True, verbose=True,log_dir=None):
        train_monitor=[]
        test_monitor=[]


        np.random.shuffle(X)
        train_batch_sum=self.read_batch(X[:10])
        writer_test=None

        if not X_test is None:
            x_test_indices=list(range(X_test.shape[0]))
            np.random.shuffle(X_test)
            test_batch_sum=self.read_batch(X_test[x_test_indices[:10]])
            writer_test = tf.summary.FileWriter(log_dir+"/test",self.sess.graph) if log_dir else None
        writer_train = tf.summary.FileWriter(log_dir+"/train",self.sess.graph) if log_dir else None

        for epoch in range(epochs):
            train_out,test_out=self.partial_fit(X,X_test, batch_size)
            train_monitor+=train_out
            test_monitor+=test_out
            
            if plot:
                self.plot_reconstruction(train_batch_sum)

            if writer_train!=None:
                summary=self.sess.run(self.summary_op, feed_dict={self.X:train_batch_sum})
                writer_train.add_summary(summary, epoch)
                writer_train.flush()
                if writer_test!=None:
                    summary=self.sess.run(self.summary_op, feed_dict={self.X:test_batch_sum})
                    writer_test.add_summary(summary, epoch)
                    writer_test.flush()

            if(verbose):
                # print the last one only
                print("Train {}".format(train_monitor[-1]))
                print("Test {}".format(test_monitor[-1]))
        if plot:
            #ignore epochs by indexing 1:
            plot_loss(np.array(train_monitor),"train")
            plot_loss(np.array(test_monitor),"test")
        return train_monitor,test_monitor
