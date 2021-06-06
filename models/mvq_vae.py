import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from vqvae.models.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from vqvae.models.cnn_vae import cnnVAE

class MVQVAE:

    def __init__(self,image_size,channels, D,K,L,commitment_beta=1,lr=0.02,c=1, num_convs=4,num_fc=2,spatial=False, concat=False,joint=False, penalize_perplexity=False, use_mse=False):
        self.commitment_beta=commitment_beta
        # dimension of quantization vector
        self.D=D
        # number of qunatization vectors
        self.K=K
        #self.K2=K2
        # number of codebooks
        self.L=L
        # vq-layer output
        self.use_mse=use_mse
        self.z=None
        self.joint=joint
        self.image_size=image_size
        self.channels=channels
        self.penalize_perplexity=penalize_perplexity
        self.spatial=spatial
        self.concat=concat 

        self.display_layer=None

        self.num_hiddens=255
        self.num_res_hiddens=64

        self.lr=lr
        self.num_convs=num_convs

        self.X=None

        self.inputs=None
        self.outputs=None

        self.embedding_dim=D
        self.inputs=None
        self.outputs=None

        self.losses=[]
        self.summary_op=None
        self.build_model()
        self.saver = tf.train.Saver()
        self.start_session()
        

    def start_session(self):
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
        self.summary_op=tf.summary.merge_all()

    def save(self,file):
        self.saver.save(self.sess, file)

    def load(self,file):
        self.saver.restore(self.sess, file)

    def quantize(self, embeddings,encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        w = tf.transpose(embeddings, [1, 0])
        # TODO(mareynolds) in V1 we had a validate_indices kwarg, this is no longer
        # supported in V2. Are we missing anything here?
        return tf.nn.embedding_lookup(w, encoding_indices)
    
    def vq_layer(self,inputs,D,K,name="lookup_table",init=tf.truncated_normal_initializer(mean=0., stddev=.1)):

        embeddings = tf.get_variable(name,shape=[D, K],dtype=tf.float32,initializer=init)#
        #shape=[self.D, self.K],
        print(embeddings.shape)
        z_e=inputs
        flat_inputs = tf.reshape(inputs, [-1, D])
        distances = (
            tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
            2 * tf.matmul(flat_inputs, embeddings) +
            tf.reduce_sum(embeddings**2, 0, keepdims=True))
        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self.K)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = self.quantize(embeddings,encoding_indices)
        e_k=quantized


        # Straight Through Estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        #avg_probs = tf.reduce_mean(encodings, 0)
        avg_probs = tf.reduce_mean(tf.nn.softmax(distances,axis=-1),0)

        perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                        tf.math.log(avg_probs + 1e-10)))

        commitment_loss = tf.reduce_mean((z_e - tf.stop_gradient(e_k)) ** 2)
        vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - e_k) ** 2)


        distance=(
            tf.reduce_sum(tf.transpose(embeddings,[1,0])**2, 1, keepdims=True) -
            2 * tf.matmul(tf.transpose(embeddings,[1,0]), embeddings) +
            tf.reduce_sum(embeddings**2, 0, keepdims=True))

        return {"outputs":quantized,"perplexity":perplexity,"distance":distance,"commitment_loss":commitment_loss,"vq_loss":vq_loss,"encodings":encoding_indices}
        

    def _z_init(self,inputs):
        with tf.variable_scope("vq"):
            self.vq_loss=tf.constant([0.])
            self.commitment_loss=tf.constant([0.]) 
            x=inputs   
            #self.mu=(tf.keras.layers.Conv2D(self.D,3,strides=(1,1),padding="same", activation=None, name="dec_conv_mu"))(x)
            #self.sigma=(tf.keras.layers.Conv2D(self.D,3,strides=(1,1),padding="same", activation=None, name="dec_conv_std"))(x)
        

            #self.vq_inputs=cnnVAE.sample(self.mu,self.sigma)#tf.split(inputs,self.L,axis=-1)
            print(inputs)
            if self.joint:
                pre_vq=(tf.keras.layers.Conv2D(self.D,3,strides=(1,1),padding="same", activation=None, name="dec_conv"))(x)
                self.vq_inputs=list([pre_vq])
            else:
                if self.spatial:
                    pre_vq=(tf.keras.layers.Conv2D(self.D*self.L,3,strides=(1,1),padding="same", activation=None, name="dec_conv"))(x)
                    self.vq_inputs=tf.split(tf.reshape(pre_vq,(-1,int(pre_vq.shape[1])**2,self.D*self.L)),self.L,axis=1)

                else:
                    self.vq_inputs=tf.split((tf.keras.layers.Conv2D(self.D*self.L,3,strides=(1,1),padding="same", activation=None, name="dec_conv"))(x),self.L,axis=-1)
            print("vq_inputs %s"%self.vq_inputs)
            z=[]
            encodings=[]
            self.perplexity=[]
            
            for i in range(self.L):
                out=self.vq_layer(self.vq_inputs[i] if not self.joint else self.vq_inputs[0],self.D*self.L if self.spatial else self.D,self.K,name="lookup_table_0_%d"%i)#init=inits[i])
                self.vq_loss+=out["vq_loss"]
                self.commitment_loss+=out["commitment_loss"]
                
                self.distance=out["distance"]
                #out=self.vq_layer(out["outputs"],self.D,self.K2,name="lookup_table_1_%d"%i)#init=inits[i])
                #self.vq_loss+=out["vq_loss"]
                #self.commitment_loss+=out["commitment_loss"]

                
                z.append(out["outputs"])
                encodings.append(tf.cast(tf.expand_dims(out["encodings"],-1),tf.float32))

            #self.z=out["outputs"]
            #z.append(out["outputs"])
            #encodings.append(tf.cast(tf.expand_dims(out["encodings"],-1),tf.float32))
       
            if self.joint:
                if self.concat:
                    print(z)
                    self.z=tf.concat(z,axis=-1)
                else:
                    self.z=tf.math.add_n(z,axis=-1)
            else:
                if self.spatial:
                    self.z=tf.reshape(tf.concat(z,axis=1),[-1]+list(np.array(pre_vq.shape[1:]).astype(np.int32)))
                else:
                    self.z=tf.concat(z,axis=-1)
            self.encodings=tf.concat(encodings,axis=-1)
            print("encodings",  self.encodings.shape)
            print("z",self.z.shape)

            #self.z=out["outputs"]#tf.concat(z,axis=-1)
            self.vq_loss=tf.reshape( self.vq_loss,[])
            self.commitment_loss=tf.reshape( self.commitment_loss,[])
        return self.z


    def _loss_init(self,inputs,outputs):

        """
        VAE-LOSS
        """
        print(inputs.shape)
        print(outputs.shape)
        if not self.use_mse:
            reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)
        else:
            reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)

        self.reconstr_loss=tf.reduce_mean(reconstr_loss)

        """
        VQ-LOSS
        """
        self.loss=self.reconstr_loss +   self.vq_loss + self.commitment_loss * self.commitment_beta + 0.001#*tf.reduce_mean(self.distance)


        self.losses={}
        self.losses["total loss"]=self.loss
        self.losses["VQ"]=self.vq_loss
        self.losses["Commitment"]=self.commitment_loss
        self.losses["reconstruction"]=self.reconstr_loss


        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        tf.summary.scalar("total_loss", self.loss)

    def _train_init(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.optimizer_distance = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train = [self.optimizer.minimize(self.loss)]
        if(self.penalize_perplexity):
            self.train.append(self.optimizer_distance.minimize(-self.perplexity))#-tf.reduce_mean(self.embeddings_distance))]#

    def save(self,file):
        self.saver.save(self.sess, file)

    def load(self,file):
        self.saver.restore(self.sess, file)


    def _encoder_init(self,x):

        with tf.variable_scope("encoder"):
            num_hiddens=self.num_hiddens
            num_res_hiddens=self.num_res_hiddens
            embedding_dim=self.embedding_dim
        

            i=0
            conv=(tf.keras.layers.Conv2D(num_hiddens//2,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(x)

            for j in range(self.num_convs):
                i+=1
                conv=(tf.keras.layers.Conv2D(num_hiddens,4,strides=(2,2),padding="same", activation=tf.nn.relu, name="enc_conv_%d"%i))(conv)

            for j in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_enc_conv_%d"%j))(conv)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="res_enc_conv_%d"%j))(first_res)
                conv+=second_res # resnet v1
            conv=tf.nn.relu(conv)
            #conv=(tf.keras.layers.Conv2D(embedding_dim*self.L,1,strides=(1,1),padding="same", activation=None, name="to_vq"))(conv)
            #conv=(tf.keras.layers.Conv2D(self.D,1,strides=(1,1),padding="same", activation=None, name="to_vq"))(conv)
            #self.orig_shape=(-1,conv.shape[1],conv.shape[2],conv.shape[3])
            #self.orig_shape=conv.shape
            #print(self.orig_shape)
            #flat=tf.keras.layers.Flatten()(conv)
            #d=tf.keras.layers.Dense(10,activation="relu")(flat)
            #out=tf.keras.layers.Dense(2,activation="relu")(d)
        return conv#out#tf.reshape(conv,(-1,np.prod(conv.shape[1:])))

    def _decoder_init(self,x):
        with tf.variable_scope("decoder"):
            num_hiddens=self.num_hiddens
            num_res_hiddens=self.num_res_hiddens
            embedding_dim=self.embedding_dim

            #x=tf.keras.layers.Dense(np.prod(self.orig_shape[1:]),activation="relu")(x)
            print("Decoder-Input: %s"%x.shape)
            deconv=x#tf.reshape(x,self.orig_shape)
            

            conv=(tf.keras.layers.Conv2D(num_hiddens,3,strides=(1,1),padding="same", activation=None, name="dec_conv_0"))(deconv)

            for j in range(2):
                first_res=(tf.keras.layers.Conv2D(num_res_hiddens,3,strides=(1,1),padding="same", activation=tf.nn.relu, name="dec_res_conv_%d"%j))(conv)
                second_res=(tf.keras.layers.Conv2D(num_hiddens,1,strides=(1,1),padding="same", activation=tf.nn.relu, name="dec_res_conv_2_%d"%j))(first_res)
                conv+=tf.nn.relu(second_res) # resnet v1
            i=0
            deconv=conv
            for j in range(self.num_convs):
                deconv = tf.keras.layers.Conv2DTranspose( num_hiddens//2, 4, strides=(2, 2), padding="same", activation=tf.nn.relu, name="dec_deconv_%d"%(i)) (deconv)
                i+=1
            last_layer = tf.keras.layers.Conv2DTranspose( self.channels*(256 if not self.use_mse else 1), 4, strides=(2, 2), padding="same", activation=None, name="dec_deconv_%d"%(i)) (deconv)
        
        print("last layer: %s"%last_layer.shape)

        return tf.reshape(last_layer,[-1,self.image_size,self.image_size,self.channels,(256 if not self.use_mse else 1)])

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
