import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from vqvae.models.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from vqvae.models.cnn_vae import cnnVAE

class VQVAE(cnnVAE):

    def __init__(self,image_size,channels, D,K,L,commitment_beta=0.25,lr=0.002,c=1, num_convs=4,num_fc=2):
        self.commitment_beta=commitment_beta
        # dimension of quantization vector
        self.D=D
        # number of qunatization vectors
        self.K=K
        # number of codebooks
        self.L=L
        # vq-layer output
        self.z=None

        super().__init__(image_size,channels,D,lr,c, num_convs,num_fc)

    def quantize(self, embeddings,encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        w = tf.transpose(embeddings, [1, 0])
        # TODO(mareynolds) in V1 we had a validate_indices kwarg, this is no longer
        # supported in V2. Are we missing anything here?
        return tf.nn.embedding_lookup(w, encoding_indices)
    
    def vq_layer(self,inputs,name="lookup_table"):

        embeddings = tf.get_variable(name,dtype=tf.float32,shape=[self.D, self.K],initializer=tf.truncated_normal_initializer(mean=0., stddev=.1))
        z_e=inputs
        flat_inputs = tf.reshape(inputs, [-1, self.D])
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
        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                        tf.math.log(avg_probs + 1e-10)))

        commitment_loss = self.commitment_beta * tf.reduce_mean((z_e - tf.stop_gradient(e_k)) ** 2)
        vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - e_k) ** 2)
        return {"outputs":quantized,"perplexity":perplexity,"commitment_loss":commitment_loss,"vq_loss":vq_loss,"encodings":encoding_indices}
        

    def _z_init(self,inputs):
        with tf.variable_scope("vq"):
            self.vq_loss=tf.constant([0.])
            self.commitment_loss=tf.constant([0.])            
            self.vq_inputs=inputs

            z=[]
            encodings=[]
            self.perplexity=[]
            
            for i in range(self.L):

                out=self.vq_layer(inputs,name="lookup_table_%d"%i)
                
                
                self.vq_loss+=out["vq_loss"]

                self.commitment_loss+=out["commitment_loss"]
                
                self.perplexity.append(out["perplexity"])
                

                z.append(out["outputs"])
                encodings.append(tf.cast(tf.expand_dims(out["encodings"],-1),tf.float32))

            self.encodings=tf.concat(encodings,axis=-1)
            self.z=tf.concat(z,axis=-1)
            self.vq_loss=tf.reshape( self.vq_loss,[])
            self.commitment_loss=tf.reshape( self.commitment_loss,[])
        return self.z


    def _loss_init(self,inputs,outputs):

        """
        VAE-LOSS
        """

        reconstr_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs,logits=outputs)
        self.reconstr_loss=tf.reduce_mean(reconstr_loss)

        """
        VQ-LOSS
        """

        self.loss=self.reconstr_loss +   self.vq_loss + self.commitment_loss * self.commitment_beta


        self.losses={}
        self.losses["total loss"]=self.loss
        self.losses["VQ"]=self.vq_loss
        self.losses["Commitment"]=self.commitment_loss
        self.losses["reconstruction"]=self.reconstr_loss


        tf.summary.scalar('vq_loss', self.vq_loss )
        tf.summary.scalar('commitment_loss', self.commitment_loss )
        tf.summary.scalar("reconstr_loss", self.reconstr_loss)
        tf.summary.scalar("total_loss", self.loss)
