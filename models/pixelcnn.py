import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from tensorflow.keras.utils import to_categorical
from vqvae.models.vae_utils import plot_reconstruction,shuffle_X_y,plot_loss
from vqvae.models.cnn_vae import cnnVAE
from tqdm import tqdm
from PIL import Image
# Networks used in Pixel CNN


def get_weights(shape, name, horizontal, mask_mode='noblind', mask=None):
    weights_initializer =tf.random_uniform_initializer()
    W = tf.get_variable(name, shape, tf.float32, weights_initializer)

    '''
        Use of masking to hide subsequent pixel values 
    '''
    if mask:
        filter_mid_y = shape[0]//2
        filter_mid_x = shape[1]//2
        mask_filter = np.ones(shape, dtype=np.float32)
        if mask_mode == 'noblind':
            if horizontal:
                # All rows after center must be zero
                mask_filter[filter_mid_y+1:, :, :, :] = 0.0
                # All columns after center in center row must be zero
                mask_filter[filter_mid_y, filter_mid_x+1:, :, :] = 0.0
            else:
                if mask == 'a':
                    # In the first layer, can ONLY access pixels above it
                    mask_filter[filter_mid_y:, :, :, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    mask_filter[filter_mid_y+1:, :, :, :] = 0.0

            if mask == 'a':
                # Center must be zero in first layer
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        else:
            mask_filter[filter_mid_y, filter_mid_x+1:, :, :] = 0.
            mask_filter[filter_mid_y+1:, :, :, :] = 0.

            if mask == 'a':
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.
                
        W *= mask_filter 
    return W

def get_bias(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer)

def conv_op(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

class GatedCNN():
    def __init__(self, W_shape, fan_in, horizontal, gated=True, payload=None, mask=None, activation=True, conditional=None, conditional_image=None):
        self.fan_in = fan_in
        in_dim = self.fan_in.get_shape()[-1]
        self.W_shape = [W_shape[0], W_shape[1], in_dim, W_shape[2]]  
        self.b_shape = W_shape[2]

        self.in_dim = in_dim
        self.payload = payload
        self.mask = mask
        self.activation = activation
        self.conditional = conditional
        self.conditional_image = conditional_image
        self.horizontal = horizontal
        
        if gated:
            self.gated_conv()
        else:
            self.simple_conv()

    def gated_conv(self):
        W_f = get_weights(self.W_shape, "v_W", self.horizontal, mask=self.mask)
        W_g = get_weights(self.W_shape, "h_W", self.horizontal, mask=self.mask)

        b_f_total = get_bias(self.b_shape, "v_b")
        b_g_total = get_bias(self.b_shape, "h_b")
        if self.conditional is not None:
            h_shape = int(self.conditional.get_shape()[1])
            V_f = get_weights([h_shape, self.W_shape[3]], "v_V", self.horizontal)
            b_f = tf.matmul(self.conditional, V_f)
            V_g = get_weights([h_shape, self.W_shape[3]], "h_V", self.horizontal)
            b_g = tf.matmul(self.conditional, V_g)

            b_f_shape = tf.shape(b_f)
            b_f = tf.reshape(b_f, (b_f_shape[0], 1, 1, b_f_shape[1]))
            b_g_shape = tf.shape(b_g)
            b_g = tf.reshape(b_g, (b_g_shape[0], 1, 1, b_g_shape[1]))

            b_f_total = b_f_total + b_f
            b_g_total = b_g_total + b_g
        if self.conditional_image is not None:
            b_f_total = b_f_total + tf.layers.conv2d(self.conditional_image, self.in_dim, 1, use_bias=False, name="ci_f")
            b_g_total = b_g_total + tf.layers.conv2d(self.conditional_image, self.in_dim, 1, use_bias=False, name="ci_g")

        conv_f = conv_op(self.fan_in, W_f)
        conv_g = conv_op(self.fan_in, W_g)
       
        if self.payload is not None:
            conv_f += self.payload
            conv_g += self.payload

        self.fan_out = tf.multiply(tf.tanh(conv_f + b_f_total), tf.sigmoid(conv_g + b_g_total))

    def simple_conv(self):
        W = get_weights(self.W_shape, "W", self.horizontal, mask_mode="standard", mask=self.mask)
        b = get_bias(self.b_shape, "b")
        conv = conv_op(self.fan_in, W)
        if self.activation: 
            self.fan_out = tf.nn.relu(tf.add(conv, b))
        else:
            self.fan_out = tf.add(conv, b)

    def output(self):
        return self.fan_out 


class PixelCnn:

    def __init__(self,image_size,channels, K,model,lr=0.02):
        #self.commitment_beta=commitment_beta
        # number of qunatization vectors
        self.K=K
        #self.encoding_size=encoding_size
        self.image_size=image_size
        self.channels=channels
        self.lr=lr
        self.model=model
        self.build_model()
        self.saver = tf.train.Saver()
        self.start_session()
    
    def start_session(self):
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
        self.summary_op=tf.summary.merge_all()


    def build_model(self):

        tf.reset_default_graph()

        #images
        self.X=tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.channels], name="x_input")
        self.pixelcnn()
        #self._loss_init(inputs,outputs)
        self._train_init()


    def pixelcnn(self,num_layers_pixelcnn=12, fmaps_pixelcnn=64):
        inputs=self.X
        self.h=None
        full_horizontal=True
        v_stack_in = inputs
        h_stack_in = inputs

        for i in range(num_layers_pixelcnn):
            filter_size = 3 if i > 0 else 7
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)
            with tf.variable_scope("v_stack"+i):
                v_stack = GatedCNN([filter_size, filter_size, fmaps_pixelcnn], v_stack_in, False, mask=mask, conditional=self.h).output()
                v_stack_in = v_stack

            with tf.variable_scope("v_stack_1"+i):
                v_stack_1 = GatedCNN([1, 1, fmaps_pixelcnn], v_stack_in, False, gated=False, mask=None).output()

            with tf.variable_scope("h_stack"+i):
                h_stack = GatedCNN([filter_size if full_horizontal else 1, filter_size, fmaps_pixelcnn], h_stack_in, True, payload=v_stack_1, mask=mask, conditional=self.h).output()

            with tf.variable_scope("h_stack_1"+i):
                h_stack_1 = GatedCNN([1, 1, fmaps_pixelcnn], h_stack, True, gated=False, mask=None).output()
                if residual:
                    h_stack_1 += h_stack_in # Residual connection
                h_stack_in = h_stack_1

        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, fmaps_pixelcnn], h_stack_in, True, gated=False, mask='b').output()
        
        color_dim = 256
        with tf.variable_scope("fc_2"):
            self.fc2 = GatedCNN([1, 1, self.channels * color_dim], fc1, True, gated=False, mask='b', activation=False).output()
            self.fc2 = tf.reshape(self.fc2, (-1, self.channels*self.image_size**2,color_dim))
        
        inputs=tf.reshape(self.X,(-1,self.channels*self.image_size**2), name="inputs")
        self.outputs=tf.reshape(self.fc2,(-1,self.channels*self.image_size**2,self.K),name="outputs")
        outputs=self.outputs
        dist = tf.distributions.Categorical(logits=outputs)
        self.sampled_pixelcnn = dist.sample()
        self.log_prob_pixelcnn = dist.log_prob(self.sampled_pixelcnn)

        #inputs = tf.cast(inputs, tf.int32)
        #self.inputs=inputs

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.fc2, labels=tf.cast(tf.reshape(self.X, [-1,self.channels*self.image_size**2]), dtype=tf.int32)))#tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=inputs))


    def _train_init(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train = [self.optimizer.minimize(self.loss)]

    def read_batch(self,paths):
        return paths

        imgs=np.zeros([len(paths),self.image_size,self.image_size,self.channels],dtype=np.int32)

        mode="RGB" if(self.channels==3) else "L"
        for i,img in enumerate(paths):
            _img=Image.open(img).convert(mode)
            imgs[i]=np.array(_img.resize((self.image_size,self.image_size))).reshape(self.image_size,self.image_size,self.channels)
        
        return paths
    def get_feed_dict(self,X_batch):
        _x=self.read_batch(X_batch)
        #encodings=self.model.sess.run(self.model.encodings,feed_dict={self.model.X:X_batch})
        feed_dict={self.X:_x}
        return feed_dict

    def partial_fit(self,X,X_test=None, batch_size=64):

        np.random.shuffle(X)
        num_batches=X.shape[0]//batch_size
        train_out=[]
        with tqdm(range(num_batches)) as t:
            for i in t:
                X_batch=X[i*batch_size:(i+1)*batch_size]
                #X_images=self.read_batch(X_batch)
                loss,_=self.sess.run([self.loss]+[self.train],feed_dict=self.get_feed_dict(X_batch))
                
                #losses=self.sess.run( list(self.losses.values()),
                #                        feed_dict=self.get_feed_dict(X_images))
                #loss_monitor=list(zip(self.losses.keys(),losses))
                desc=""
                desc+=("%s: %.2f\t"%("loss",loss))
                t.set_description(desc)
                train_out.append(loss)


        return train_out

    def fit(self,X,epochs=10):
        train_loss_pixelcnn = []
        for i in range(epochs):
            self.partial_fit(X)
        