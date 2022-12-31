
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.losses import mse, binary_crossentropy, kl_divergence
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import numpy as np


class Tf_VAE:

    def __init__(self, tf_params):
        self.tf_params = tf_params

    def build_network(self, original_dim):

        input_shape = (original_dim,)
        intermediate_dim = int(original_dim / 2)
        latent_dim = int(original_dim / 3)

        # encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        self.z_mean = Dense(latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use the reparameterization trick and get the output from the sample() function
        z = Lambda(self.sample, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])
        encoder = Model(inputs, z, name='encoder')

        # decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # Instantiate the decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # full VAE model
        outputs = decoder(encoder(inputs))
        vae_model = Model(inputs, outputs, name='vae_mlp')

        self.vae_model = vae_model

    def model_fit(self, train):

        opt = optimizers.Adagrad(learning_rate=self.tf_params['learning_rate'],
                                 clipvalue=0.5)
        self.vae_model.compile(optimizer='adam', loss=self.vae_loss)
        self.vae_model.fit(train, train,
                            shuffle=self.tf_params['shuffle'],
                            epochs=self.tf_params['epochs'],
                            batch_size=self.tf_params['batch_size'])

    def model_predict(self, test):
        prediction = self.vae_model.predict(test)

        return prediction

    def vae_loss(self, x, x_decoded_mean):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.square(K.exp(self.z_log_var)), axis=-1)
        # return the average loss over all
        total_loss = K.mean(reconstruction_loss + kl_loss)
        # total_loss = reconstruction_loss + kl_loss
        return total_loss

    def sample(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
