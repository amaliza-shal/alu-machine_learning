#!/usr/bin/env python3
"""
Variational Autoencoder
"""
import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    input_dims: dimensions of the model input
    hidden_layers: list of number of nodes for each hidden layer in encoder
    latent_dims: dimensions of the latent space representation
    Returns: encoder, decoder, auto
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sig = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Sampling function for the latent space"""
        mu, log_sig = args
        epsilon = tf.keras.backend.random_normal(
            shape=(tf.shape(mu)[0], latent_dims), mean=0.0, stddev=1.0
        )
        return mu + tf.exp(log_sig / 2) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([mu, log_sig])
    encoder = keras.Model(encoder_input, [z, mu, log_sig])

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    reconstructed = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, reconstructed)

    # Autoencoder
    auto_output = decoder(encoder(encoder_input)[0])
    auto = keras.Model(encoder_input, auto_output)

    # VAE Loss = Reconstruction Loss + KL Divergence
    def vae_loss(y_true, y_pred):
        recon_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        recon_loss *= input_dims
        kl_loss = 1 + log_sig - tf.square(mu) - tf.exp(log_sig)
        kl_loss = tf.reduce_mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        return tf.reduce_mean(recon_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
