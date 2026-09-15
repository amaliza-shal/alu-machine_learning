#!/usr/bin/env python3
"""
Variational Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(units=latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(units=latent_dims, activation=None)(x)

    def sampling(args):
        z_m, z_lv = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(z_lv / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z, z_mean, z_log_var])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(units=input_dims,
                                         activation='sigmoid')(x)

    decoder = keras.Model(decoder_inputs, decoder_outputs)

    # Full Autoencoder
    auto_outputs = decoder(encoder(encoder_inputs)[0])
    auto = keras.Model(encoder_inputs, auto_outputs)

    def vae_loss(y_true, y_pred):
        recon_loss = keras.backend.binary_crossentropy(y_true, y_pred)
        recon_loss = keras.backend.sum(recon_loss, axis=-1)
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_var - keras.backend.square(z_mean) -
            keras.backend.exp(z_log_var), axis=-1)
        return keras.backend.mean(recon_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
