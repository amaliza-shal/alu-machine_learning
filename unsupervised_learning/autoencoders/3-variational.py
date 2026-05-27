#!/usr/bin/env python3
"""
Defines function that creates a variational autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    input_dims: int, dimensions of the model input
    hidden_layers: list of ints, number of nodes for each hidden layer
                   in the encoder (reversed for decoder)
    latent_dims: int, dimensions of the latent space representation

    Returns: encoder, decoder, auto
        encoder outputs: latent representation, mean, log variance
        decoder outputs: reconstructed input
        auto is compiled with adam and binary cross-entropy loss
    """
    if type(input_dims) is not int:
        raise TypeError(
            "input_dims must be an int containing dimensions of model input")
    if type(hidden_layers) is not list:
        raise TypeError("hidden_layers must be a list of ints "
                        "representing number of nodes for each layer")
    for nodes in hidden_layers:
        if type(nodes) is not int:
            raise TypeError("hidden_layers must be a list of ints "
                            "representing number of nodes for each layer")
    if type(latent_dims) is not int:
        raise TypeError(
            "latent_dims must be an int containing dimensions of "
            "latent space representation")

    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    # Mean and log-variance layers use linear activation (None)
    z_mean = keras.layers.Dense(
        units=latent_dims, activation='linear')(x)
    z_log_var = keras.layers.Dense(
        units=latent_dims, activation='linear')(x)

    # Reparameterization trick via Lambda layer
    def sampling(args):
        """Samples from latent distribution using reparameterization."""
        mean, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mean))
        return mean + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(
        inputs=encoder_inputs,
        outputs=[z, z_mean, z_log_var]
    )

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(
        units=input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # Full Autoencoder
    auto_inputs = encoder_inputs
    z, z_mean, z_log_var = encoder(auto_inputs)
    auto_outputs = decoder(z)

    auto = keras.Model(inputs=auto_inputs, outputs=auto_outputs)

    # KL divergence loss added as a layer loss
    def kl_loss(z_mean, z_log_var):
        """Computes KL divergence between latent distribution and N(0,1)."""
        kl = -0.5 * keras.backend.sum(
            1 + z_log_var - keras.backend.square(z_mean)
            - keras.backend.exp(z_log_var),
            axis=-1
        )
        return kl

    auto.add_loss(keras.backend.mean(kl_loss(z_mean, z_log_var)))

    # Sum binary crossentropy over input dims to match expected loss scale
    def reconstruction_loss(y_true, y_pred):
        """Binary crossentropy summed over input dims (not averaged)."""
        return keras.backend.sum(
            keras.backend.binary_crossentropy(y_true, y_pred),
            axis=-1
        )

    auto.compile(optimizer='adam', loss=reconstruction_loss)

    return encoder, decoder, auto