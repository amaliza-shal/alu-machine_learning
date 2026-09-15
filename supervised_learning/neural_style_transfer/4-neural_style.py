#!/usr/bin/env python3
"""Shared implementation for the neural style transfer assignment."""

import numpy as np
import tensorflow as tf

if not tf.executing_eagerly():
    if (hasattr(tf, "compat") and hasattr(tf.compat, "v1") and
            hasattr(tf.compat.v1, "enable_eager_execution")):
        tf.compat.v1.enable_eager_execution()
    elif hasattr(tf, "enable_eager_execution"):
        tf.enable_eager_execution()


class NST:
    """Perform neural style transfer with VGG19 features."""

    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    content_layer = "block5_conv2"
    _stage = 4

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """Initialize and preprocess the style and content images."""
        self._validate_image(style_image, "style_image")
        self._validate_image(content_image, "content_image")
        self._validate_weight(alpha, "alpha")
        self._validate_weight(beta, "beta")
        if self._stage >= 10:
            self._validate_weight(var, "var")

        if not tf.executing_eagerly():
            if (hasattr(tf, "compat") and hasattr(tf.compat, "v1") and
                    hasattr(tf.compat.v1, "enable_eager_execution")):
                tf.compat.v1.enable_eager_execution()
            elif hasattr(tf, "enable_eager_execution"):
                tf.enable_eager_execution()
            else:
                tf.config.run_functions_eagerly(True)
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        if self._stage >= 10:
            self.var = var
        if self._stage >= 1:
            self.load_model()
        if self._stage >= 3:
            self.generate_features()

    @staticmethod
    def _validate_image(image, name):
        """Validate an RGB image represented as a NumPy array."""
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "{} must be a numpy.ndarray with shape (h, w, 3)".format(
                    name))

    @staticmethod
    def _validate_weight(value, name):
        """Validate a non-negative numeric cost weight."""
        if (isinstance(value, bool) or
                not isinstance(value, (int, float, np.number)) or value < 0):
            raise TypeError("{} must be a non-negative number".format(name))

    @staticmethod
    def scale_image(image):
        """Resize an RGB image to a maximum dimension of 512 pixels."""
        NST._validate_image(image, "image")
        height, width = image.shape[:2]
        scale = 512 / max(height, width)
        new_size = (max(1, round(height * scale)),
                    max(1, round(width * scale)))
        if hasattr(tf.image, "resize"):
            scaled = tf.image.resize(image, new_size, method="bicubic")
        else:
            scaled = tf.image.resize_images(
                image, new_size, method=tf.image.ResizeMethod.BICUBIC)
        scaled = tf.clip_by_value(tf.cast(scaled, tf.float32), 0.0, 255.0)
        return tf.expand_dims(scaled / 255.0, axis=0)

    def load_model(self):
        """Build the frozen VGG19 feature extraction model."""
        base = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet")

        def replace_pooling(layer):
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                return tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name,
                )
            return layer

        vgg = tf.keras.models.clone_model(base, clone_function=replace_pooling)
        for cloned, original in zip(vgg.layers, base.layers):
            if original.get_weights():
                cloned.set_weights(original.get_weights())
        vgg.trainable = False
        output_layers = [vgg.get_layer(
            name).output for name in self.style_layers]
        output_layers.append(vgg.get_layer(self.content_layer).output)
        self.model = tf.keras.Model(vgg.input, output_layers)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the normalized Gram matrix for a convolution output."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                input_layer.shape.rank != 4):
            raise TypeError("input_layer must be a tensor of rank 4")
        shape = tf.shape(input_layer)
        features = tf.reshape(input_layer, (shape[0], -1, shape[3]))
        gram = tf.matmul(features, features, transpose_a=True)
        return gram / tf.cast(shape[1] * shape[2]
                              * shape[3], input_layer.dtype)

    def generate_features(self):
        """Extract target style Gram matrices and the content feature."""
        vgg19 = tf.keras.applications.vgg19
        style_input = vgg19.preprocess_input(self.style_image * 255.0)
        content_input = vgg19.preprocess_input(self.content_image * 255.0)
        style_outputs = self.model(style_input)
        content_outputs = self.model(content_input)
        self.gram_style_features = [self.gram_matrix(
            output) for output in style_outputs[:-1]]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for one convolution layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                style_output.shape.rank != 4):
            raise TypeError("style_output must be a tensor of rank 4")
        channels = style_output.shape[-1]
        expected = ("gram_target must be a tensor of shape [1, {}, {}] where "
                    "{} is the number of channels in style_output").format(
                        channels, channels, channels)
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape.rank != 3 or
                tuple(gram_target.shape) != (1, channels, channels)):
            raise TypeError(expected)
        gram_output = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_output - gram_target))

    def style_cost(self, style_outputs):
        """Calculate the evenly weighted style cost."""
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length))
        costs = [self.layer_style_cost(output, target) for output, target in
                 zip(style_outputs, self.gram_style_features)]
        return tf.add_n(costs) / length

    def content_cost(self, content_output):
        """Calculate the content cost for a generated feature output."""
        expected = tuple(self.content_feature.shape)
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                tuple(content_output.shape) != expected):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(expected))
        return tf.reduce_mean(
            tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """Calculate the weighted content, style, and variation costs."""
        self._validate_generated(generated_image)
        processed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0)
        outputs = self.model(processed)
        content = self.content_cost(outputs[-1])
        style = self.style_cost(outputs[:-1])
        if self._stage >= 10:
            variation = self.variational_cost(generated_image)
            total = self.alpha * content + self.beta * style
            return total + self.var * variation, content, style, variation
        return self.alpha * content + self.beta * style, content, style

    def compute_grads(self, generated_image):
        """Calculate image gradients and the current transfer costs."""
        self._validate_generated(generated_image)
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            costs = self.total_cost(generated_image)
            total = costs[0]
        return (tape.gradient(total, generated_image), *costs)

    @staticmethod
    def variational_cost(generated_image):
        """Calculate the total neighboring-pixel variation cost."""
        horizontal = tf.square(
            generated_image[:, 1:, :-1] - generated_image[:, :-1, :-1])
        vertical = tf.square(
            generated_image[:, :-1, 1:] - generated_image[:, :-1, :-1])
        return tf.reduce_sum(horizontal + vertical)

    def _validate_generated(self, generated_image):
        """Validate a generated image against the content image shape."""
        expected = tuple(self.content_image.shape)
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                tuple(generated_image.shape) != expected):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    expected))

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Optimize and return the lowest-cost generated image."""
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if (step is not None and
                (not isinstance(step, int) or isinstance(step, bool))):
            raise TypeError("step must be an integer")
        if step is not None and (step <= 0 or step > iterations):
            message = ("iterations must be positive and less than iterations"
                       if self._stage >= 10 else
                       "step must be positive and less than iterations")
            raise ValueError(message)
        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        for value, name in ((beta1, "beta1"), (beta2, "beta2")):
            if not isinstance(value, float):
                raise TypeError("{} must be a float".format(name))
            if value < 0 or value > 1:
                raise ValueError(
                    "{} must be in the range [0, 1]".format(name))

        generated = tf.Variable(self.content_image)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2)
        best_cost = float("inf")
        best_image = tf.identity(generated)
        for iteration in range(iterations + 1):
            result = self.compute_grads(generated)
            gradients, total, content, style = result[:4]
            variation = result[4] if self._stage >= 10 else None
            if (step is not None and
                    (iteration % step == 0 or iteration == iterations)):
                if self._stage >= 10:
                    print(
                        "Cost at iteration {}: {}, content {}, style {}, "
                        "var {}".format(
                            iteration, total, content, style, variation))
                else:
                    print(
                        "Cost at iteration {}: {}, content {}, style {}"
                        .format(iteration, total, content, style))
            if float(total) < best_cost:
                best_cost = float(total)
                best_image = tf.identity(generated)
            if iteration < iterations:
                optimizer.apply_gradients([(gradients, generated)])
                generated.assign(tf.clip_by_value(generated, 0.0, 1.0))
        return best_image, best_cost
