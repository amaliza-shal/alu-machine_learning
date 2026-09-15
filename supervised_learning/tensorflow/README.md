# TensorFlow

This project involves building and training a neural network using TensorFlow 1.12.

## Tasks

### 0. Placeholders
- [0-create_placeholders.py](0-create_placeholders.py): Write the function `def create_placeholders(nx, classes):` that returns two placeholders, `x` and `y`, for the neural network.

### 1. Layers
- [1-create_layer.py](1-create_layer.py): Write the function `def create_layer(prev, n, activation):` that creates a neural network layer.

### 2. Forward Propagation
- [2-forward_prop.py](2-forward_prop.py): Write the function `def forward_prop(x, layer_sizes=[], activations=[]):` that creates the forward propagation graph for the neural network.

### 3. Accuracy
- [3-calculate_accuracy.py](3-calculate_accuracy.py): Write the function `def calculate_accuracy(y, y_pred):` that calculates the accuracy of a prediction.

### 4. Loss
- [4-calculate_loss.py](4-calculate_loss.py): Write the function `def calculate_loss(y, y_pred):` that calculates the softmax cross-entropy loss of a prediction.

### 5. Train_Op
- [5-create_train_op.py](5-create_train_op.py): Write the function `def create_train_op(loss, alpha):` that creates the training operation for the network.

### 6. Train
- [6-train.py](6-train.py): Write the function `def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):` that builds, trains, and saves a neural network classifier.

### 7. Evaluate
- [7-evaluate.py](7-evaluate.py): Write the function `def evaluate(X, Y, save_path):` that evaluates the output of a neural network.
