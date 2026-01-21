import numpy as np
#from ImageSegmentation import final_dataset,label_encoder   # Your dataset
import os
import json
import sys
import torch
import cv2
from sklearn.metrics import classification_report
import time
# open log file
log_file = open("Training_And_Validation.txt", "w")

class Logger(object):
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)   # still print to console
        self.file.write(message)       # also save to file
        self.file.flush()              # make sure it writes immediately

    def flush(self):
        # needed for Python compatibility
        self.terminal.flush()
        self.file.flush()

# Redirect stdout (print statements)
sys.stdout = Logger(sys.stdout, log_file)

def load_weights(cls, path="trained_model/cnn_model.npz"):
    """Load CNN model architecture and weights from a file and return an instance."""
    data = np.load(path, allow_pickle=True)

    # Load config
    config = json.loads(str(data["config"]))

    # Create a new instance of the class
    model = cls(
        input_shape=tuple(config["input_shape"]),
        num_classes=config["num_classes"],
        conv_layers=config["conv_layers"],
        hidden_units=config["hidden_units"],
        dropout_rate=config["dropout_rate"],
        leaky_alpha=config.get("leaky_alpha", 0.01)  # default if missing
    )

    # Build fresh model layers
    model.layers = []
    model._build_model()

    # Load weights into layers
    for i, layer in enumerate(model.layers):
        if layer['type'] == 'conv':
            layer['filters'] = data[f"W{i}"]
            layer['biases'] = data[f"b{i}"]
        elif layer['type'] in ['dense', 'output']:
            layer['weights'] = data[f"W{i}"]
            layer['biases'] = data[f"b{i}"]

    print(f"[INFO] Model loaded from {path}")
    return model


# ------------------------------------------------------
# CNN Model Class with Debug Diagnostics + Max Pooling
# ------------------------------------------------------
class CNNModelTraining:
    def __init__(self, input_shape, num_classes, conv_layers=[(8,3), (16,3)], hidden_units=[128,64], dropout_rate=0.3, leaky_alpha=0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_layers_config = conv_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.leaky_alpha = leaky_alpha
        self.layers = []
        self.epoch_accuracy = []
        self._build_model()
        print("[INIT] Model initialized with:")
        print("       Input shape:", input_shape)
        print("       Conv layers:", conv_layers)
        print("       Hidden units:", hidden_units)
        print("       Output classes:", num_classes)
        print("       LeakyReLU alpha:", leaky_alpha)

    # -----------------------
    # Build Model
    # -----------------------
    def _build_model(self):
        in_shape = self.input_shape

        # --- Convolutional + Max Pool layers ---
        for num_filters, ksize in self.conv_layers_config:
            # He init for conv (ReLU/LeakyReLU)
            filters = np.random.randn(num_filters, ksize, ksize, in_shape[2]) * np.sqrt(2.0/(ksize*ksize*in_shape[2]))
            biases = np.zeros(num_filters)
            out_h = in_shape[0] - ksize + 1
            out_w = in_shape[1] - ksize + 1

            self.layers.append({
                'type': 'conv',
                'filters': filters,
                'biases': biases,
                'input_shape': in_shape,
                'output_shape': (out_h, out_w, num_filters),
                'ksize': ksize,
                'num_filters': num_filters,
                'input': None,
                'output': None
            })
            in_shape = (out_h, out_w, num_filters)

            # Max Pool layer
            pool_h, pool_w = in_shape[0] // 2, in_shape[1] // 2
            self.layers.append({
                'type': 'pool',
                'input_shape': in_shape,
                'output_shape': (pool_h, pool_w, in_shape[2]),
                'input': None,
                'output': None,
                'switches': None
            })
            in_shape = (pool_h, pool_w, in_shape[2])

        # Flatten size
        flattened_size = int(np.prod(in_shape))

        # --- Dense layers ---
        prev_units = flattened_size
        for units in self.hidden_units:
            # Xavier/Glorot init for dense (works well for tanh/sigmoid, ok for others too)
            limit = np.sqrt(6.0 / (prev_units + units))
            weights = np.random.uniform(-limit, limit, (units, prev_units))
            biases = np.zeros(units)
            self.layers.append({
                'type': 'dense',
                'weights': weights,
                'biases': biases,
                'input_shape': (prev_units,),
                'input': None,
                'z': None,
                'output_shape': (units,)
            })
            prev_units = units

        # --- Output layer ---
        limit = np.sqrt(6.0 / (prev_units + self.num_classes))
        weights = np.random.uniform(-limit, limit, (self.num_classes, prev_units))
        biases = np.zeros(self.num_classes)
        self.layers.append({
            'type': 'output',
            'weights': weights,
            'biases': biases,
            'input_shape': (prev_units,),
            'input': None,
            'z': None,
            'output_shape': (self.num_classes,)
        })

    # -----------------------
    # Forward Pass (single sample)
    # -----------------------
    def forward(self, x, training=True):
        # Expect x shape: (H, W, C) single sample
        out = x
        for idx, layer in enumerate(self.layers):
            if layer['type'] == 'conv':
                layer['input'] = out
                out = self._conv_forward(out, layer)
                layer['output'] = out

            elif layer['type'] == 'pool':
                layer['input'] = out
                out, switches = self._max_pool_forward(out)
                layer['output'] = out
                layer['switches'] = switches

            elif layer['type'] == 'dense':
                flat = out.flatten()
                layer['input'] = flat.copy()
                z = np.dot(layer['weights'], flat) + layer['biases']
                layer['z'] = z  # store pre-activation for backward

                # LeakyReLU (no normalization here)
                out = np.where(z > 0, z, self.leaky_alpha * z)

                if training and self.dropout_rate > 0.0:
                    mask = (np.random.rand(*out.shape) > self.dropout_rate).astype(np.float32)
                    out *= mask / (1.0 - self.dropout_rate)

            elif layer['type'] == 'output':
                flat = out.flatten()
                layer['input'] = flat.copy()
                z = np.dot(layer['weights'], flat) + layer['biases']
                layer['z'] = z
                # stable 1D softmax
                out = self._softmax(z)

        return out  # returned shape: (num_classes,)

    # -----------------------
    # Softmax (1D logits -> 1D probs)
    # -----------------------
    def _softmax(self, z):
        # z: 1D logits (num_classes,)
        z = np.array(z, dtype=np.float64)
        z = np.clip(z, -50.0, 50.0)
        z = z - np.max(z)
        exps = np.exp(z)
        s = np.sum(exps)
        if s == 0:
            return np.ones_like(z) / len(z)
        return exps / (s + 1e-12)

    # -----------------------
    # Gradient Clipping Helper (elementwise or by norm)
    # -----------------------
    def _clip_grad(self, grad, max_norm=5.0):
        # grad may be array; clip by norm
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * (max_norm / (norm + 1e-6))
        return grad

    # ===============================================================
    # Convolution + LeakyReLU forward
    # ===============================================================
    def _conv_forward(self, x, layer):
        """
        Performs the forward pass for a convolutional layer followed by LeakyReLU activation.
        
        Args:
            x: Input feature map of shape (H_in, W_in, C_in)
            layer: Dictionary containing:
                - 'filters': numpy array of filters (kernels)
                - 'biases': biases for each filter
                - 'ksize': kernel size 
                - 'output_shape': tuple (H_out, W_out, F)
        
        Returns:
            out: Output feature map after convolution + LeakyReLU activation
        """

        H, W, F = layer['output_shape']     # Output spatial height, width, and number of filters
        k = layer['ksize']                  # Kernel size (e.g., 3 → 3x3 filters)
        out = np.zeros((H, W, F))           # Initialize output feature map

        # Loop over each filter
        for f in range(F):
            filt = layer['filters'][f]      # Current filter (shape: k x k x C_in)
            bias = layer['biases'][f]       # Corresponding bias term

            # Slide the filter spatially over the input
            for i in range(H):
                for j in range(W):
                    # Sum over all channels for convolution
                    for c in range(x.shape[2]):
                        out[i, j, f] += np.sum(x[i:i+k, j:j+k, c] * filt[:, :, c])
                    
                    # Add bias and apply LeakyReLU activation
                    val = out[i, j, f] + bias
                    out[i, j, f] = val if val > 0 else self.leaky_alpha * val

        return out


    # ===============================================================
    # Max Pool Forward
    # ===============================================================
    def _max_pool_forward(self, x, size=2, stride=2):
        """
        Performs max pooling on the input feature map.
        
        Args:
            x: Input feature map (H_in, W_in, C)
            size: Pooling window size (default = 2)
            stride: Step size between pooling windows (default = 2)
        
        Returns:
            out: Downsampled feature map
            switches: Boolean mask recording the max locations (for backprop)
        """

        H, W, C = x.shape
        out_h = H // size
        out_w = W // size
        out = np.zeros((out_h, out_w, C))               # Pooled output
        switches = np.zeros_like(x, dtype=bool)         # Mask to store max positions

        # Process each channel independently
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    # Define the current pooling window
                    h_start = i * stride
                    w_start = j * stride
                    patch = x[h_start:h_start+size, w_start:w_start+size, c]

                    # Find max and record its position
                    max_val = np.max(patch)
                    out[i, j, c] = max_val
                    switches[h_start:h_start+size, w_start:w_start+size, c] = (patch == max_val)

        return out, switches


    # ===============================================================
    # Max Pool Backward
    # ===============================================================
    def _max_pool_backward(self, d_out, switches, pool_input_shape, size=2, stride=2):
        """
        Performs the backward pass for a max pooling layer.
        
        Args:
            d_out: Gradient of the loss w.r.t. the pooled output
            switches: Boolean mask from forward pass (indicating max positions)
            pool_input_shape: Shape of the input to the pooling layer (H_in, W_in, C)
            size: Pool window size (default = 2)
            stride: Stride used during pooling (default = 2)
        
        Returns:
            dX: Gradient of the loss w.r.t. the input of the pooling layer
        """

        # Ensure d_out has correct shape
        if d_out.ndim != 3:
            d_out = d_out.reshape((pool_input_shape[0] // size,
                                   pool_input_shape[1] // size,
                                   pool_input_shape[2]))

        H, W, C = pool_input_shape
        dX = np.zeros(pool_input_shape)      # Initialize gradient for input
        out_h, out_w, _ = d_out.shape

        # Distribute gradients only to positions that were max during forward pass
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    dX[h_start:h_start+size, w_start:w_start+size, c] += \
                        d_out[i, j, c] * switches[h_start:h_start+size, w_start:w_start+size, c]

        return dX

    # -----------------------
    # Compute gradients for a single sample (no weight updates) — returns grads dict
    # -----------------------
    def _compute_sample_grads(self, y_true):
        """
        Assumes forward(x) has been called and layer['input'], layer['z'], layer['output'], layer['switches'] are set.
        y_true: 1D one-hot vector shape (num_classes,)
        Returns grad_acc dict with keys per layer type:
          - for dense/output: {'dW': array_like, 'db': array_like}
          - for conv: {'dF': same shape as filters, 'db': scalar per filter}
        The order matches self.layers indices.
        """
        grads = [None] * len(self.layers)
        d_out = None

        # iterate reversed
        for r_idx, layer in enumerate(reversed(self.layers)):
            idx = len(self.layers) - 1 - r_idx
            if layer['type'] == 'output':
                probs = self._softmax(layer['z'])
                d_out = probs - y_true  # shape (num_classes,)
                # grads for output weights/biases
                dW = np.outer(d_out, layer['input'])  # (num_classes, prev_units)
                db = d_out.copy()
                grads[idx] = {'dW': dW, 'db': db}
                # propagate to previous layer
                d_out = np.dot(layer['weights'].T, d_out)  # shape (prev_units,)

            elif layer['type'] == 'dense':
                # layer['z'] shape: (units,)
                # derivative of LeakyReLU: 1 if z>0 else alpha
                d_activation = np.where(layer['z'] > 0, 1.0, self.leaky_alpha)  # (units,)
                dz = d_out * d_activation  # (units,)
                # compute grads
                dW = np.outer(dz, layer['input'])  # (units, in_dim)
                db = dz.copy()
                grads[idx] = {'dW': dW, 'db': db}
                # propagate to previous layer
                d_out = np.dot(layer['weights'].T, dz)  # shape (in_dim,)

            elif layer['type'] == 'pool':
                # d_out currently is flattened shape; reshape appropriate to pool output shape
                pool_input_shape = layer['input'].shape  # (H,W,C)
                # d_out must be shape (out_h, out_w, C)
                if d_out.ndim == 1:
                    # attempt to reshape to (out_h, out_w, C)
                    out_h, out_w, C = layer['output_shape']
                    d_out = d_out.reshape((out_h, out_w, C))
                dX = self._max_pool_backward(d_out, layer['switches'], layer['input'].shape)
                d_out = dX  # pass to conv below
                grads[idx] = None  # pool has no params

            elif layer['type'] == 'conv':
                x = layer['input']
                k = layer['ksize']
                H, W, F = layer['output_shape']
                dX = np.zeros_like(x)
                dF_all = np.zeros_like(layer['filters'])
                db_all = np.zeros_like(layer['biases'])
                for f in range(F):
                    grad_accum = np.zeros_like(layer['filters'][f])
                    for i in range(H):
                        for j in range(W):
                            # gradient wrt activation: LeakyReLU derivative
                            local_mask = 1.0 if layer['output'][i, j, f] > 0 else self.leaky_alpha
                            grad_val = local_mask * d_out[i, j, f]
                            # accumulate
                            for c in range(x.shape[2]):
                                patch = x[i:i+k, j:j+k, c]
                                grad_accum[:, :, c] += grad_val * patch
                                dX[i:i+k, j:j+k, c] += layer['filters'][f][:, :, c] * grad_val
                    dF_all[f] = grad_accum
                    db_all[f] = np.sum(np.where(layer['output'][:, :, f] > 0, 1.0, self.leaky_alpha) * d_out[:, :, f])
                grads[idx] = {'dF': dF_all, 'db_conv': db_all}
                d_out = dX  # propagate to earlier layers

        return grads

    # -----------------------
    # Loss (single sample or batch-aware)
    # -----------------------
    def cross_entropy(self, probs, y_true):
        # probs: (num_classes,) or (batch, num_classes)
        # y_true: same shape
        probs = np.clip(probs, 1e-12, 1.0)
        if probs.ndim == 1:
            return -np.sum(y_true * np.log(probs))
        else:
            return -np.mean(np.sum(y_true * np.log(probs), axis=1))

    # -----------------------
    # Apply averaged grads (grads_acc is list of per-layer averaged grads)
    # -----------------------
    def _apply_grads(self, grads_acc, lr):
        # grads_acc is list of same length as layers, each element either None or dict with keys depending on layer
        for idx, g in enumerate(grads_acc):
            if g is None:
                continue
            layer = self.layers[idx]
            if layer['type'] == 'output' or layer['type'] == 'dense':
                # apply clipped gradients
                dW = g['dW']
                db = g['db']
                # optional gradient clipping by norm
                dW = self._clip_grad(dW, max_norm=5.0)
                db = self._clip_grad(db, max_norm=5.0)
                layer['weights'] -= lr * dW
                layer['biases']  -= lr * db
            elif layer['type'] == 'conv':
                dF = g['dF']
                db = g['db_conv']
                # clip grads
                dF = self._clip_grad(dF, max_norm=5.0)
                db = self._clip_grad(db, max_norm=5.0)
                layer['filters'] -= lr * dF
                layer['biases'] -= lr * db

    # -----------------------
    # Train (mini-batch gradient accumulation and single update per batch)
    # -----------------------
    def train(self, X , y_onehot , X_test, y_test , epochs=10, lr=0.01, batch_size=8):
        """
        X: array-like, each sample shape (H,W,C)
        y_onehot: array-like, each sample shape (num_classes,)
        Training uses mini-batch accumulation of gradients and single update per batch.
        """
        dataset_size = len(X)
        best_acc = 0.0
        best_weights = None

        print("[Training Params] :")
        print("       Learning Rate:", lr)
        print("       Drop Out Rate:", self.dropout_rate)
        print("       Num Epochs:", epochs)
        print("       Batch size:", batch_size)
        


        for epoch in range(epochs):
            # Shuffle data each epoch
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            X_shuf = X[indices]
            y_shuf = y_onehot[indices]
            accuracy = 0
            total_loss = 0.0
          

            # Loop over mini-batches
            for i in range(0, dataset_size, batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]

                # initialize accumulators for grads
                grads_acc = [None] * len(self.layers)
                batch_loss = 0.0
               
                n_samples = len(X_batch)

                for x_sample, y_sample in zip(X_batch, y_batch):
                    probs = self.forward(x_sample, training=True)  # shape (num_classes,)
                    batch_loss += self.cross_entropy(probs, y_sample)


                    # compute grads for this sample (no update)
                    sample_grads = self._compute_sample_grads(y_sample)

                    # accumulate
                    for idx, g in enumerate(sample_grads):
                        if g is None:
                            continue
                        if grads_acc[idx] is None:
                            # initialize accumulator with zeros of same shape
                            grads_acc[idx] = {}
                            for k in g:
                                grads_acc[idx][k] = np.zeros_like(g[k])
                        # add
                        for k in g:
                            grads_acc[idx][k] += g[k]

                # average grads over batch
                for idx, g in enumerate(grads_acc):
                    if g is None:
                        continue
                    for k in g:
                        grads_acc[idx][k] = g[k] / float(n_samples)

                # apply averaged gradients (with clipping inside)
                self._apply_grads(grads_acc, lr)

                total_loss += batch_loss
                accuracy = self.get_training_metrics( X_test, y_test )

                print(f"[EPOCH {epoch+1}/{epochs}, BATCH {i//batch_size+1}] BatchLoss={batch_loss/ max(1, n_samples):.4f}  Accuracy={accuracy}")

            # Epoch metrics
            avg_loss = total_loss / dataset_size
            

            # --- Weight statistics ---
            weight_stats = []
            for idx, layer in enumerate(self.layers):
                if 'weights' in layer:
                    w = layer['weights']
                    weight_stats.append(f"Layer {idx} weights: mean={np.mean(w):.4e}, std={np.std(w):.4e}, max={np.max(w):.4e}, min={np.min(w):.4e}")
            print(f"\n[EPOCH {epoch+1}] Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
            print("[Weight Stats] per layer:")
            for ws in weight_stats:
                print("   ", ws)

            self.epoch_accuracy.append(accuracy)

            # Save best weights
            if accuracy > best_acc:
                best_acc = accuracy
                best_weights = []
                for layer in self.layers:
                    layer_copy = {}
                    for key in layer:
                        if key in ['weights','biases','filters']:
                            layer_copy[key] = np.copy(layer[key])
                    best_weights.append(layer_copy)
                self.save_model()

            # Learning rate decay
            lr *= 0.98

        print(f"[TRAIN] Best accuracy: {best_acc:.4f}")

        # Restore best weights if any
        if best_weights is not None:
            for layer, saved in zip(self.layers, best_weights):
                for key in saved:
                    layer[key] = saved[key]



    def log_gradients(self, epoch, batch):
        print(f"[Grad Stats] E{epoch}B{batch}")
        for idx, (dW, db) in enumerate(self.grads):  # assuming grads = [(dW, db), ...]
            print(f"    Layer {idx}: dW mean={dW.mean():.2e}, std={dW.std():.2e}, "
                  f"min={dW.min():.2e}, max={dW.max():.2e}")
    # -----------------------
    # Predict
    # -----------------------
    def predict(self, X):
        probs = self.forward(X, training=False)
        return np.argmax(probs), probs
    

    
    def save_model(self, path="trained_model/cnn_model_basic.npz"):
        """Save CNN model architecture and weights to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
        # Save architecture/config
        config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'conv_layers': self.conv_layers_config,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate
        }
    
        # Save weights
        weights = {}
        for i, layer in enumerate(self.layers):
            if layer['type'] == 'conv':
                weights[f"W{i}"] = layer['filters']
                weights[f"b{i}"] = layer['biases']
            elif layer['type'] in ['dense', 'output']:
                weights[f"W{i}"] = layer['weights']
                weights[f"b{i}"] = layer['biases']
    
        # Save both config and weights in npz
        np.savez(path, config=json.dumps(config), **weights)
        print(f"[INFO] Model saved to {path}")

    # ------------------------------------------------------
    # Predictions metrics & Evaluation
    # ------------------------------------------------------
    def get_training_metrics(self,X_test,Y_test):
      y_pred = []
      for X in X_test:
          pred_class, _ = self.predict(X)
          y_pred.append(pred_class)
      
      # Convert to numpy array
      y_pred = np.array(y_pred)
      
      # Accuracy
      acc = accuracy_score(y_test_labels, y_pred)
      print(f"\n[Test Accuracy] {acc:.4f}")
      
      # Confusion Matrix
      cm = confusion_matrix(y_test_labels, y_pred)
      print("\nConfusion Matrix:")
      print(cm)
      
      # Detailed per-class report
      print("\nPer-class results:")
      for cls in range(self.num_classes):
          correct = cm[cls, cls]
          total = cm[cls].sum()
          wrong = total - correct
          print(f"Class {cls}: Total={total}, Correct={correct}, Wrong={wrong}") 
      return acc

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


Model = load_weights(CNNModelTraining, r"C:\Users\clear\Desktop\BSC Computer Science hons\Hons Project 2025\D09\submission\WebApplicationPrototype\static\trained_model\cnn_model_basic.npz")

