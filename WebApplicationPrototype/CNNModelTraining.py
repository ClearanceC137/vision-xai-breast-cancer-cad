import numpy as np
#from ImageSegmentation import final_dataset  # Your dataset
import os
import json
import sys


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
    def __init__(self, input_shape, num_classes,
                 conv_layers=[(8,3), (16,3)],
                 hidden_units=[128,64],
                 dropout_rate=0.3,
                 leaky_alpha=0.01):
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

        # Flatten
        flattened_size = int(np.prod(in_shape))

        # --- Dense layers ---
        prev_units = flattened_size
        for units in self.hidden_units:
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
    # Forward pass
    # -----------------------
    def forward(self, x, training=True):
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
                layer['z'] = z
                out = np.where(z > 0, z, self.leaky_alpha * z)

                if training and self.dropout_rate > 0.0:
                    mask = (np.random.rand(*out.shape) > self.dropout_rate).astype(np.float32)
                    out *= mask / (1.0 - self.dropout_rate)

            elif layer['type'] == 'output':
                flat = out.flatten()
                layer['input'] = flat.copy()
                z = np.dot(layer['weights'], flat) + layer['biases']
                layer['z'] = z
                out = self._softmax(z)

        return out

    # -----------------------
    # Softmax
    # -----------------------
    def _softmax(self, z):
        z = np.array(z, dtype=np.float64)
        z = np.clip(z, -50.0, 50.0)
        z -= np.max(z)
        exps = np.exp(z)
        s = np.sum(exps)
        if s == 0:
            return np.ones_like(z) / len(z)
        return exps / (s + 1e-12)

    # -----------------------
    # Gradient clipping
    # -----------------------
    def _clip_grad(self, grad, max_norm=5.0):
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad *= max_norm / (norm + 1e-6)
        return grad

    # -----------------------
    # Convolution forward
    # -----------------------
    def _conv_forward(self, x, layer):
        H, W, F = layer['output_shape']
        k = layer['ksize']
        out = np.zeros((H, W, F))
        for f in range(F):
            filt = layer['filters'][f]
            bias = layer['biases'][f]
            for i in range(H):
                for j in range(W):
                    for c in range(x.shape[2]):
                        out[i, j, f] += np.sum(x[i:i+k, j:j+k, c] * filt[:, :, c])
                    val = out[i, j, f] + bias
                    out[i, j, f] = val if val > 0 else self.leaky_alpha * val
        return out

    # -----------------------
    # Max Pool forward/backward
    # -----------------------
    def _max_pool_forward(self, x, size=2, stride=2):
        H, W, C = x.shape
        out_h = H // size
        out_w = W // size
        out = np.zeros((out_h, out_w, C))
        switches = np.zeros_like(x, dtype=bool)
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    patch = x[h_start:h_start+size, w_start:w_start+size, c]
                    max_val = np.max(patch)
                    out[i, j, c] = max_val
                    switches[h_start:h_start+size, w_start:w_start+size, c] = (patch == max_val)
        return out, switches

    def _max_pool_backward(self, d_out, switches, pool_input_shape, size=2, stride=2):
        if d_out.ndim != 3:
            d_out = d_out.reshape((pool_input_shape[0]//size, pool_input_shape[1]//size, pool_input_shape[2]))
        H, W, C = pool_input_shape
        dX = np.zeros(pool_input_shape)
        out_h, out_w, _ = d_out.shape
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    dX[h_start:h_start+size, w_start:w_start+size, c] += \
                        d_out[i, j, c] * switches[h_start:h_start+size, w_start:w_start+size, c]
        return dX

    # -----------------------
    # Compute gradients
    # -----------------------
    def _compute_sample_grads(self, y_true):
        grads = [None] * len(self.layers)
        d_out = None
        for r_idx, layer in enumerate(reversed(self.layers)):
            idx = len(self.layers) - 1 - r_idx
            if layer['type'] == 'output':
                probs = self._softmax(layer['z'])
                d_out = probs - y_true
                dW = np.outer(d_out, layer['input'])
                db = d_out.copy()
                grads[idx] = {'dW': dW, 'db': db}
                d_out = np.dot(layer['weights'].T, d_out)

            elif layer['type'] == 'dense':
                d_activation = np.where(layer['z'] > 0, 1.0, self.leaky_alpha)
                dz = d_out * d_activation
                dW = np.outer(dz, layer['input'])
                db = dz.copy()
                grads[idx] = {'dW': dW, 'db': db}
                d_out = np.dot(layer['weights'].T, dz)

            elif layer['type'] == 'pool':
                pool_input_shape = layer['input'].shape
                out_h, out_w, C = layer['output_shape']
                if d_out.ndim == 1:
                    d_out = d_out.reshape((out_h, out_w, C))
                dX = self._max_pool_backward(d_out, layer['switches'], pool_input_shape)
                d_out = dX
                grads[idx] = None

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
                            local_mask = 1.0 if layer['output'][i,j,f]>0 else self.leaky_alpha
                            grad_val = local_mask * d_out[i,j,f]
                            for c in range(x.shape[2]):
                                patch = x[i:i+k, j:j+k, c]
                                grad_accum[:,:,c] += grad_val * patch
                                dX[i:i+k, j:j+k, c] += layer['filters'][f][:,:,c] * grad_val
                    dF_all[f] = grad_accum
                    db_all[f] = np.sum(np.where(layer['output'][:,:,f]>0,1.0,self.leaky_alpha)*d_out[:,:,f])
                grads[idx] = {'dF': dF_all, 'db_conv': db_all}
                d_out = dX
        return grads

    # -----------------------
    # Cross-entropy loss
    # -----------------------
    def cross_entropy(self, probs, y_true):
        probs = np.clip(probs, 1e-12, 1.0)
        if probs.ndim == 1:
            return -np.sum(y_true * np.log(probs))
        else:
            return -np.mean(np.sum(y_true * np.log(probs), axis=1))

    # -----------------------
    # Apply gradients
    # -----------------------
    def _apply_grads(self, grads_acc, lr):
        for idx, g in enumerate(grads_acc):
            if g is None:
                continue
            layer = self.layers[idx]
            if layer['type'] in ['dense','output']:
                dW = self._clip_grad(g['dW'])
                db = self._clip_grad(g['db'])
                layer['weights'] -= lr * dW
                layer['biases'] -= lr * db
            elif layer['type'] == 'conv':
                dF = self._clip_grad(g['dF'])
                db = self._clip_grad(g['db_conv'])
                layer['filters'] -= lr * dF
                layer['biases'] -= lr * db

    # -----------------------
    # Training
    # -----------------------
    def train(self, X, y_onehot, X_test, y_test, epochs=10, lr=0.01, batch_size=8):
        dataset_size = len(X)
        best_acc = 0.0
        best_weights = None

        print("[Training Params] :")
        print(" Learning Rate:", lr)
        print(" Drop Out Rate:", self.dropout_rate)
        print(" Num Epochs:", epochs)
        print(" Batch size:", batch_size)

        for epoch in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            X_shuf = X[indices]
            y_shuf = y_onehot[indices]

            total_loss = 0.0

            for i in range(0, dataset_size, batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]

                grads_acc = [None]*len(self.layers)
                batch_loss = 0.0
                n_samples = len(X_batch)

                for x_sample, y_sample in zip(X_batch, y_batch):
                    probs = self.forward(x_sample, training=True)
                    batch_loss += self.cross_entropy(probs, y_sample)
                    sample_grads = self._compute_sample_grads(y_sample)
                    for idx, g in enumerate(sample_grads):
                        if g is None:
                            continue
                        if grads_acc[idx] is None:
                            grads_acc[idx] = {}
                            for k in g:
                                grads_acc[idx][k] = np.zeros_like(g[k])
                        for k in g:
                            grads_acc[idx][k] += g[k]

                # average grads over batch
                for idx, g in enumerate(grads_acc):
                    if g is None:
                        continue
                    for k in g:
                        grads_acc[idx][k] /= float(n_samples)

                self._apply_grads(grads_acc, lr)
                total_loss += batch_loss

            # TODO: define get_training_metrics() externally or implement here
            accuracy = self.get_training_metrics(X_test, y_test)

            print(f"[EPOCH {epoch+1}/{epochs}] Loss={total_loss/dataset_size:.4f} Accuracy={accuracy:.4f}")

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

            # Learning rate decay
            lr *= 0.98

        print(f"[TRAIN] Best accuracy: {best_acc:.4f}")
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
    

    
    def save_model(self, path="trained_model/cnn_model.npz"):
        """Save CNN model architecture and weights to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
        # Save architecture/config
        config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'conv_layers': self.conv_layers_config,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rateconsidering 
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
      for cls in range(num_classes):
          correct = cm[cls, cls]
          total = cm[cls].sum()
          wrong = total - correct
          print(f"Class {cls}: Total={total}, Correct={correct}, Wrong={wrong}") 
      return acc

Model = load_weights(CNNModelTraining, "trained_model/cnn_model.npz")
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score


# ------------------------------------------------------
# Prepare Data
# ------------------------------------------------------
#bottleneck_features, class_labels = zip(*final_dataset)
#bottleneck_features = np.array(bottleneck_features, dtype=np.float32)
#class_labels = np.array(class_labels)

#num_classes = len(np.unique(class_labels))
#y_onehot = np.eye(num_classes)[class_labels]

#X_cnn = bottleneck_features
#input_shape = X_cnn[0].shape

# Split into 80% train, 20% test
#X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
#    X_cnn, y_onehot, class_labels, test_size=0.2, random_state=42, stratify=class_labels
#)

# ------------------------------------------------------
# Build & Train Model
# ------------------------------------------------------
#model = CNNModelTraining(
#    input_shape=input_shape,
#    num_classes=num_classes,
#    conv_layers=[(32,3),(64,3)],
#    hidden_units=[256,128],
#    dropout_rate=0.3
#)
#model.train(X_train, y_train, X_test, y_test, epochs=10, lr=1e-3, batch_size=32)
