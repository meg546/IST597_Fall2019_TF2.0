""" 
author:-aam35 and meg546
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from utils import mnist_reader
from sklearn.model_selection import train_test_split

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img_size = 28
img_shape = (28, 28)
img_size_flat = img_size * img_size 
n_classes = 10

def load_data(val_ratio=0.1):
    fmnist_folder = 'data/fashion'
    X_train, y_train = mnist_reader.load_mnist(fmnist_folder, kind='train')
    X_test, y_test = mnist_reader.load_mnist(fmnist_folder, kind='t10k')

    X_train, X_test = X_train.astype('float32') / 255.0, X_test.astype('float32') / 255.0
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42)

    y_train_oh, y_val_oh, y_test_oh = np.eye(10)[y_train], np.eye(10)[y_val], np.eye(10)[y_test]

    return (X_train, y_train, y_train_oh,
            X_val, y_val, y_val_oh,
            X_test, y_test, y_test_oh)

class LogisticRegression:
    def __init__(self, learning_rate=0.001, optimizer="adam", l2_lambda=0.0):
        self.w = tf.Variable(tf.random.normal([img_size_flat, n_classes], stddev=0.01))
        self.b = tf.Variable(tf.zeros([n_classes]))
        self.lr = learning_rate
        self.l2_lambda = l2_lambda

        if optimizer == "adam":
            self.optimizer = tf.optimizers.Adam(learning_rate)
        elif optimizer == "sgd":
            self.optimizer = tf.optimizers.SGD(learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = tf.optimizers.RMSprop(learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def model(self, X):
        return tf.matmul(X, self.w) + self.b

    def loss(self, y_true, logits):
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))
        reg = self.l2_lambda * tf.nn.l2_loss(self.w)
        return ce + reg

    def accuracy(self, y_true, logits):
        preds = tf.argmax(tf.nn.softmax(logits), 1)
        truth = tf.argmax(y_true, 1)
        return tf.reduce_mean(tf.cast(tf.equal(preds, truth), tf.float32))

    def train_epoch(self, dataset):
        total_loss, n_batches = 0, 0
        for X, y in dataset:
            with tf.GradientTape() as tape:
                logits = self.model(X)
                loss_val = self.loss(y, logits)
            grads = tape.gradient(loss_val, [self.w, self.b])
            self.optimizer.apply_gradients(zip(grads, [self.w, self.b]))
            total_loss += loss_val.numpy()
            n_batches += 1
        return total_loss / n_batches

def evaluate_accuracy(model, dataset):
    correct, total = 0, 0
    for X, y in dataset:
        logits = model.model(X)
        preds = tf.argmax(tf.nn.softmax(logits), axis=1)
        truth = tf.argmax(y, axis=1)
        correct += tf.reduce_sum(tf.cast(preds == truth, tf.float32)).numpy()
        total += y.shape[0]
    return correct / total

def evaluate_test(model, dataset, y_test_raw):
    preds_all, truth_all = [], []
    correct, total = 0, 0
    for X, y in dataset:
        logits = model.model(X)
        preds = tf.argmax(tf.nn.softmax(logits), axis=1).numpy()
        truths = tf.argmax(y, axis=1).numpy()
        preds_all.extend(preds)
        truth_all.extend(truths)
        correct += np.sum(preds == truths)
        total += y.shape[0]
    return correct / total, np.array(preds_all), np.array(truth_all)

def run_experiment(batch_size=128, n_epochs=30, optimizer="adam", val_ratio=0.1, l2_lambda=0.0, device=None, learning_rate=0.001):
    if device is None:
        gpus = tf.config.list_physical_devices('GPU')
        device = "/GPU:0" if gpus else "/CPU:0"
    
    X_train, y_train_raw, y_train, X_val, y_val_raw, y_val, X_test, y_test_raw, y_test = load_data(val_ratio)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(25000).batch(batch_size)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)


    print(f"Using device: {device}")
    
    with tf.device(device):
        model = LogisticRegression(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)

        history = {"loss": [], "train_acc": [], "val_acc": []}
        start_time = time.time()
        for epoch in range(n_epochs):
            loss_val = model.train_epoch(train_ds)
            train_acc = evaluate_accuracy(model, train_ds)
            val_acc = evaluate_accuracy(model, val_ds)
            history["loss"].append(loss_val)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Loss {loss_val:.4f} | Train {train_acc:.4f} | Val {val_acc:.4f}")

        elapsed = time.time() - start_time
        test_acc, preds, truths = evaluate_test(model, test_ds, y_test_raw)
        print(f"Final Test Accuracy: {test_acc:.4f} (Training time: {elapsed:.2f}s)")

    return {
        'model': model,
        'history': history,
        'test_acc': test_acc,
        'preds': preds,
        'truths': truths,
        'elapsed': elapsed,
        'test_ds': test_ds,
        'X_test': X_test,
        'y_test': y_test_raw
    }

def plot_images(images, y, yhat=None):
    assert len(images) >= 9 and len(y) >= 9
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if yhat is None:
            xlabel = f"True: {class_names[y[i]]}"
        else:
            xlabel = f"True: {class_names[y[i]]}\nPred: {class_names[yhat[i]]}"

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle("Fashion MNIST Predictions", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_weights(w):
    if isinstance(w, tf.Variable):
        w = w.numpy()
    
    w_min = w.min()
    w_max = w.max()

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        image = w[:, i].reshape(img_shape)
        ax.set_title(f"{class_names[i]}", fontsize=10)
        ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle("Learned Weight Vectors for Each Class", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_training_history(results_dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for name, results in results_dict.items():
        hist = results['history']
        axes[0].plot(hist['loss'], label=name)
        axes[1].plot(hist['train_acc'], label=name)
        axes[2].plot(hist['val_acc'], label=name)
    
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Validation Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def visualize_sample_predictions(results):
    model = results['model']
    X_test = results['X_test']
    y_test = results['y_test']
    
    sample_X = X_test[:9]
    sample_y = y_test[:9]
    
    logits = model.model(sample_X)
    preds = tf.argmax(tf.nn.softmax(logits), axis=1).numpy()
    
    plot_images(sample_X, sample_y, yhat=preds)

if __name__ == "__main__":
    print(" Fashion MNIST Logistic Regression - Assignment 2")

    # Experiment 1: Compare Optimizers
    print("\n### Experiment 1: Comparing Optimizers ###")
    optimizer_results = {}
    for opt in ["adam", "sgd", "rmsprop"]:
        print(f"\n--- Training with {opt.upper()} ---")
        results = run_experiment(
            optimizer=opt,
            learning_rate=0.01 if opt == 'sgd' else 0.001,
            n_epochs=30,
            batch_size=128
        )
        optimizer_results[opt] = results
    
    print("\n--- Optimizer Comparison Summary ---")
    for opt, res in optimizer_results.items():
        print(f"{opt.upper()}: Test Acc={res['test_acc']:.4f}, Time={res['elapsed']:.2f}s")
    
    plot_training_history(optimizer_results)
    
    # Experiment 2: Different Batch Sizes
    print("\n### Experiment 2: Effect of Batch Size ###")
    batch_results = {}
    for bs in [32, 64, 128, 256]:
        print(f"\n--- Batch size={bs} ---")
        results = run_experiment(batch_size=bs, n_epochs=20, optimizer="adam")
        batch_results[f"batch_{bs}"] = results
    
    print("\n--- Batch Size Comparison ---")
    for name, res in batch_results.items():
        print(f"{name}: Test Acc={res['test_acc']:.4f}, Time={res['elapsed']:.2f}s")
    
    plot_training_history(batch_results)
    
    # Experiment 3: Train/Val Split
    print("\n### Experiment 3: Effect of Train/Val Split ###")
    split_results = {}
    for ratio in [0.05, 0.1, 0.2]:
        print(f"\n--- Val ratio={ratio} ---")
        results = run_experiment(val_ratio=ratio, n_epochs=20, optimizer="adam")
        split_results[f"val_{int(ratio*100)}%"] = results
    
    plot_training_history(split_results)
    
    # Experiment 4: Longer Training (Check Overfitting)
    print("\n### Experiment 4: Longer Training (with/without Regularization) ###")
    print("\n--- No Regularization ---")
    no_reg = run_experiment(n_epochs=50, optimizer="adam", l2_lambda=0.0)
    
    print("\n--- With L2 Regularization ---")
    with_reg = run_experiment(n_epochs=50, optimizer="adam", l2_lambda=0.01)
    
    reg_results = {"no_reg": no_reg, "with_reg": with_reg}
    plot_training_history(reg_results)
    
    # Experiment 5: GPU vs CPU
    print("\n### Experiment 5: GPU vs CPU Performance ###")
    print("\n--- CPU ---")
    cpu_results = run_experiment(device="/CPU:0", n_epochs=10, optimizer="adam")
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("\n--- GPU ---")
            gpu_results = run_experiment(device="/GPU:0", n_epochs=10, optimizer="adam")
            print(f"\nSpeedup: {cpu_results['elapsed'] / gpu_results['elapsed']:.2f}x")
    except:
        print("No GPU available")
    
    best_model = optimizer_results['adam']
    
    print("\n### Visualizing Best Model (Adam) ###")
    plot_weights(best_model['model'].w)
    visualize_sample_predictions(best_model)
    plot_confusion_matrix(best_model['truths'], best_model['preds'])
    
    # Experiment 6: Compare with SVM and Random Forest
    print("\n### Experiment 6: Comparison with SVM and Random Forest ###")
    X_train, y_train_raw, _, _, _, _, X_test, y_test_raw, _ = load_data()
    
    print("Training SVM...")
    svm = LinearSVC(max_iter=5000)
    svm.fit(X_train, y_train_raw)
    svm_acc = accuracy_score(y_test_raw, svm.predict(X_test))
    print(f"SVM Test Accuracy: {svm_acc:.4f}")
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train_raw)
    rf_acc = accuracy_score(y_test_raw, rf.predict(X_test))
    print(f"Random Forest Test Accuracy: {rf_acc:.4f}")
    
    print(f"Logistic Regression Test Accuracy: {best_model['test_acc']:.4f}")
    
    # Experiment 7: t-SNE Visualization of Weights
    print("\n### Experiment 7: t-SNE Clustering of Weight Vectors ###")
    W = best_model['model'].w.numpy().T  # Shape: (10, 784)
    
    print("Computing t-SNE embedding...")
    emb = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(W)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:,0], emb[:,1], s=200, c=range(10), cmap='tab10', alpha=0.6)
    for i, txt in enumerate(class_names):
        plt.annotate(txt, (emb[i,0], emb[i,1]), fontsize=12, ha='center')
    plt.title("t-SNE Visualization of Learned Weight Vectors")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print(" All Experiments Completed!")
    print("=" * 70)
