"""
author:-aam35 and meg546
"""
import time
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

# Create data
NUM_EXAMPLES = 10000

seed = sum(ord(c) for c in "meg546")
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
print("Using seed:", seed)

class LinearRegressionExperiment:
  def __init__(self, num_examples: int = NUM_EXAMPLES, noise_level: float = 1.0):
    self.num_examples = num_examples
    self.noise_level = noise_level
    self.results = {}

  def generate_data(self, noise_type: str = "gaussian", noise_level: float = None) -> tuple[tf.Tensor, tf.Tensor]:
    if noise_level is None:
      noise_level = self.noise_level

    # Generate inputs
    X = tf.random.normal([self.num_examples])

    # Set noise type based on input
    if noise_type == "gaussian":
      noise = tf.random.normal([self.num_examples], stddev=noise_level)
    elif noise_type == "uniform":
      noise = tf.random.uniform([self.num_examples], minval=-noise_level*sqrt(3), maxval=noise_level*sqrt(3))
    else:
      noise = tf.random.normal([self.num_examples], stddev=noise_level)

    y = X * 3 + 2 + noise
    return X, y

  def initialize_weights(self, init_type: str = "normal") -> tuple[tf.Variable, tf.Variable]:
    if init_type == "normal":
      W = tf.Variable(tf.random.normal([]))
      b = tf.Variable(tf.zeros([]))
    elif init_type == "uniform":
      W = tf.Variable(tf.random.uniform([], minval=-1, maxval=1))
      b = tf.Variable(tf.random.uniform([], minval=-1, maxval=1))
    elif init_type == "zeros":
      W = tf.Variable(tf.zeros([]))
      b = tf.Variable(tf.zeros([]))

    return W, b

  def prediction(self, x: tf.Tensor, W: tf.Variable, b: tf.Variable) -> tf.Tensor:
    return W * x + b

  def squared_loss(self, y: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(y - y_predicted))

  def huber_loss(self, y: tf.Tensor, y_predicted: tf.Tensor, delta = None) -> tf.Tensor:
    if delta is None:
      delta = self.noise_level
    error = y - y_predicted
    small_error = tf.abs(error) <= delta
    small_error_loss = 0.5 * tf.square(error)
    big_error_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.reduce_mean(tf.where(small_error, small_error_loss, big_error_loss))

  def l1_loss(self, y: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(y - y_predicted))

  def hybrid_loss(self, y: tf.Tensor, y_predicted: tf.Tensor, alpha: float = 0.5) -> tf.Tensor:
    return alpha * self.l1_loss(y, y_predicted) + (1 - alpha) * self.squared_loss(y, y_predicted)

  def train_model(self, 
                  X: tf.Tensor, 
                  y: tf.Tensor, 
                  loss_function: str = "squared", 
                  learning_rate: float = 0.001, 
                  train_steps: int = 2000,
                  patience: int = 5,
                  weight_init: str = "normal",
                  add_weight_noise: bool = False,
                  add_lr_noise: bool = False,
                  noise_schedule: str = "per_epoch") -> dict:

    W, b = self.initialize_weights(weight_init)

    loss_functions = {
      "squared": self.squared_loss,
      "huber": self.huber_loss,
      "l1": self.l1_loss,
      "hybrid": self.hybrid_loss
    }
    loss_fn = loss_functions[loss_function]

    losses = []
    learning_rates = []
    patience_counter = 0
    best_loss = float('inf')
    current_lr = learning_rate

    start_time = time.time()

    for i in range(train_steps):
      with tf.GradientTape() as tape:
        y_pred = self.prediction(X, W, b)
        loss = loss_fn(y, y_pred)
      
      dW, db = tape.gradient(loss, [W, b])

      if noise_schedule == "per_epoch" and (i + 1) % 5 == 0:
          if add_lr_noise:
              noise_factor = tf.random.normal([]) * 0.05
              current_lr = current_lr * (1 + noise_factor)

          if add_weight_noise:
              W.assign_add(tf.random.normal([]) * 0.005)
              b.assign_add(tf.random.normal([]) * 0.005)
    
      W.assign_sub(current_lr * dW)
      b.assign_sub(current_lr * db)

      if loss < best_loss - 1e-3:
          best_loss = loss
          patience_counter = 0
      else:
          patience_counter += 1

      if patience_counter >= patience:
          current_lr *= 0.5
          patience_counter = 0
          print(f"Epoch {i}: Reducing learning rate to {current_lr}")

      losses.append(loss.numpy())
      learning_rates.append(current_lr)
    
    end_time = time.time()
    print(f"Init={weight_init}, Final W={W.numpy():.3f}, b={b.numpy():.3f}, Final Loss={losses[-1]:.4f}")

    return {
      "W": W.numpy(),
      "b": b.numpy(),
      "losses": losses,
      "learning_rates": learning_rates,
      "final_loss": losses[-1],
      "training_time": end_time - start_time,
      "convergence_epoch": len(losses)
    }

  def compare_cpu_gpu(self, X, y, loss_function="squared", train_steps=2000):
    devices = ["/CPU:0"]
    try:
      tf.config.list_physical_devices('GPU')
      devices.append('/GPU:0')
    except:
      pass
    
    times = {}
    for device in devices:
      with tf.device(device):
        start = time.time()
        result = self.train_model(X, y, loss_function=loss_function, train_steps=train_steps)
        end = time.time()
        times[device] = (end - start) / train_steps
    
    return times
  
  def run_with_multiple_seeds(self, X, y, seeds, **train_kwargs):
    results = []
    for s in seeds:
        tf.random.set_seed(s)
        np.random.seed(s)
        res = self.train_model(X, y, **train_kwargs)
        results.append(res["final_loss"])
    return {
        "mean_loss": np.mean(results),
        "std_loss": np.std(results),
        "all_losses": results
    }
  
  def run_experiments(self):
    print("Running experiments...")
    
    # ===== Generate base (noise-free) data =====
    X_clean, y_clean = self.generate_data(noise_type="gaussian", noise_level=0.0)
    X, y = self.generate_data()  # default includes Gaussian noise, noise_level=1.0

    # -------------------------
    # 1. Loss Functions
    # -------------------------
    print("\n1. Testing Different Loss Functions")
    loss_functions = ["squared", "huber", "l1", "hybrid"]
    for loss_function in loss_functions:
      print(f"Testing {loss_function} loss...")
      result = self.train_model(X, y, loss_function=loss_function, train_steps=2000)
      self.results[f"loss_{loss_function}"] = result

    # -------------------------
    # 2. Learning Rates
    # -------------------------
    print("\n2. Testing Different Learning Rates")
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    for learning_rate in learning_rates:
      print(f"Testing learning rate: {learning_rate}")
      result = self.train_model(X, y, learning_rate=learning_rate, train_steps=2000)
      self.results[f"learning_rate_{learning_rate}"] = result

    # -------------------------
    # 3. Weight Initializations
    # -------------------------
    print("\n3. Testing Different Weight Initializations")
    weight_initializations = ["normal", "uniform", "zeros"]
    for weight_initialization in weight_initializations:
      print(f"Testing {weight_initialization} weight initialization...")
      result = self.train_model(X, y, weight_init=weight_initialization, train_steps=2000)
      self.results[f"weight_init_{weight_initialization}"] = result

    # -------------------------
    # 4. Noise Types
    # -------------------------
    print("\n4. Testing Different Noise Types (Gaussian, Uniform, Exponential)")
    noise_types = ["gaussian", "uniform", "exponential"]
    for noise_type in noise_types:
      print(f"Testing {noise_type} noise...")
      X_noisy, y_noisy = self.generate_data(noise_type=noise_type)
      result = self.train_model(X_noisy, y_noisy, train_steps=2000)
      self.results[f"noise_{noise_type}"] = result
    
    # -------------------------
    # 5. Noise Levels
    # -------------------------
    print("\n5. Testing Different Noise Levels")
    noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    for noise_level in noise_levels:
      print(f"Testing {noise_level} noise level...")
      X_noisy, y_noisy = self.generate_data(noise_level=noise_level)
      result = self.train_model(X_noisy, y_noisy, train_steps=2000)
      self.results[f"noise_level_{noise_level}"] = result

    # -------------------------
    # 5b. With vs Without Data Noise
    # -------------------------
    print("\n5b. Testing With vs Without Data Noise (explicit)")
    result_clean = self.train_model(X_clean, y_clean, train_steps=2000)
    result_noisy = self.train_model(X, y, train_steps=2000)
    self.results["data_noise_off"] = result_clean
    self.results["data_noise_on"] = result_noisy

    # -------------------------
    # 6. Weight Noise During Training
    # -------------------------
    print("\n6. Testing Weight Noise During Training")
    results = self.train_model(X, y, add_weight_noise=True, train_steps=2000)
    self.results["weight_noise"] = results

    # -------------------------
    # 7. Learning Rate Noise During Training
    # -------------------------
    print("\n7. Testing Learning Rate Noise During Training")
    result = self.train_model(X, y, add_lr_noise=True, train_steps=2000)
    self.results["lr_noise"] = result

    # -------------------------
    # 8. Patience Scheduling
    # -------------------------
    print("\n8. Testing Patience Scheduling")
    result = self.train_model(X, y, patience=5, train_steps=2000)
    self.results["patience_scheduling"] = result

    print("\nAll experiments completed")
    return self.results

  
  def plot_prediction_fit(self, key="loss_squared"):
      """Visualize predicted vs true regression line."""
      if key in self.results:
          result = self.results[key]
          W, b = result["W"], result["b"]
          X, y = self.generate_data()
          y_pred = W * X + b
          plt.scatter(X, y, alpha=0.3, label="True Data")
          plt.scatter(X, y_pred, alpha=0.3, label="Predicted")
          plt.legend()
          plt.title("Prediction vs True Data")
          plt.show()
        
  def plot_results(self):
    # 1. Loss Functions Comparison
    plt.figure(figsize=(10, 6))
    for loss_function in ["squared", "l1", "huber", "hybrid"]:
      if f"loss_{loss_function}" in self.results:
        plt.plot(self.results[f"loss_{loss_function}"]["losses"], 
                  label=f"{loss_function} loss", linewidth=2)
    plt.title("Comparison of Loss Functions")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Learning Rates Comparison
    plt.figure(figsize=(10, 6))
    for lr in [0.0001, 0.001, 0.01, 0.1]:
      if f"learning_rate_{lr}" in self.results:
        plt.plot(self.results[f"learning_rate_{lr}"]["losses"], 
                  label=f"LR: {lr}", linewidth=2)
    plt.title("Effect of Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. Weight Initializations Comparison
    plt.figure(figsize=(10, 6))
    for init in ["normal", "uniform", "zeros"]:
      if f"weight_init_{init}" in self.results:
        plt.plot(self.results[f"weight_init_{init}"]["losses"], 
                  label=f"{init} init", linewidth=2)
    plt.title("Effect of Weight Initialization")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4. Noise Types Comparison
    plt.figure(figsize=(10, 6))
    for noise_type in ["gaussian", "uniform", "exponential"]:
      if f"noise_{noise_type}" in self.results:
        plt.plot(self.results[f"noise_{noise_type}"]["losses"], 
                  label=f"{noise_type} noise", linewidth=2)
    plt.title("Effect of Different Noise Types in Data")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5. Noise Levels Comparison
    plt.figure(figsize=(10, 6))
    for nl in [0.1, 0.5, 1.0, 2.0, 5.0]:
      if f"noise_level_{nl}" in self.results:
        plt.plot(self.results[f"noise_level_{nl}"]["losses"], 
                  label=f"Noise Level {nl}", linewidth=2)
    plt.title("Effect of Different Noise Levels in Data")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5b. With vs Without Data Noise
    plt.figure(figsize=(10, 6))
    if "data_noise_off" in self.results:
      plt.plot(self.results["data_noise_off"]["losses"], label="No Data Noise", linewidth=2)
    if "data_noise_on" in self.results:
      plt.plot(self.results["data_noise_on"]["losses"], label="With Data Noise", linewidth=2)
    plt.title("Effect of Adding Data Noise")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 6. Weight Noise Effect
    plt.figure(figsize=(10, 6))
    if "weight_noise" in self.results:
      plt.plot(self.results["weight_noise"]["losses"], label="With Weight Noise", linewidth=2)
    if "loss_squared" in self.results:
      plt.plot(self.results["loss_squared"]["losses"], label="Baseline", linewidth=2)
    plt.title("Effect of Adding Weight Noise")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 7. Learning Rate Noise Effect
    plt.figure(figsize=(10, 6))
    if "lr_noise" in self.results:
      plt.plot(self.results["lr_noise"]["losses"], label="With LR Noise", linewidth=2)
    if "loss_squared" in self.results:
      plt.plot(self.results["loss_squared"]["losses"], label="Baseline", linewidth=2)
    plt.title("Effect of Adding Learning Rate Noise")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 8. Patience Scheduling Effect
    if "patience_scheduling" in self.results:
      fig, ax1 = plt.subplots(figsize=(10, 6))
      ax1.plot(self.results["patience_scheduling"]["losses"], color="blue", linewidth=2, label="Loss")
      ax1.set_xlabel("Epoch")
      ax1.set_ylabel("Loss", color="blue")
      ax1.tick_params(axis="y", labelcolor="blue")
      ax1.set_yscale("log")
      ax2 = ax1.twinx()
      ax2.plot(self.results["patience_scheduling"]["learning_rates"], color="red", linewidth=2, label="Learning Rate")
      ax2.set_ylabel("Learning Rate", color="red")
      ax2.tick_params(axis="y", labelcolor="red")
      ax2.set_yscale("log")
      plt.title("Effect of Patience Scheduling")
      fig.tight_layout()
      plt.show()


def main():
  experiment = LinearRegressionExperiment()
  X, y = experiment.generate_data()
  
  # CPU vs GPU comparison
  times = experiment.compare_cpu_gpu(X, y, train_steps=2000)
  print("Per epoch times: ", times)

  seeds = [seed + i for i in range(5)]
  results = experiment.run_with_multiple_seeds(X, y, seeds, train_steps=2000)
  print("Reproducibility test: ", results)

  results = experiment.run_experiments()
  experiment.plot_results()
  experiment.plot_prediction_fit()

if __name__ == "__main__":
  main()