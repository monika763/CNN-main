#  Deep Learning Assignment – CNN 

This project implements a **custom CNN model** trained on the iNaturalist dataset for a 10-class image classification task. It includes training, hyperparameter tuning using Weights & Biases (W&B), and model evaluation.


##  Project Structure

### 1.  Output Size Computation
- A helper function was created to **compute the output size** after each convolution and pooling operation.
- Ensures smooth dimension flow from convolutional layers to fully connected layers.

### 2.  Custom CNN Model
- `ConvNetModel` class:
  - **5 Convolutional Blocks**: Each follows the structure Conv → Activation → MaxPool
  - Supports **custom filter sizes, kernel sizes, activation functions**, and pooling strategy
  - Optionally includes **Batch Normalization** and **Dropout**
  - Ends with **a fully connected layer** and **output layer** with 10 output neurons (for 10 classes)

---

##  Data Handling

### `data_load()`
- Loads data and splits it into training and validation sets
- Parameters:
  - `batch_size`
  - `val_split`
  - `data_augmentation` (True/False)
### `test_data_load()`
 - Loads test data and prepare for test.

##  Training and Sweeps

### `model_train_val()`
- Trains the CNN model using:
  - Loss: CrossEntropyLoss
  - Optimizer: Adam
  - Default Learning Rate: 0.001

### `sweep()`
- Uses **W&B Bayesian Optimization sweeps** to tune hyperparameters:
  - Kernel sizes across 5 layers
  - Dropout values (0.2, 0.3)
  - Activation functions (`relu`, `mish`, `silu`, `gelu`)
  - Dense layer sizes (128, 256)
  - Batch normalization (True/False)
  - Filter organizations (various combinations)
  - Data augmentation (True/False)
  - Epochs (5, 10, 15)
- Efficient way to explore multiple model variants and configurations

---

##  Final Evaluation

### `test_evaluation_model()`
- Loads best weights from training
- Evaluates on test set
- Computes overall and class-wise accuracy and loss

### img_plot()

-Visualizes predictions

-Displays 30 images in a 10×3 grid with actual vs predicted labels




##  Dependencies

- Python 3.7+
- torch, torchvision
- wandb

---

## ▶ Run

```python
# To train
main()

# Run the sweep agent
wandb.agent(sweep_id, function=main, count=number of counts required)

---

