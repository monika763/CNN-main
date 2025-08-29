## Deep Learning Assignment – Fine-Tuning Pre-trained Models (Part B)
This project demonstrates the fine-tuning of large pre-trained models (such as ResNet50) on the iNaturalist 12K dataset for a 10-class image classification task. The goal is to compare transfer learning techniques with training from scratch (Part A), and explore multiple fine-tuning strategies for efficiency and accuracy.

## Model Architecture
- Base Model: ResNet50 (pre-trained on ImageNet)

- Modified Final Layer: Fully connected layer changed to output 10 classes instead of 1000

- Device: CUDA-enabled GPU for faster training

## Fine-Tuning Strategies Explored
We tested several strategies to make training tractable while maximizing performance:

- Freezing all layers except the final classification layer

- Freezing the first 80% of layers, training the rest

- Unfreezing all layers and training the full network

## Implementation Steps
### pretrain_model(freeze_percent, freeze_all_except_last_layer, num_classes)
Loads a ResNet50 model

Modifies the last fully connected layer to match 10 classes

Allows partial or complete freezing of layers

data_load()
Loads and processes the iNaturalist dataset

Includes options for data augmentation, batch size, and validation split

### `data_load()`
- Loads data and splits it into training and validation sets
- Parameters:
  - `batch_size`
  - `val_split`
  - `data_augmentation` (True/False)
### `test_data_load()`
 - Loads test data and prepare for test.

### model_train_val()
Trains the model using:

- Loss: CrossEntropyLoss

- Optimizer: Adam

- Customizable number of epochs and learning rate

## Experiment Logging with W&B
Used Weights & Biases (wandb) for:

- Tracking metrics

- Logging training/validation accuracy

- Comparing model performances across fine-tuning strategies

## Evaluation
### test_evaluation_model()
- Evaluates best model from training phase

- Returns test accuracy and class-wise performance



## Results & Insights

Strategy	Validation Accuracy	Observations
Freeze all layers except the last layer	74.65%	Very efficient and accurate; fastest training
Freeze 80% layers and train remaining 20%	73.75%	Balanced trade-off; slightly slower but performs well
Unfreeze all layers	~72% (varied)	More expensive to train; no significant gain in performance
Freeze last FC layer only (train conv blocks)	Lower	Not effective; classifier not adapted to new classes
## Dependencies
Python 3.7+

torch, torchvision

wandb

matplotlib, numpy
## Run
```
wandb.init(project="DL_Assignment_2", name="pre_trainedmodel_rs_net")
model = pretrain_model(freeze_percent=0.8, num_classes=10)
trained_model = model_train_val(model, train_loader, val_loader, epochs=5)
wandb.finish()
```

