# Plant-Disease-detection-system-
# Plant Disease Detection using Deep Learning
# OVERVIEW:
This repository contains a Deep Learning project that uses a Convolutional Neural Network (CNN) to detect and classify plant diseases from images of leaves. By automating the diagnosis process, this model can help in early identification of crop diseases, facilitating better agricultural management and yield protection.

# Dataset
The model is trained on the New Plant Diseases Dataset, which contains images of healthy and diseased leaves categorized into 38 different classes (including Apple, Corn, Grape, Potato, Tomato, etc.).

# Dataset Link:https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset


Tech Stack
Python 3.x

TensorFlow / Keras (for building and training the deep learning model)

Pandas & NumPy (for data manipulation)

Matplotlib & Seaborn (for data visualization)

Scikit-Learn (for model evaluation metrics)

# Project Workflow
# Data Preprocessing:

Images are loaded directly from the directory using tf.keras.utils.image_dataset_from_directory.

The images are resized to 128x128 pixels and grouped into batches of 32.

# Model Architecture:

Built a custom CNN from scratch using the Sequential API.

Consists of multiple stacked Conv2D and MaxPooling2D layers with increasing filters (32, 64, 128, 256, 512) to extract complex features.

Employs Dropout layers to prevent overfitting.

Fully connected Dense layers at the end output the probabilities for the 38 classes using the softmax activation function.

# Model Training:

Compiled with the Adam optimizer (learning rate = 0.0001) and categorical_crossentropy loss.

Trained for 10 epochs.

Evaluation & Visualization:

Plotted training and validation accuracy over epochs.

Evaluated model performance using Scikit-Learn's classification_report and confusion_matrix.

# Results
The CNN model achieved excellent performance on the dataset:

Training Accuracy: ~98.1%

Validation Accuracy: ~95.0%

The detailed classification report and confusion matrix indicate that the model effectively distinguishes between the 38 different classes with high precision and recall.

# Saved Model
The trained model weights and architecture are saved as trained_model.keras after training, which can be loaded later for making predictions on new, unseen leaf images without needing to retrain.
