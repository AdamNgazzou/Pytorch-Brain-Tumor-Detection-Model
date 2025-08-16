# Brain Tumor MRI Classification with Transfer Learning and Custom CNN

This notebook demonstrates building a brain tumor MRI classification model using both a pre-trained ResNet18 model and a custom Convolutional Neural Network (CNN) architecture trained from scratch.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Key Aspects

### Dataset

*   **Type:** Brain MRI Scans
*   **Classes:** 'negative' (no tumor), 'positive' (tumor)
*   **Size:** 3000 images total (2400 training, 600 testing)
*   **Preprocessing & Augmentation:**
    *   Resized to (224, 224)
    *   Random Rotation, Horizontal Flip, Brightness/Contrast
    *   Converted to Tensors
    *   Normalized

### Models & Training

*   **Model 1: Pre-trained ResNet18 (Transfer Learning / Fine-Tuning)**
    *   **Base Model:** Pre-trained **ResNet18**
    *   **Method:** **Transfer Learning / Fine-Tuning**
    *   **Modification:** Replaced the final layer with `nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))`
    *   **Why Fine-Tune?** Leverage features learned from a large dataset (ImageNet) and adapt them for brain MRI classification.
    *   **Training:** Trained for 10 epochs.

*   **Model 2: Custom SimpleCNN (Trained from Scratch)**
    *   **Architecture:** A simple CNN with convolutional, ReLU, and max pooling layers, followed by fully connected layers and dropout.
    *   **Method:** Trained from scratch on the brain MRI dataset.
    *   **Training:** Trained for 20 epochs.

*   **Shared Training Details:**
    *   **Loss Function:** `nn.CrossEntropyLoss`
    *   **Optimizer:** **SGD** (lr=0.001, momentum=0.9, weight_decay=0.0001)
    *   **Techniques:** Dropout, Weight Decay

### Evaluation & Results

*   **Metrics Used:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
*   **Key Results (Test Set):**
    *   **Pre-trained ResNet18:**
        *   Accuracy: **~99.67%**
        *   Precision: **~99.67%**
        *   Recall: **~99.67%**
        *   F1-score: **~99.67%**
    *   **Custom SimpleCNN:**
        *   Accuracy: **~97.17%**
        *   Precision: **~97.17%**
        *   Recall: **~97.17%**
        *   F1-score: **~97.17%**
*   **Visualizations:** Class distribution bar plot and Confusion Matrix.

### Conclusion

*   Both **transfer learning** with a pre-trained ResNet18 and training a **custom CNN from scratch** achieved comparable and good performance on the brain tumor MRI classification task.
*   The performance metrics indicate that both approaches were successful in classifying brain MRI images.

### Reproducibility

*   **Dataset:** [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection/data)
*   **Requirements:** PyTorch, torchvision, matplotlib, scikit-learn, torchmetrics, mlxtend.
*   Follow the code cells in order.
