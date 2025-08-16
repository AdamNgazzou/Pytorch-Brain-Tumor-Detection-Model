# Brain Tumor MRI Classification with Transfer Learning

This notebook demonstrates building a brain tumor MRI classification model using a pre-trained ResNet18 model.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)
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

### Model & Fine-Tuning

*   **Base Model:** Pre-trained **ResNet18**
*   **Method:** **Transfer Learning / Fine-Tuning**
*   **Modification:** Replaced the final layer with `nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))`
*   **Why Fine-Tune?** Leverage features learned from a large dataset (ImageNet) and adapt them for brain MRI classification.

### Training Process

*   **Epochs:** 10
*   **Loss Function:** `nn.CrossEntropyLoss`
*   **Optimizer:** **SGD** (lr=0.001, momentum=0.9, weight_decay=0.0001)
*   **Techniques:** Dropout, Weight Decay

### Evaluation & Results

*   **Metrics Used:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
*   **Key Result (Test Set):**
    *   Accuracy: **~99.67%**
    *   Precision: **~99.67%**
    *   Recall: **~99.67%**
    *   F1-score: **~99.67%**
*   **Improvements:** Fine-tuning significantly improved performance compared to training from scratch.
*   **Visualizations:** Class distribution bar plot and Confusion Matrix.

### Conclusion

*   Successfully built a brain tumor MRI classifier using **transfer learning**.
*   Fine-tuning **improved classification performance**.
*   Achieved **good performance metrics** on the test set.

### Reproducibility

*   **Dataset:** [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
*   **Requirements:** PyTorch, torchvision, matplotlib, scikit-learn, torchmetrics, mlxtend.
*   Follow the code cells in order.
