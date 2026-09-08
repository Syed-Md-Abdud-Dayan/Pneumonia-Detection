# Pneumonia Detection using CNN

## 1. Project Objective

The objective of this project is to develop a deep learning model that classifies chest X-ray images into two categories:

- **NORMAL**
- **PNEUMONIA**

The project follows a complete deep-learning workflow including exploratory data analysis, image preprocessing, model training, prediction, and evaluation.

---

## 2. Approach

The project follows an end-to-end supervised deep-learning workflow:

1. Load the chest X-ray dataset.
2. Inspect the dataset structure and class distribution.
3. Perform exploratory data analysis.
4. Preprocess the X-ray images.
5. Apply data augmentation to the training images.
6. Handle class imbalance using class weights.
7. Use a pretrained CNN model as the feature extractor.
8. Train the model using training and validation data.
9. Monitor training and validation Accuracy, Loss, Precision, and Recall.
10. Evaluate the best model on the independent test set.
11. Analyze the results using a Classification Report, Confusion Matrix, and ROC Curve.

---

## 3. Methodology

### 3.1 Dataset

The dataset contains chest X-ray images organized into three main sets:

```text
Archive/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The model performs binary classification:

```text
NORMAL    → Normal chest X-ray
PNEUMONIA → Chest X-ray associated with pneumonia
```

### 3.2 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset before model training.

The analysis includes:

- Dataset structure inspection.
- Class distribution analysis.
- Visualization of NORMAL and PNEUMONIA images.

The class distribution is important because an imbalance between classes can affect the model's learning and evaluation.

### 3.3 Image Preprocessing

The chest X-ray images are resized and preprocessed before being provided to the model.

Data augmentation is applied to the training images to introduce variation and help the model generalize better. The validation and test images are processed separately without training-time augmentation.

### 3.4 Class Imbalance

Class weights are used during training to reduce the effect of imbalance between the NORMAL and PNEUMONIA classes. This helps the model give appropriate importance to both classes.

### 3.5 CNN Model

The project uses a pretrained CNN-based model with additional classification layers for binary image classification.

The overall workflow can be represented as:

```text
Input Chest X-ray
        ↓
Pretrained CNN Feature Extractor
        ↓
Classification Layers
        ↓
Sigmoid Output
        ↓
NORMAL / PNEUMONIA
```

### 3.6 Model Training

The model is trained using training data while its performance is monitored using validation data.

The training process tracks:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Training Precision
- Validation Precision
- Training Recall
- Validation Recall

The best-performing model is saved and later evaluated using the independent test dataset.

### 3.7 Model Evaluation

The final model is evaluated using several performance metrics:

- **Loss** — measures the prediction error.
- **Accuracy** — measures the proportion of correctly classified images.
- **Precision** — measures how many predicted PNEUMONIA cases are actually PNEUMONIA.
- **Recall** — measures how many actual PNEUMONIA cases are correctly identified.
- **F1 Score** — provides a balanced measure of Precision and Recall.
- **ROC-AUC** — measures the model's ability to distinguish between the two classes across different classification thresholds.

A Classification Report, Confusion Matrix, and ROC Curve are also used for detailed evaluation.

---

## 4. Visualizations

### Class Distribution

![Class Distribution](Pneumonia_Detection_Images/class_distribution.png)

The **class distribution graph** shows the number of NORMAL and PNEUMONIA images in the dataset. It helps identify whether there is any imbalance between the two classes before training the model.

### Training and Validation Accuracy

![Accuracy](Pneumonia_Detection_Images/accuracy.png)

The **training and validation accuracy graph** shows how the model's classification accuracy changes across epochs. It helps compare the model's performance on the training data with its ability to generalize to unseen validation data.

### Training and Validation Loss

![Loss](Pneumonia_Detection_Images/loss.png)

The **training and validation loss graph** shows how the model's prediction error changes during training. It helps identify whether the model is learning effectively and whether signs of overfitting or underfitting are present.

### Training and Validation Precision

![Precision](Pneumonia_Detection_Images/precision.png)

The **training and validation precision graph** shows how accurately the model identifies PNEUMONIA cases among its positive predictions across different epochs. Comparing the training and validation curves helps evaluate how well precision generalizes beyond the training data.

### Training and Validation Recall

![Recall](Pneumonia_Detection_Images/recall.png)

The **training and validation recall graph** shows how effectively the model identifies actual PNEUMONIA cases across different epochs. Comparing both curves helps evaluate whether the model maintains its ability to detect positive cases on the validation data.

### Confusion Matrix

![Confusion Matrix](Pneumonia_Detection_Images/confusion_matrix.png)

The **confusion matrix** shows the number of NORMAL and PNEUMONIA images that were correctly and incorrectly classified.

### ROC Curve

![ROC Curve](Pneumonia_Detection_Images/roc_curve.png)

The **ROC curve and AUC** provide an additional view of the model's classification ability across different decision thresholds.

---

## 5. Findings

The model's performance is assessed using multiple evaluation metrics rather than accuracy alone.

### Test Set Evaluation

The evaluation metrics summarize the model's overall performance using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

| Metric | Result |
|---|---:|
| Loss | 0.2983 |
| Accuracy | 0.8798 (87.98%) |
| Precision | 0.8870 (88.70%) |
| Recall | 0.9256 (92.56%) |
| F1 Score | 0.9059 (90.59%) |
| ROC-AUC | 0.9462 (94.62%) |

### Classification Report

The **classification report** summarizes the model's performance for both NORMAL and PNEUMONIA classes using Precision, Recall, F1 Score, and Support.

```text
              precision    recall  f1-score   support

      NORMAL     0.8664    0.8034    0.8337       234
   PNEUMONIA     0.8870    0.9256    0.9059       390

    accuracy                         0.8798       624
   macro avg     0.8767    0.8645    0.8698       624
weighted avg     0.8792    0.8798    0.8788       624
```

The results show that the model performs well on both classes. The higher Recall for the PNEUMONIA class indicates that the model successfully identifies a large proportion of the actual PNEUMONIA images.

---

## 6. Limitations

- The model is trained and evaluated on the provided dataset and may not generalize perfectly to images from different hospitals, devices, or patient populations.
- Model performance can be affected by variations in chest X-ray images.
- Performance may be influenced by class imbalance.
- The model should not be considered a standalone clinical diagnostic system.

---

## 7. Possible Improvements

- **Fine-tuning:** Unfreeze additional pretrained layers and train them with a small learning rate.
- **Hyperparameter tuning:** Experiment with learning rates, batch sizes, dropout rates, and model parameters.
- **Alternative architectures:** Compare the model with other CNN architectures.
- **Improved data augmentation:** Experiment with additional medically appropriate image transformations.
- **Cross-validation:** Use cross-validation for a more robust performance estimate.
- **Larger datasets:** Train using more diverse chest X-ray datasets to improve generalization.
- **Threshold optimization:** Investigate different classification thresholds to balance Precision and Recall.
- **Explainable AI:** Use techniques such as Grad-CAM to visualize image regions that influence predictions.

---

## 8. Conclusion

This project developed a CNN-based model to classify chest X-ray images as NORMAL or PNEUMONIA. The complete workflow included exploratory data analysis, image preprocessing, class-weight handling, model training, and comprehensive evaluation.

The final model achieved **87.98% Accuracy**, **88.70% Precision**, **92.56% Recall**, **90.59% F1 Score**, and **94.62% ROC-AUC**.

The training and validation graphs, Classification Report, Confusion Matrix, and ROC Curve provide a comprehensive view of the model's performance.

---

## 9. How to Run

### 1. Open the Notebook

Open the notebook using **Google Colab** or Jupyter Notebook.

### 2. Prepare the Dataset

Ensure that the chest X-ray dataset follows the required directory structure:

```text
Archive/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Run the Notebook

Run the notebook cells **in order from top to bottom**.

The notebook will perform:

- Dataset loading
- Exploratory Data Analysis
- Image preprocessing
- Data augmentation
- Class-weight calculation
- Model construction
- Model training
- Training and validation visualizations
- Test-set evaluation
- Classification Report
- Confusion Matrix
- ROC Curve
- Final evaluation metrics

### 4. View the Results

After execution, the notebook will display all visualizations, evaluation metrics, the Classification Report, Confusion Matrix, and ROC Curve.

---

## 10. Tools and Technologies Used

- Python
- Google Colab / Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PIL
