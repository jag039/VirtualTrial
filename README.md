# Classifier Model for Garment Classification

## 1. Introduction
The goal of this project was to train a deep learning model to classify clothing items into categories such as **Upper Garment**, **Lower Garment**, and other types. We aimed to structure classifications hierarchically by **gender** (e.g., men, women), **type** (e.g., shirt, pants), and **subcategories** (e.g., casual, formal). 

To achieve this, we utilized a pre-existing deep learning model and adapted it to improve its classification capabilities.

---

## 2. Model Selection and Adaptation
We based our implementation on a pre-existing model from GitHub: **[FarnooshGhadiri/Cloth_category_classifier](https://github.com/FarnooshGhadiri/Cloth_category_classifier)**. This model employs **ResNet-50** as a feature extractor, leveraging its strong capability in extracting image features.

### **Model Architecture**
The **ClothingClassifier** model consists of the following components:
- A **ResNet-50** backbone for feature extraction.
- A **fully connected layer** reducing the feature dimensionality to 512.
- A **presence classifier** that determines whether **topwear, bottomwear, or both** are present.
- Three **separate classifiers** for **topwear, bottomwear, and both**, which classify the detected clothing type into specific categories.

---

## 3. Training Strategy
We adapted the training pipeline to address a key limitation in the dataset: **each image in the DeepFashion dataset is labeled with only one category** (either topwear or bottomwear), which prevents the model from recognizing both garments when present in an image.

### **Training Process**
#### **Dataset Preparation**
- We used the **DeepFashion dataset** and applied data transformations to enhance model generalization.

#### **Training Loop**
- The **presence classifier** predicts if **topwear, bottomwear, or both** are present in an image.
- The **category classifiers** predict specific types of clothing within those broad categories.
- The **loss function** computes classification errors, and the model is optimized using the **Adam optimizer**.

---

## 4. Testing and Evaluation
We tested the model using a **separate test dataset**. 

### **Issues Encountered**
- The primary issue encountered was that the **presence classifier consistently flagged all three categories** (**topwear, bottomwear, both**) as present in every image. This indicates a potential flaw in the **loss function** or **training methodology**, which requires further debugging and refinement.
