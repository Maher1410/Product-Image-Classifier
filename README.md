# Product Image Classifier for Slash AI Internship

## Overview

This project aims to develop a machine learning model capable of classifying product images into predefined categories, enhancing the intelligence of an e-commerce app. By automating the categorization process, we improve the user experience, streamline product listing, and contribute to better inventory management.

## Approach

### 1. Dataset Preparation

We started by downloading a diverse dataset of product images from the Slash platform, organized into categories like Fashion, Nutrition, Accessories, etc. These images were then resized and normalized to maintain consistency, and data augmentation techniques were employed to enrich the dataset and reduce overfitting.

### 2. Model Selection and Training

Leveraging TensorFlow and Keras, we opted for a Sequential model incorporating Convolutional Neural Networks (CNNs). The model architecture includes Conv2D layers for feature extraction, MaxPooling2D layers for dimensionality reduction, and Dense layers for classification. BatchNormalization was applied to ensure faster convergence, and Dropout layers were included to prevent overfitting.

We used the Adam optimizer, categorical crossentropy as the loss function, and accuracy as the metric for model performance evaluation. The model was trained on a split dataset comprising 80% training data and 20% validation data.

### 3. Evaluation and Testing

The model's effectiveness was gauged through its performance on a separate test set, not seen by the model during training or validation. We used accuracy as the primary metric and employed a confusion matrix to visualize the model's classification capabilities across different categories.

### 4. Fine-Tuning and Optimization

Based on initial test results, we fine-tuned the model by adjusting parameters like the learning rate, number of epochs, and the model architecture. This iterative process aimed to balance model complexity with performance, ensuring high accuracy while avoiding overfitting.

## Functionalities

- **Automatic Image Categorization**: Automates the process of sorting product images into predefined categories, reducing manual workload and improving efficiency.
- **High Accuracy**: Achieves high accuracy in classification tasks, ensuring that products are categorized correctly for better inventory management and user search experience.
- **Scalability**: Designed to be scalable, the model can be trained on additional data or categories as the e-commerce platform grows.
- **User-Friendly Interface**: Includes a simple, intuitive interface for users to upload images for classification, making it accessible to individuals with varying technical backgrounds.

## Technologies Used

- Python
- TensorFlow and Keras
- OpenCV for image processing
- Pandas and Numpy for data manipulation
- Matplotlib and Seaborn for visualization

## How to Run the Project

Instructions on setting up the environment, downloading dependencies, and running the project are provided in the accompanying documentation.

## Future Work

- Explore more advanced model architectures like ResNet or EfficientNet for potential improvements in accuracy.
- Implement additional data augmentation techniques to further enrich the training dataset.
- Expand the dataset to include more categories and diverse images for enhanced model robustness.

## Conclusion

This project demonstrates the potential of machine learning in transforming e-commerce platforms, making product categorization more efficient and reliable. As technology advances, such applications will become increasingly integral to the digital marketplace.

## Contact

For any inquiries or contributions, please contact [Aly Maher] at [aly.abdelrahman@gu.edu.eg].

