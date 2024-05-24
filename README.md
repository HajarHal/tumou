# Brain Tumor Classification Project

## Overview

This project focuses on the classification of brain tumors using deep learning techniques. By leveraging Convolutional Neural Networks (CNNs), we aim to accurately distinguish between four types of brain conditions based on MRI images:

1. **Glioma Tumor**
2. **Meningioma Tumor**
3. **Pituitary Tumor**
4. **No Tumor**

The goal is to provide a reliable tool for early detection and diagnosis of brain tumors, aiding medical professionals in making informed decisions.

## Dataset

The dataset used in this project is a publicly available collection of MRI images. It includes thousands of images categorized into the aforementioned four classes. Each image is resized to 150x150 pixels to maintain consistency in model training and evaluation.

- **Training Data**: Images used to train the CNN model.
- **Testing Data**: Images used to evaluate the performance of the trained model.

## Model Architecture

The project employs a Convolutional Neural Network (CNN) for image classification. The architecture includes:

- Multiple convolutional layers with ReLU activation
- MaxPooling layers to reduce spatial dimensions
- Dropout layers to prevent overfitting
- Fully connected dense layers for final classification

The model is compiled using the `categorical_crossentropy` loss function and the `Adam` optimizer, with accuracy as the evaluation metric.

## Training and Evaluation

The model is trained over 20 epochs with a validation split to monitor performance and adjust the training process. Key metrics such as training accuracy, validation accuracy, training loss, and validation loss are tracked and visualized to ensure the model's effectiveness.

## Application Deployment

An interactive web application is developed using Streamlit. This application allows users to upload MRI images and receive instant predictions on the type of brain tumor. The user-friendly interface aims to make the diagnostic tool accessible to medical professionals and researchers.

## Results

The trained CNN model demonstrates high accuracy in classifying the different types of brain tumors. Performance metrics include:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

These metrics are evaluated on the test set to validate the model's effectiveness.

## Future Work

Potential improvements and future directions for the project include:

- Expanding the dataset to include more diverse MRI images.
- Experimenting with different CNN architectures and hyperparameters.
- Integrating additional diagnostic features such as tumor segmentation and volume estimation.

## Conclusion

This project showcases the potential of deep learning in medical image analysis, providing a valuable tool for the early detection of brain tumors. By combining advanced machine learning techniques with accessible deployment methods, we aim to support and enhance medical diagnostic processes.

## Document (Rapport)
https://drive.google.com/file/d/133MmDaWGjhFfhUWaVqCGC3FkDYaFMwuu/view?usp=drive_link
---
## Lien d'application
https://hajarhal-tumou-app-axpqcv.streamlit.app/

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/brain-tumor-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd brain-tumor-classification
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

Upload an MRI image and get the classification result instantly!

---

## Contact

For any questions or feedback, please open an issue or contact me at [hajarhalmaoui.1@gmail.com].
