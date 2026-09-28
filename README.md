# MLOps Image Classifier

An end-to-end image classification project built with **TensorFlow, Keras, Streamlit, and Docker**. The application uses a Convolutional Neural Network (CNN) to classify images into six different scene categories.

## Project Overview

This project demonstrates a complete machine learning workflow, starting from dataset preparation and model training to application development and containerization.

The trained model can classify images into:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

The Streamlit application provides an interactive interface where users can upload an image and receive the predicted scene along with the model's confidence score.

## Features

* Image classification using a CNN
* Six scene categories
* Image upload through Streamlit
* Prediction and confidence score
* Model information dashboard
* Responsive custom UI
* TensorFlow/Keras model
* Docker containerization
* Ready for Kubernetes and CI/CD integration

## Tech Stack

| Technology   | Purpose                          |
| ------------ | -------------------------------- |
| Python       | Programming language             |
| TensorFlow   | Machine Learning / Deep Learning |
| Keras        | Neural Network API               |
| NumPy        | Numerical processing             |
| Pillow       | Image processing                 |
| Streamlit    | Web application                  |
| Docker       | Containerization                 |
| Git & GitHub | Version control                  |

## Dataset

The project uses the **Intel Image Classification dataset**.

The dataset contains six scene categories:

```text
buildings
forest
glacier
mountain
sea
street
```

Total images used:

**14,034**

Dataset split:

* Training: 11,228 images
* Validation: 2,806 images

## Model Architecture

The image classifier is a Convolutional Neural Network consisting of:

```text
Input Image
    ↓
Rescaling
    ↓
Conv2D - 32 filters
    ↓
MaxPooling2D
    ↓
Conv2D - 64 filters
    ↓
MaxPooling2D
    ↓
Conv2D - 128 filters
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense - 128 neurons
    ↓
Dropout - 0.5
    ↓
Dense - 6 neurons
    ↓
Softmax
```

### Input

```text
150 × 150 × 3
```

### Output

```text
6 classes
```

## Model Performance

The trained model achieved approximately:

**82.86% validation accuracy**

The model was trained for **10 epochs** using the Adam optimizer and sparse categorical cross-entropy loss.

## Project Structure

```text
mlops-image-classifier/
│
├── app/
│   └── app.py
│
├── data/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
│
├── models/
│   └── image_classifier.keras
│
├── src/
│   └── train.py
│
├── notebooks/
│
├── predict.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ali-Husnain-AI-ML/mlops-image-classifier.git
```

```bash
cd mlops-image-classifier
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

Windows PowerShell:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Streamlit

```bash
streamlit run app/app.py
```

The application will open in your browser.

## Run with Docker

Build the Docker image:

```bash
docker build -t mlops-image-classifier .
```

Run the container:

```bash
docker run -p 8501:8501 mlops-image-classifier
```

Then open:

```text
http://localhost:8501
```

## Prediction

The project also includes a standalone prediction script:

```bash
python predict.py
```

The script loads the trained model and predicts the scene category of an image.

Example:

```text
Prediction: mountain
Confidence: 92.12%
```

## MLOps Workflow

The project is designed around the following workflow:

```text
Dataset
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Saved Model
   ↓
Prediction Pipeline
   ↓
Streamlit Application
   ↓
Docker
   ↓
Kubernetes
   ↓
CI/CD
   ↓
Cloud Deployment
```

## Future Improvements

Planned MLOps improvements include:

* Kubernetes deployment
* CI/CD pipeline
* Automated Docker builds
* Model versioning
* Cloud deployment
* Monitoring
* Automated testing
* Model performance tracking

## Author

**Muhammad Ali Husnain**

BS Computer Science Graduate
Aspiring AI / Machine Learning Engineer

GitHub:
https://github.com/Ali-Husnain-AI-ML

LinkedIn:
https://www.linkedin.com/in/m-ali-husnain-5583a1367

## License

This project is intended for educational and portfolio purposes.
