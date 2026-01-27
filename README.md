# Chest-X-Ray-Analyzer

## Authors

- Przemysław Bukała — @Przemekbukala  
- Piotr Waszak — @Waszka20  
- Rafał Wybraniec — @ralphy127  
- Filip Bętkowski — @filbet4


## Project Description

Chest-X-Ray-Analyzer is a simple web application that allows users to upload chest X-ray images and receive an initial automated analysis using a machine learning model. The system classifies X-ray images as *normal*, *pneumonia* or *tuberculosis* and presents the prediction together with a confidence score.

The project combines a PyTorch-based machine learning model with a Django web application.

## Application Overview

Main features include:

* User registration and authentication
* Uploading chest X-ray images (JPG format)
* Automated image classification using a trained ML model
* Visualization of model attention using heatmaps
* History of uploaded images and predictions per user


## Technology Stack

* **Backend:** Django (Python)
* **Machine Learning:** PyTorch
* **Frontend:** Django Templates (HTML/CSS)
* **Database:** PostgreSQL 




## Database

The project uses **PostgreSQL** as the main database.

Before running the application, make sure that:
- PostgreSQL is installed and running
- A database for the project has been created

Database connection settings (database name, user, password, host, port)
can be configured in:

**settings.py**

If needed, the database engine can be changed or reconfigured directly in the Django settings file.


## Setup and Configuration

To run the application locally, follow the steps below.

### 1. Virtual Environment (Optional)

Install virtual environment tool:

```bash
sudo apt install python3-virtualenv
```

Create virtual environment:

```bash
virtualenv venv --python=python3.9
```

Activate virtual environment:

* **Linux / macOS:**

  ```bash
  source venv/bin/activate
  ```
* **Windows:**

  ```bash
  .\venv\Scripts\activate
  ```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


### 3. Database Setup

Apply migrations:

```bash
python manage.py migrate
```

### 4. Run Application

Start the development server:

```bash
python manage.py runserver
```

The application will be available at:

```
http://127.0.0.1:8000/
```


## Machine Learning Model

### Model Location

The machine learning part of the project is located in:

**xray_analyzer/machine_learning/**


### Training the Model


For information, you can check out file:

**xray_analyzer/mashine_learning/help.py**

## Dataset

The machine learning model is trained using the **Chest X-Ray Dataset** available on Kaggle.
https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset



