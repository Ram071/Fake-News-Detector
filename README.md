Fake News Detector Using LSTM 📰
Overview

This project is a Fake News Detection system built using a Long Short-Term Memory (LSTM) neural network. The model is designed to classify news articles as real or fake based on their textual content.

The model was trained using a dataset containing approximately 20,000 training samples and evaluated on 5,000 test samples, achieving an accuracy of approximately 90% on the test data.

The project uses several popular Python and machine-learning libraries, including TensorFlow, Keras, Scikit-learn, and NLTK. A Streamlit web application is also included to provide a simple and interactive interface for making predictions.

Project Structure
Fake-News-Detector/
│
├── data/
│   └── Dataset files
│
├── models/
│   └── Model training and evaluation files
│
├── deploy/
│   └── Streamlit application files
│
└── README.md

Directory Description

data/ – Contains the dataset used for training and testing the model.

models/ – Contains the model-training code, data preprocessing, model development, and evaluation scripts.

deploy/ – Contains the files required to run the Streamlit web application.

Technologies and Dependencies

The project is built using the following technologies and libraries:

Python 3

TensorFlow – For building and training the neural network.

Keras – For developing the LSTM model.

Scikit-learn – For dataset splitting and model evaluation.

NLTK – For natural language processing and text preprocessing.

Streamlit – For creating the interactive web application.

NumPy – For numerical and array operations.

Pandas – For loading and processing the dataset.

Installation

Clone the repository and install the required dependencies:

pip install tensorflow scikit-learn nltk keras streamlit numpy pandas

Usage
1. Train the Model

Run the model-training script:

python src/model.py


This will preprocess the dataset, train the LSTM model, and evaluate its performance.

2. Evaluate the Model

The model evaluation is performed after training and includes metrics such as:

Accuracy

Confusion matrix

Precision

Recall

F1-score

You can run:

python src/model.py


to train and evaluate the model.

3. Run the Streamlit Application

Start the web application using:

streamlit run app/app.py


After running the command, Streamlit will provide a local URL where you can access the Fake News Detector through your web browser.

Results

The trained model achieved approximately 90% accuracy on the test dataset.

The result demonstrates that an LSTM-based neural network can be used to classify news text into real and fake categories. However, accuracy alone does not establish whether an individual news article is actually true or false; predictions should be treated as model outputs and independently verified.

Acknowledgments

The dataset used for this project was obtained from the Kaggle Fake News competition:

https://www.kaggle.com/c/fake-news/data

Special thanks to the open-source Python and machine-learning communities for the tools and libraries used in this project.

Happy Coding! 🚀
