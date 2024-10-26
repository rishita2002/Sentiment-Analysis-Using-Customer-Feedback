
# Sentiment Analysis on Customer Feedback

This repository contains a project focused on sentiment analysis of customer feedback collected from e-commerce platforms. The project utilizes machine learning and deep learning techniques to classify feedback as positive or negative. With models like **Long Short-Term Memory (LSTM)** and **Bidirectional Encoder Representations from Transformers (BERT)**, the solution achieves high accuracy in understanding customer sentiments.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Project Overview
Understanding customer feedback is essential for businesses to improve product quality and enhance customer satisfaction. This project provides an automated solution to classify feedback sentiments, helping businesses make informed decisions based on customer reviews.

**Key Features:**
- Classification of customer feedback into positive or negative sentiment.
- Integration of deep learning models: **LSTM** for sequential data processing and **BERT** for contextual understanding.
- A user-friendly web interface for real-time sentiment analysis using Flask.

## Dataset
The dataset for this project is collected from the [Hugging Face Open Source Library](https://huggingface.co/datasets) and contains Amazon product reviews in English. Key fields include:
- `review_id`: Unique identifier for each review
- `product_id`: Identifier for the product being reviewed
- `review_body`: Text content of the review
- `stars`: Rating given by the customer (1-5)
- `product_category`: Category of the product

## Model Architecture
### 1. **LSTM (Long Short-Term Memory)**
   - LSTM is effective in understanding the sequential nature of text data, capturing dependencies across words in reviews.
   - Architecture includes an embedding layer, LSTM layers, and dense layers for classification.

### 2. **BERT (Bidirectional Encoder Representations from Transformers)**
   - BERT captures contextual meaning and semantics by processing text bidirectionally.
   - Pre-trained on a vast corpus, BERT’s embeddings are fine-tuned on our dataset for precise sentiment classification.

## Setup and Installation
1. **Clone the Repository**
    ```bash
    git clone https://github.com/rishita2002/Sentiment-Analysis-Using-Customer-Feedback.git
    cd Sentiment-Analysis-Using-Customer-Feedback
    ```

2. **Install Dependencies**
    Ensure you have Python 3.8+ installed. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Dataset**
    Download the dataset from Hugging Face or any other source of customer reviews, and place it in the `/data` directory.

4. **Run the Application**
    Launch the Flask application:
    ```bash
    python app.py
    ```
    Visit `http://127.0.0.1:5000` in your browser to access the sentiment analysis interface.

## Usage
- **Single Review Input:** Enter a single review in the text box and click "Analyze" to view the sentiment classification and probability score.
- **Batch Processing:** Upload a `.csv` file containing reviews, and the application will process and display the results for each entry.

## Results
The model achieves high accuracy in distinguishing between positive and negative sentiments. Evaluation metrics:
- **Accuracy:** 80.34%
- **Precision, Recall, and F1 Score:** Varies based on model architecture; detailed analysis available in the report.


## Future Work
Potential areas for expansion:
- **Multilingual Support:** Extend sentiment analysis to multiple languages.
- **Model Optimization:** Experiment with hybrid models and fine-tune hyperparameters.
- **Advanced Sentiment Categories:** Incorporate a broader range of sentiments, such as neutral or mixed emotions.


Dataset link : https://drive.google.com/drive/u/0/folders/1Xym22vc-guTMvPGI00vOUz_hszmB3dPU
