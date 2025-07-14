# 🧠 Hate Speech Detector Using Bidirectional LSTM

## 📌 Overview
This project implements a deep learning solution for classifying text comments into three categories: **Hate Speech**, **Offensive Language**, and **Neutral**. The core of the solution is a Bidirectional Long Short-Term Memory (BiLSTM) neural network, chosen for its effectiveness in processing sequential data like text. The project covers the entire machine learning pipeline from data loading and preprocessing to model training, evaluation, and custom predictions.

---

## 🧰 Features

- 🧼 Robust text preprocessing (stopword removal, lemmatization)
- ⚖️ Dataset balancing with sampling strategies
- 📊 Word cloud visualization for lexical analysis
- 🏗️ Bidirectional LSTM with dropout & regularization
- 🔍 Performance metrics: accuracy, loss, confusion matrix
- 🔮 Prediction utility with per-class confidence scoring

---

## 📁 Dataset Details

| Label      | Meaning     | Sample Size |
|------------|-------------|-------------|
| `0`        | Hate        | 18,858        |
| `1`        | Offensive   | 22,144       |
| `2`        | Neutral     | 8,586        |

Stored at: `dataset\Labeled_Dataset_v1.csv`

---

## 📊 Results & Evaluation
✅ Validation Accuracy: ~86%

🔥 Confusion matrix heatmap

📈 Loss and accuracy graphs

📜 Classification report across all labels

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.x
* pip (Python package installer)
* Google Colab (recommended for easy setup and GPU access) or a local environment with Jupyter Notebook.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ksrawat19/hate-speech-detector.git
    cd hate-speech-detector
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **NLTK Data Downloads:**
    The notebook automatically handles NLTK data downloads. Ensure your internet connection is active when running the notebook for the first time to download `stopwords`, `omw-1.4`, `wordnet`, and `punkt`.

    ---

    ## 🏃 Run

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook notebooks/hate_speech_detector_main.ipynb
    ```
    Or, if using Google Colab, upload the `hate_speech_detector_main.ipynb` file directly.

2.  **Run All Cells:**
    Execute all cells in the notebook from top to bottom. The notebook guides you through:
    * Data loading and initial distribution visualization.
    * Applying class balancing (check `sampling_config` for current strategy).
    * Text preprocessing and word cloud generation.
    * Model tokenization, padding, building, and training.
    * Evaluation metrics (classification report, confusion matrix, validation accuracy).
    * Demonstrating predictions on example sentences.


---
## 💡 Future Work & Improvements

* **Pre-trained Word Embeddings:** Experiment with pre-trained word embeddings (e.g., GloVe, Word2Vec, FastText) to potentially improve model performance, especially on smaller datasets.
* **Advanced Models:** Explore more complex architectures like Transformer models (e.g., BERT, RoBERTa) for state-of-the-art results, potentially via fine-tuning.
* **Cross-Validation:** Implement k-fold cross-validation for more robust and reliable model evaluation.
* **Deployment:** Build a simple web application (e.g., using Flask or FastAPI) to deploy the model for real-time predictions.
* **Expanded Dataset:** Incorporate more diverse and larger datasets to enhance generalization.


---

Thanks for stopping by! 🚀✨
