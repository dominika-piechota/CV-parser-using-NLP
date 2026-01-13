
# ATS INSIGHT - Intelligent Resume Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Regex-orange)
![ML](https://img.shields.io/badge/ML-Gradient%20Boosting-green)

## 01. Why do we need it?
Many companies use **ATS (Applicant Tracking Systems)** that automatically analyze and filter CVs. People looking for a job often submit well-prepared CVs but do not receive replies.

**The Problem:** It’s hard to know whether your document is ATS-compatible or contains the keywords that companies look for.

**My Solution:** An intelligent tool that reverses this process – predicting the "Fit Score" (1.0 - 5.0) against 11 different IT job roles.

## 02. Research Questions
This project aims to answer the following questions defined during the initial pitch:
1.  **Extraction:** How can information such as skills, education, and experience be effectively extracted from a CV (including scans)?
2.  **Scoring:** Is it possible to calculate an objective CV quality indicator?
3.  **Impact:** Which elements – skills, education, or description – have the greatest impact on the result?

---

## 03. Project Structure

```text
ATS-Insight/
│
├── predict_my_cv.py         # MAIN APP: Predicts score for your PDF
├── nlp.py                   # NLP Logic (Cleaning, Regex, Heuristics)
├── requirements.txt
├── .env.example             # Config for Tesseract/Poppler paths
├── README.md
│
├── database/                # Processed Data (JSONL)
│
├── input_data/              # Data Source
│   ├── raw/                 # Categorized PDFs (Kaggle, Synthetic, etc.)
|   ├── resumes_dataset.jsonl # More data for better model   
│
├── models/
│   └── best_model_balanced.pkl
│
├── analysis_results/        # Visualization
│
└── dev_tools/               # Backend Tools
    ├── read_cv.py           # Extraction Layer (PDF/OCR -> JSONL)
    ├── chose_model.py       # ML Layer (GridSearch + Training)
    └── descriptive_analysis.py # Visualization Layer

```

---


## 04. The Extraction Layer (ETL)

The first challenge in parsing resumes is handling the variety of file formats. A standard Python string search fails on image-based PDFs. I implemented a **Hybrid Two-Layer Extraction Engine** in `read_cv.py`.

### A. Digital Extraction

The system first attempts to read the file using **PyMuPDF (Fitz)**. This is fast and accurate for digitally created PDFs (e.g., exported from Word/LaTeX).

### B. Fallback OCR (Optical Character Recognition)

If the digital extraction results in an empty string or insufficient text (< 50 characters), the system flags the file as a "Scan". It then triggers a secondary pipeline:

1. **PDF-to-Image:** Converts pages to high-res images using `poppler`.
2. **Tesseract Engine:** Passes images through `pytesseract` to reconstruct the text layer.

### C. Data Augmentation & Labeling

Since real-world "Fit Scores" are subjective, I created a synthetic labeling logic.

* **The Matrix:** A scoring dictionary maps source categories to target roles (e.g., A "Java Developer" CV gets a 5.0 score for a Java role, but a 1.0 score for a "HR" role).
* **Augmentation:** Each raw resume is multiplied into 11 training records (one for each supported role), creating a dataset that teaches the model both **positive** (good fit) and **negative** (bad fit) correlations.


## 05. NLP & Feature Engineering

Raw text is noisy. The `nlp.py` module transforms unstructured strings into structured features usable by the model.

### 1. Cleaning & normalization

* **Artifact Removal:** Uses `ftfy` to fix encoding errors (mojibake) common in PDF parsing.
* **PII Stripping:** Regex removes emails and phone numbers to ensure the model learns skills, not personal identities.
* **Lemmatization:** Uses `spacy` to reduce words to their root forms (e.g., "managing" -> "manage").

### 2. Section Parsing

I implemented a heuristic parser that splits the CV into logical blocks (`SUMMARY`, `EXPERIENCE`, `EDUCATION`) using a combination of Regex headers and spacing analysis.

### 3. Entity Extraction

* **Skill Matching:** A `Spacy PhraseMatcher` compares the text against a database of ~100 IT skills (Python, Docker, AWS).
* **Education Parsing:** A stream-based parser extracts degrees, years, and institutions. It normalizes degrees (e.g., "B.S.", "Bachelor of Science" -> "Bachelor") to apply ordinal ranking.

### 4. Vectorization (TF-IDF)

Instead of simple one-hot encoding, I used **TF-IDF (Term Frequency-Inverse Document Frequency)**. This statistical measure evaluates how relevant a word is to a document in a collection. It helps the model distinguish that "Python" is a high-value keyword for a Data Scientist, whereas common words like "work" are less significant.


## 06. Machine Learning Model

The heart of the system is a supervised regression model trained to predict the continuous "Fit Score".

### The Pipeline (`chose_model.py`)

I used `scikit-learn` Pipelines to prevent data leakage during training.

1. **Preprocessing:**
* **Categorical:** `OneHotEncoder` for the Target Job Role.
* **Text:** `TfidfVectorizer` (Max 500 features for Summary, 300 for Skills).
* **Numerical:** `StandardScaler` for Education Rank (0=None, 3=Bachelor, 5=PhD,...).


2. **Imbalance Handling:** The dataset naturally contains more "bad fits" than "good fits". I implemented **Undersampling**, reducing the majority class to maintain a healthy 1:1.5 ratio.
3. **Regressor:** **`HistGradientBoostingRegressor`**.
* *Why?* It is an implementation of Gradient Boosting Trees (similar to LightGBM). It is significantly faster than standard Gradient Boosting on larger datasets and natively handles missing values (NaNs).

### Evaluation

The model was tuned using `GridSearchCV` (3-fold cross-validation) optimizing for **R2 Score**.

* **Result:** The model successfully generalizes, achieving positive R2 scores and a **Mean Absolute Error (MAE) of ~1.0**.
* **Interpretation:** While the model is not a perfect oracle, it serves as a highly effective **suitability filter**. An error margin of 1 point means the model might fluctuate between a "Good" (4.0) and "Average" (3.0) rating, but it **reliably distinguishes** between a valid candidate and a complete mismatch (e.g., an HR resume applied to a Java Developer role). The results are logically consistent: missing key hard skills drastically lowers the score, proving that the model has correctly learned the underlying semantic patterns of the job market.
* **Limitations:** It is important to note that the current performance is heavily constrained by the **dataset quality**. The training data relies partly on synthetic labeling logic (heuristic matrix) rather than human-verified ground truth. Acquiring a larger, real-world dataset with manual HR annotations would likely **significantly improve** the model's precision and reduce the error rate. However, such a dataset is realy hard to get.


## 07. Visualization & Results

The `descriptive_analysis.py` script generates insights answering Research Question #3 (Impact):

* **Feature Importance:** Shows which keywords and skills drive the score up (e.g., "Python" for Data Scientist).
* **Education Heatmap:** Visualizes the correlation between degrees and specific roles.

---

## How to Run

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/ATS-Insight.git

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Configure Environment:**
Create a `.env` file in the root directory and add your local paths to OCR tools:
```ini
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\Program Files\poppler\Library\bin

```


### Usage (For End Users)
**The model is already pre-trained and included in the repository.** You do not need to run any training scripts.

* **Predict Score:** `python predict_my_cv.py`
    * Simply run the command and provide the **absolute path to your PDF file** when prompted. The app will output your Fit Score (1-5) for each job.

### For Developers (Reproduce Analysis)
Only run these if you want to rebuild the dataset or retrain the model from scratch:

* **Database creation:** `python read_cv.py` (Transforms scanned CVs into a JSONL)
* **NLP:** `python nlp.py` (For language processing)
* **Train Model:** `python dev_tools/chose_model.py` (Runs GridSearch and saves new .pkl)
---

## Authors

* **Dominika Piechota**
