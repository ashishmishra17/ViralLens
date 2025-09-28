# ViralLens Project

## Overview
ViralLens is a modular machine learning and NLP project for data cleaning, model training, PDF text extraction, semantic search, and API deployment. It demonstrates a full workflow from raw data to ML model and retrieval system using modern Python libraries.

## Project Structure
```
ViralLens/
├── RAMAYANA.pdf                # Source PDF for text extraction
├── RAMAYANA_text.txt           # Extracted text from RAMAYANA.pdf
├── placement.csv               # Raw placement dataset
├── placement_cleaned.csv       # Cleaned placement dataset
├── requirement.txt             # Python dependencies
├── src/
│   ├── data_cleaning.py        # Cleans placement.csv, encodes, splits data
│   ├── model_training.py       # Trains Random Forest, evaluates, tunes
│   ├── model_api.py            # FastAPI endpoint for model predictions
│   ├── pdf_extract.py          # Extracts text from RAMAYANA.pdf
│   ├── ramayana_retrieval.py   # Embedding, FAISS, semantic search
```

## Setup
1. **Clone the repository**
2. **Install dependencies:**
   ```sh
   pip install -r requirement.txt
   ```

## Usage
### 1. Data Cleaning
Run:
```sh
python src/data_cleaning.py
```
- Cleans `placement.csv`, encodes categorical variables, handles missing values, splits data, and saves `placement_cleaned.csv`.

### 2. Model Training & Evaluation
Run:
```sh
python src/model_training.py
```
- Trains a Random Forest model, reports accuracy and F1 score, tunes `n_estimators` hyperparameter.

### 3. FastAPI Model Serving
Run:
```sh
uvicorn src.model_api:app --reload
```
- Serves `/predict` endpoint for model inference.
- Example request:
  ```sh
  curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"data": {"feature1": value1, ...}}'
  ```

### 4. PDF Text Extraction
Run:
```sh
python src/pdf_extract.py
```
- Extracts text from `RAMAYANA.pdf` to `RAMAYANA_text.txt`.

### 5. Semantic Search & Retrieval
Run:
```sh
python src/ramayana_retrieval.py
```
- Generates embeddings for passages in `RAMAYANA_text.txt` using Hugging Face models.
- Stores embeddings in FAISS.
- Retrieves top 2 relevant passages for a user query.

## Requirements
- Python 3.8+
- See `requirement.txt` for all dependencies (pandas, scikit-learn, fastapi, uvicorn, PyPDF2, sentence-transformers, faiss-cpu, etc.)

## Customization
- Update feature names in API requests to match your dataset.
- Modify retrieval code for different chunking or models.

## License
MIT

## Author
ashishmishra17
