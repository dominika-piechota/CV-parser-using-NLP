import os
import json
import fitz  # pymupdf
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import os
import sys
from dotenv import load_dotenv  # pip install python-dotenv

# Load environment variables from .env file
load_dotenv()

# Tesseract configuration, file .env should contain TESSERACT_PATH and POPPLER_PATH if needed
# if there is no such variable, it is assumed that Tesseract and Poppler are in system PATH
path_tesseract = os.getenv('TESSERACT_PATH')
if path_tesseract:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = path_tesseract

# Poppler configuration - adds Poppler bin folder to system PATH for the runtime
path_poppler = os.getenv('POPPLER_PATH')
if path_poppler:
    os.environ["PATH"] += os.pathsep + path_poppler



# Ustalanie ścieżek względem lokalizacji tego pliku (dev_tools/ten_plik.py)
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent  # Wyjście dwa razy w górę: plik -> dev_tools -> PROJEKT

# Wczytanie .env z głównego katalogu (a nie z dev_tools)
load_dotenv(PROJECT_ROOT / ".env")

# Definicja ścieżek przy użyciu PROJECT_ROOT
# Dzięki temu "./" nie jest już potrzebne, bo mamy pełną ścieżkę absolutną
RAW_DATA_DIR = PROJECT_ROOT / "input_data" / "raw"
OUTPUT_FILE = PROJECT_ROOT / "database" / "dataset_made_of_raw_cv.jsonl"


ALL_ROLES = [
    "Business Analyst", "Data Scientist", "DevOps Engineer", "ETL Developer",
    "Java Developer", "Python Developer", "React Developer", "SQL Developer",
    "Web Developer", "SAP Developer", "Tester"
]

# Assigning scores based on source category and target role
SCORING_LOGIC = {
    "Business Analyst": {"Business Analyst": 5, "SQL Developer": 2, "SAP Developer": 2, "Data Scientist": 1},
    "Data Scientist": {"Data Scientist": 5, "Python Developer": 4, "SQL Developer": 3, "ETL Developer": 3, "Business Analyst": 2},
    "DevOps Engineer": {"DevOps Engineer": 5, "Python Developer": 3, "Java Developer": 2, "Web Developer": 1},
    "ETL Developer": {"ETL Developer": 5, "SQL Developer": 4, "Data Scientist": 3, "Python Developer": 2},
    "Java Developer": {"Java Developer": 5, "Web Developer": 3, "SQL Developer": 2, "DevOps Engineer": 1, "Python Developer": 1},
    "Python Developer": {"Python Developer": 5, "Data Scientist": 3, "DevOps Engineer": 3, "Web Developer": 2, "ETL Developer": 2},
    "React Developer": {"React Developer": 5, "Web Developer": 4, "Java Developer": 1, "Python Developer": 1},
    "SQL Developer": {"SQL Developer": 5, "ETL Developer": 4, "Data Scientist": 3, "Business Analyst": 2, "Web Developer": 1},
    "Web Developer": {"Web Developer": 5, "React Developer": 4, "Java Developer": 3, "Python Developer": 3, "DevOps Engineer": 1},
    "SAP Developer": {"SAP Developer": 5, "Business Analyst": 2, "Java Developer": 1},
    "Tester": {"Tester": 5, "Java Developer": 2, "Python Developer": 2, "Web Developer": 1, "Business Analyst": 1}
}

# TEXT EXTRACTION FUNCTION

# try to extract text from PDF using digital method first, then OCR if needed
def extract_content(pdf_path):
    text = ""
    method = "digital"

    # try digital extraction first, for normal PDFs
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"Warning: Fitz failed on {pdf_path.name}: {e}")

    # if text is too short (less than 50 characters), assume it's a scanned PDF (photo) and use OCR
    if len(text.strip()) < 50:
        print(f"   -> Wykryto skan (len={len(text.strip())}). Uruchamiam OCR...")
        method = "ocr"
        try:
            # convert PDFs to images
            images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
            
            ocr_text_list = []
            for i, image in enumerate(images):
                # read text from each image using pytesseract
                page_text = pytesseract.image_to_string(image, lang='eng') 
                ocr_text_list.append(page_text)
            
            text = "\n".join(ocr_text_list)
            
        except Exception as e:
            print(f"   -> OCR Error: {e}")
            return "", "error"

    # cleaning text by removing excessive whitespace
    clean_text = " ".join(text.split())
    return clean_text, method


# DATASET BUILDING FUNCTION

def build_dataset():
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)

    print(f"Start of processing, result will be available in: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # Iter through folders (each folder = one job)
        for folder in RAW_DATA_DIR.iterdir():
            if not folder.is_dir(): continue
            
            source_category = folder.name
            
            if source_category not in SCORING_LOGIC:
                continue

            print(f"Current folder: {source_category}...")
            
            files = list(folder.glob("*.pdf"))
            for pdf_file in files:
                
                # extract text using previously defined function
                cv_text, method_used = extract_content(pdf_file)
                
                # if extraction failed, skip this file
                if not cv_text:
                    print(f"   -> Empty file or error: {pdf_file.name}")
                    continue

                # DATA AUGMENTATION - make eleven records per CV, one for each target role
                relevant_scores = SCORING_LOGIC[source_category]
                
                for target_role in ALL_ROLES:
                    score = relevant_scores.get(target_role, 0)
                    
                    record = {
                        "cv_id": pdf_file.stem,
                        "target_role": target_role,
                        "score": score,
                        "cv_text": cv_text,
                        "metadata": {
                            "filename": pdf_file.name,
                            "original_category": source_category,
                            "extraction_method": method_used
                        }
                    }
                    
                    # save record as JSON line
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nSuccessfully completed!")

if __name__ == "__main__":
    build_dataset()
