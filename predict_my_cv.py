import sys
import json
import joblib
import numpy as np
import fitz  # PyMuPDF - wymagane: pip install pymupdf
from pathlib import Path

# --- BLOK BEZPIECZEŃSTWA: SPRAWDZANIE WERSJI ---
# Skoro nie chcesz zmieniać środowiska ręcznie, ten kod przynajmniej
# powie Ci jasno, dlaczego nie działa, zamiast wyrzucać dziwne błędy.
try:
    import pandas as pd
except ImportError:
    print("❌ BŁĄD: Brak biblioteki pandas.")
    sys.exit(1)
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print("\n" + "!"*60)
        print("❌ BŁĄD KRYTYCZNY ŚRODOWISKA (NUMPY VERSION CONFLICT)")
        print("!"*60)
        print("Twój Python ma zainstalowany NumPy 2.x, ale Pandas/Sklearn wymagają 1.x.")
        print("Tego NIE DA się obejść w kodzie. Musisz wpisać w terminalu:")
        print('   pip install "numpy<2.0"')
        print("!"*60 + "\n")
        sys.exit(1)
    raise e

# --- IMPORTY Z PLIKU NLP.PY ---
try:
    # Zakładamy, że plik nlp.py leży w tym samym folderze
    from nlp import clean_and_repair_text, nlp_clean_text, parse_cv_sections, structure_education_stream
except ImportError as e:
    print("❌ BŁĄD: Nie znaleziono pliku 'nlp.py' w tym folderze.")
    print(f"Szczegóły: {e}")
    sys.exit(1)

# ==============================================================================
# KONFIGURACJA MODELU
# ==============================================================================

MODEL_FILE = Path("./models/best_model_balanced.pkl")

# Musi być zdefiniowane dla pickle
def to_dense(X): return X.toarray()

DEGREE_RANKING = {
    "UNKNOWN": 0, "HIGH SCHOOL": 1, "DIPLOMA": 1, "ASSOCIATE": 2,
    "BACHELOR": 3, "BACHELOR'S": 3, "BS": 3, "B.S": 3, "BA": 3, "BSC": 3, "ENGINEER": 3,
    "MASTER": 4, "MASTER'S": 4, "MS": 4, "M.S": 4, "MSC": 4, "MA": 4, "MBA": 5,
    "PHD": 5, "PH.D": 5, "DOCTORATE": 5
}

ALL_POSSIBLE_ROLES = [
    "Business Analyst", "Data Scientist", "DevOps Engineer", 
    "ETL Developer", "Java Developer", "Python Developer", 
    "React Developer", "SQL Developer", "Web Developer", 
    "SAP Developer", "Tester"
]

# ==============================================================================
# FUNKCJE POMOCNICZE
# ==============================================================================

def extract_text_simple(pdf_path):
    """
    Szybkie czytanie PDF. Zakłada, że plik jest cyfrowy (nie skan).
    Nie używa OCR.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"⚠️  Błąd otwierania PDF: {e}")
        return ""
    return text

def get_max_degree_rank(structured_edu_list):
    max_rank = 0
    for edu in structured_edu_list:
        degree_txt = edu.get("degree", "")
        if not degree_txt: continue
        
        d_upper = degree_txt.upper()
        best = 0
        for k, v in DEGREE_RANKING.items():
            if k in d_upper and v > best: best = v
        
        if best > max_rank: max_rank = best
    return max_rank

def process_cv_for_model(file_path):
    # 1. Odczyt bez OCR
    print(f"📄 Czytanie pliku: {file_path.name}...")
    raw_text = extract_text_simple(file_path)
    
    if not raw_text or len(raw_text) < 10:
        print("⚠️  Plik wydaje się pusty lub jest zdjęciem (skanem).")
        print("    Ten skrypt nie obsługuje OCR. Użyj pliku tekstowego/PDF cyfrowego.")
        return None

    # 2. NLP z Twojego pliku nlp.py
    print("🧹 Analiza NLP...")
    clean_text = clean_and_repair_text(raw_text)
    parsed_data = parse_cv_sections(clean_text)
    
    # 3. Ekstrakcja cech pod model
    skills_list = parsed_data.get("Extracted_Skills_List", [])
    skills_str = " ".join(skills_list)
    
    raw_edu_text = parsed_data["Sections"].get("Education", "")
    structured_edu = structure_education_stream(raw_edu_text) 
    degree_rank = get_max_degree_rank(structured_edu)
    
    raw_summary = parsed_data["Sections"].get("Summary", "") or parsed_data["Sections"].get("Other", "")[:1000]
    summary_clean = nlp_clean_text(raw_summary)

    fields_list = [e.get("field_of_study") for e in structured_edu if e.get("field_of_study") != "Unknown"]
    fields_str = " ".join(fields_list)

    return {
        "summary": summary_clean,
        "skills_text": skills_str,
        "degree_rank": degree_rank,
        "fields_text": fields_str,
        "_display_skills": skills_list,
        "_display_edu": structured_edu
    }

# ==============================================================================
# GŁÓWNA LOGIKA
# ==============================================================================

def main(pdf_path):
    path = Path(pdf_path)
    if not path.exists():
        print("❌ Plik nie istnieje.")
        return

    # KROK 1: Przetwarzanie danych
    features = process_cv_for_model(path)
    if not features: return

    print("\n--- ZNALEZIONE DANE ---")
    print(f"🎓 Wykształcenie (Ranga {features['degree_rank']}):")
    for edu in features["_display_edu"]:
        print(f"   - {edu['degree']} ({edu['field_of_study']}) @ {edu['institution']}")
    print(f"🛠️  Umiejętności: {', '.join(features['_display_skills'][:10])}...")

    # KROK 2: Ładowanie modelu
    if not MODEL_FILE.exists():
        print(f"❌ Nie znaleziono modelu: {MODEL_FILE}")
        return
    
    print("\n🧠 Uruchamianie modelu...")
    try:
        global to_dense
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        return

    # KROK 3: Predykcja
    rows = []
    for role in ALL_POSSIBLE_ROLES:
        rows.append({
            "target_role": role,
            "summary": features["summary"],
            "skills_text": features["skills_text"],
            "degree_rank": features["degree_rank"],
            "fields_text": features["fields_text"]
        })
    
    df_predict = pd.DataFrame(rows)
    scores = model.predict(df_predict)

    # KROK 4: Wyniki
    results = pd.DataFrame({"Rola": ALL_POSSIBLE_ROLES, "Score": scores})
    results = results.sort_values(by="Score", ascending=False).reset_index(drop=True)

    print("\n" + "="*40)
    print(f"📊 DOPASOWANIE ZAWODOWE: {path.name}")
    print("="*40)
    
    for i, row in results.iterrows():
        sc = row['Score']
        bar = "█" * int(sc * 4)
        print(f"{i+1}. {row['Rola']:<20} | {sc:.2f}/5.0 | {bar}")

    print("="*40)
    print(f"🏆 Rekomendacja: {results.iloc[0]['Rola']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Użycie: python predict_my_cv.py plik.pdf")
        f = input("Podaj ścieżkę do pliku PDF: ").strip('"')
        if f: main(f)