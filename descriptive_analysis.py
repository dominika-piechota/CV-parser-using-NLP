import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import re
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud  # pip install wordcloud

# Ustawienie backendu, żeby nie otwierać okienek
plt.switch_backend('Agg')

# ==============================================================================
# KONFIGURACJA
# ==============================================================================

DATA_PATH = r"./database/resumes_dataset_CLEANED.jsonl" 
OUTPUT_DIR = "analysis_results"
MIN_SCORE = 4.0  # Analizujemy tylko CV z wynikiem powyżej tego progu

# Mapowanie stopni naukowych (uproszczone)
DEGREE_MAPPING = {
    'PHD': 'PhD', 'DOCTORATE': 'PhD',
    'MASTER': 'Master', 'MBA': 'Master', 'MS': 'Master', 'MSC': 'Master', 'M.S': 'Master',
    'BACHELOR': 'Bachelor', 'BS': 'Bachelor', 'BSC': 'Bachelor', 'B.S': 'Bachelor', 'ENGINEER': 'Bachelor',
    'ASSOCIATE': 'Associate', 'DIPLOMA': 'Diploma', 'HIGH SCHOOL': 'High School'
}

DEGREE_ORDER = ['PhD', 'Master', 'Bachelor', 'Associate', 'Diploma', 'High School', 'Unknown']

# ==============================================================================
# FUNKCJE POMOCNICZE
# ==============================================================================

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def normalize_degree(edu_list):
    """Znajduje najwyższy stopień naukowy w liście edukacji."""
    if not isinstance(edu_list, list) or not edu_list:
        return "Unknown"
    
    found_degrees = set()
    for edu in edu_list:
        deg = str(edu.get('degree', '')).upper()
        if not deg: continue
        
        # Proste dopasowanie
        mapped = "Unknown"
        for key, val in DEGREE_MAPPING.items():
            if key in deg:
                mapped = val
                break # Znaleziono mapowanie
        found_degrees.add(mapped)
    
    # Zwracamy najwyższy (według listy DEGREE_ORDER)
    for d in DEGREE_ORDER:
        if d in found_degrees:
            return d
    return "Unknown"

# ==============================================================================
# FUNKCJE RYSOWANIA
# ==============================================================================

def plot_class_distribution(df):
    print("📊 1. Generowanie wykresu: Rozkład ról...")
    plt.figure(figsize=(12, 8))
    order = df['target_role'].value_counts().index
    
    sns.countplot(y='target_role', data=df, order=order, palette='viridis', hue='target_role', legend=False)
    plt.title(f'Liczba kandydatów na stanowisko (Score > {MIN_SCORE})', fontsize=16)
    plt.xlabel('Liczba CV')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_class_distribution.png")
    plt.close()

def plot_text_length(df):
    print("📊 2. Generowanie wykresu: Długość CV...")
    # Wyciągamy tekst z features
    df['text_len'] = df['features'].apply(lambda x: len(x.get('summary_clean', '').split()) if isinstance(x, dict) else 0)
    
    plt.figure(figsize=(14, 8))
    order = df['target_role'].value_counts().index
    sns.boxplot(x='text_len', y='target_role', data=df, order=order, palette='Set2', hue='target_role', legend=False)
    
    plt.title('Długość podsumowania CV (liczba słów)', fontsize=16)
    plt.xlim(0, 600)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_text_length.png")
    plt.close()

def plot_top_skills(df):
    print("📊 3. Generowanie wykresu: Najpopularniejsze umiejętności...")
    
    all_skills = []
    # Iterujemy po features i wyciągamy listy skills
    for feat in df['features']:
        if isinstance(feat, dict) and 'skills' in feat and isinstance(feat['skills'], list):
            # Normalizacja: małe litery, usuwanie duplikatów w ramach jednego CV
            skills_in_cv = set(s.lower().strip() for s in feat['skills'])
            all_skills.extend(skills_in_cv)
            
    if not all_skills:
        print("⚠️ Brak umiejętności do analizy.")
        return

    # Liczenie top 20
    counter = Counter(all_skills)
    top_skills = counter.most_common(20)
    skills, counts = zip(*top_skills)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(counts), y=[s.title() for s in skills], palette='magma', hue=[s.title() for s in skills], legend=False)
    plt.title('Top 20 najczęściej występujących umiejętności (cały zbiór)', fontsize=16)
    plt.xlabel('Liczba wystąpień')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_top_skills.png")
    plt.close()

def plot_education_heatmap(df):
    print("📊 4. Generowanie wykresu: Poziom wykształcenia...")
    
    # 1. Wyciągamy najwyższy stopień dla każdego wiersza
    df['highest_degree'] = df['features'].apply(lambda x: normalize_degree(x.get('education', [])) if isinstance(x, dict) else "Unknown")
    
    # 2. Tabela przestawna (Crosstab)
    edu_counts = pd.crosstab(df['target_role'], df['highest_degree'])
    
    # Sortowanie kolumn wg hierarchii
    cols = [c for c in DEGREE_ORDER if c in edu_counts.columns]
    edu_counts = edu_counts[cols]
    
    # Sortowanie wierszy wg liczebności
    edu_counts['total'] = edu_counts.sum(axis=1)
    edu_counts = edu_counts.sort_values('total', ascending=False).drop(columns='total')
    
    # 3. Rysowanie heatmapy
    plt.figure(figsize=(12, 10))
    sns.heatmap(edu_counts, annot=True, fmt='d', cmap='Blues', linewidths=.5)
    plt.title('Poziom wykształcenia wg roli (Liczba kandydatów)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_education_matrix.png")
    plt.close()

def plot_wordcloud(df):
    print("📊 5. Generowanie chmury słów...")
    
    # Łączymy wszystkie summary w jeden wielki tekst
    text_corpus = ""
    for feat in df['features']:
        if isinstance(feat, dict):
            text_corpus += feat.get('summary_clean', '') + " "
            
    if len(text_corpus) < 100:
        print("⚠️ Za mało tekstu na chmurę słów.")
        return

    wc = WordCloud(width=1600, height=800, background_color='white', max_words=150, colormap='viridis').generate(text_corpus)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Chmura słów (Najczęstsze słowa w podsumowaniach)', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_wordcloud.png")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*40)
    print("🚀 ANALIZA OPISOWA DANYCH (DESCRIPTIVE ANALYSIS)")
    print("="*40)
    
    ensure_output_dir()
    
    # 1. Wczytanie danych
    if not os.path.exists(DATA_PATH):
        print(f"❌ Nie znaleziono pliku: {DATA_PATH}")
        return

    try:
        if str(DATA_PATH).endswith('.jsonl'):
            df = pd.read_json(DATA_PATH, lines=True)
        else:
            df = pd.read_csv(DATA_PATH)
            
        print(f"✅ Wczytano surowe dane: {len(df)} wierszy.")
        
        # 2. Filtrowanie
        if 'score' in df.columns:
            df = df[df['score'] > MIN_SCORE]
            print(f"📉 Po filtrowaniu (Score > {MIN_SCORE}): {len(df)} wierszy.")
        
        if df.empty:
            print("❌ Brak danych po filtrowaniu!")
            return

        # 3. Uruchomienie wykresów
        plot_class_distribution(df)
        plot_text_length(df)
        plot_top_skills(df)
        plot_education_heatmap(df)
        plot_wordcloud(df)
        
        print("\n" + "="*40)
        print(f"✅ ZAKOŃCZONO. Wykresy zapisano w folderze: ./{OUTPUT_DIR}")
        print("="*40)

    except Exception as e:
        print(f"❌ Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()