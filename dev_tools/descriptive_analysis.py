import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import math
from collections import Counter
from wordcloud import WordCloud
from pathlib import Path
plt.switch_backend('Agg')

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent

sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "database" / "resumes_dataset_CLEANED.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "analysis_results"

# For reliable plots, set a minimum score threshold, which filters out low-confidence entries
MIN_SCORE = 4.5

STOPWORDS = {
    'and', 'the', 'to', 'of', 'in', 'for', 'with', 'a', 'an', 'on', 'is', 'at', 
    'as', 'from', 'by', 'summary', 'experience', 'skill', 'skills', 'work', 
    'year', 'years', 'knowledge', 'strong', 'ability', 'proficient', 'using',
    'working', 'various', 'including', 'responsible', 'team', 'project', 'business'
}

DEGREE_MAPPING = {
    'PHD': 'PhD', 'DOCTORATE': 'PhD',
    'MASTER': 'Master', 'MBA': 'Master', 'MS': 'Master', 'MSC': 'Master', 'M.S': 'Master',
    'BACHELOR': 'Bachelor', 'BS': 'Bachelor', 'BSC': 'Bachelor', 'B.S': 'Bachelor', 'ENGINEER': 'Bachelor',
    'ASSOCIATE': 'Associate', 'DIPLOMA': 'Diploma', 'HIGH SCHOOL': 'High School'
}
DEGREE_ORDER = ['PhD', 'Master', 'Bachelor', 'Associate', 'Diploma', 'High School', 'Unknown']

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def normalize_degree(edu_list):
    if not isinstance(edu_list, list) or not edu_list:
        return "Unknown"
    found_degrees = set()
    for edu in edu_list:
        deg = str(edu.get('degree', '')).upper()
        if not deg: continue
        mapped = "Unknown"
        for key, val in DEGREE_MAPPING.items():
            if key in deg:
                mapped = val
                break
        found_degrees.add(mapped)
    for d in DEGREE_ORDER:
        if d in found_degrees:
            return d
    return "Unknown"

def get_text_keywords(text):
    if not isinstance(text, str): return []
    words = text.lower().replace('.', '').replace(',', '').split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def plot_class_distribution(df):
    print("Generating plot - class distribution...")
    plt.figure(figsize=(12, 8))
    order = df['target_role'].value_counts().index
    sns.countplot(y='target_role', data=df, order=order, palette='viridis', hue='target_role', legend=False)
    plt.title(f'Number of resumes (Score > {MIN_SCORE})', fontsize=16)
    plt.xlabel('Number of resumes')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_class_distribution.png")
    plt.close()

def plot_text_length(df):
    print("Generating plot - text length (no outliers)...")
    df['text_len'] = df['features'].apply(lambda x: len(x.get('summary_clean', '').split()) if isinstance(x, dict) else 0)
    
    plt.figure(figsize=(14, 8))
    order = df['target_role'].value_counts().index
    
    sns.boxplot(x='text_len', y='target_role', data=df, order=order, palette='Set2', hue='target_role', legend=False, showfliers=False)
    
    plt.title('Summary length (number of words) - Typical values', fontsize=16)
    plt.xlabel('Number of words')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_text_length.png")
    plt.close()

def plot_education_heatmap(df):
    print("Generating heatmap - education levels...")
    df['highest_degree'] = df['features'].apply(lambda x: normalize_degree(x.get('education', [])) if isinstance(x, dict) else "Unknown")
    
    edu_counts = pd.crosstab(df['target_role'], df['highest_degree'])
    cols = [c for c in DEGREE_ORDER if c in edu_counts.columns]
    edu_counts = edu_counts[cols]
    
    edu_pct = edu_counts.div(edu_counts.sum(axis=1), axis=0) * 100 # Convert to percentages
    
    sort_key = edu_pct.get('Master', 0) + edu_pct.get('PhD', 0)
    if isinstance(sort_key, pd.Series):
        edu_pct = edu_pct.loc[sort_key.sort_values(ascending=False).index]

    plt.figure(figsize=(12, 10))
    sns.heatmap(edu_pct, annot=True, fmt='.1f', cmap='Blues', linewidths=.5, cbar_kws={'label': 'Procent kandydatów (%)'})
    plt.title('Education level by role (Percentage distribution)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_education_percent.png")
    plt.close()

def plot_top_3_per_role(df, extract_func, title_prefix, filename_suffix, color_palette='magma'):
    print(f"Generating plot: {title_prefix}...")
    
    roles = df['target_role'].unique()
    num_roles = len(roles)
    cols = 3
    rows = math.ceil(num_roles / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for i, role in enumerate(roles):
        ax = axes[i]
        subset = df[df['target_role'] == role]
        
        all_items = []
        for feat in subset['features']:
            if isinstance(feat, dict):
                items = extract_func(feat)
                all_items.extend(items)
        
        if not all_items:
            ax.text(0.5, 0.5, "Brak danych", ha='center', va='center')
            ax.set_title(role)
            continue

        # Top 3
        counter = Counter(all_items)
        top_items = counter.most_common(3)
        
        if not top_items:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
        else:
            labels, values = zip(*top_items)
            labels = [l.title() for l in labels]
            sns.barplot(x=list(values), y=list(labels), ax=ax, palette=color_palette, hue=list(labels), legend=False)
        
        ax.set_title(role, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(f'{title_prefix} (Top 3 per Role)', fontsize=20, y=1.005)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename_suffix}")
    plt.close()

def extract_skills(feat):
    if 'skills' in feat and isinstance(feat['skills'], list):
        return [s.lower().strip() for s in feat['skills']]
    return []

def extract_summary_keywords(feat):
    return get_text_keywords(feat.get('summary_clean', ''))

def plot_wordcloud(df):
    print("Generating wordcloud - most frequent words...")
    text_corpus = ""
    for feat in df['features']:
        if isinstance(feat, dict):
            text_corpus += feat.get('summary_clean', '') + " "
    
    if len(text_corpus) < 100: return

    wc = WordCloud(width=1600, height=800, background_color='white', max_words=150, colormap='viridis', stopwords=STOPWORDS).generate(text_corpus)
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most frequent words in the entire dataset', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_wordcloud.png")
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*40)
    print("DESCRIPTIVE ANALYSIS (v2.0)")
    print("="*40)
    ensure_output_dir()
    
    if not os.path.exists(DATA_PATH):
        print(f"No file found: {DATA_PATH}")
        return

    try:
        if str(DATA_PATH).endswith('.jsonl'):
            df = pd.read_json(DATA_PATH, lines=True)
        else:
            df = pd.read_csv(DATA_PATH)
        print(f"✅ Loaded: {len(df)} rows.")
        
        if 'score' in df.columns:
            df = df[df['score'] > MIN_SCORE]
            print(f"After filtering (> {MIN_SCORE}): {len(df)} rows.")
        
        if df.empty:
            print("No data.")
            return

        # 3. Generowanie wykresów
        plot_class_distribution(df)
        plot_text_length(df)
        plot_education_heatmap(df)
        
        # Nowe wykresy "Top 3 per Rola"
        plot_top_3_per_role(df, extract_skills, "Most frequent skills", "4_top3_skills_per_role.png", "viridis")
        plot_top_3_per_role(df, extract_summary_keywords, "Most frequent keywords (Summary)", "5_top3_keywords_per_role.png", "mako")

        plot_wordcloud(df)
        
        print("\n" + "="*40)
        print(f"Ready! Check: ./{OUTPUT_DIR}")
        print("="*40)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()