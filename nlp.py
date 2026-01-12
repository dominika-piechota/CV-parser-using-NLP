import json
import re
import spacy
import ftfy
from pathlib import Path
from spacy.matcher import PhraseMatcher
import sys

INPUT_FILE = Path("./database/dataset_made_of_raw_cv.jsonl")             
FILE_STRUCTURED = Path("./database/dataset_structured.jsonl")    
FILE_FEATURES = Path("./database/dataset_features.jsonl")     
FILE_MERGED = Path("./database/dataset_features_merged.jsonl")     
FINAL_FILE = Path("./database/resumes_dataset_CLEANED.jsonl") 
EXTERNAL_DB_FILE = Path("./inputdata/resumes_dataset.jsonl")

print("Loading NLP model", flush=True)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("ERROR: Spacy model 'en_core_web_sm' not found.")
    sys.exit(1)

# Possible skills to match in resumes
SKILL_DB = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "SQL", "HTML", "CSS", "R", "Go", "Ruby", "PHP", 
    "React", "Angular", "Vue", "Spring Boot", "Django", "Flask", "Pandas", "NumPy", "TensorFlow", "Spark", "Hadoop",
    "SWOT", "PESTLE", "Business Analysis", "Financial Modeling", "Risk Assessment", "Gap Analysis", "Cost-Benefit Analysis", "ROI",
    "Requirements Gathering", "Stakeholder Management", "Process Improvement", "Lean", "Six Sigma", "Kanban",
    "Excel", "Word", "PowerPoint", "Miro", "Jira", "Confluence", "Trello", "Asana", "Tableau", "Power BI", "Google Analytics", 
    "Salesforce", "SAP", "Oracle", "SharePoint", "Visio", "Figma", "Slack", "Zoom", "Teams",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "Linux", "Unix",
    "Agile", "Scrum", "Waterfall", "Project Management", "Decision-Making", "Communication"
]
# Constructing PhraseMatcher for skill extraction
# attr="LOWER" makes matching case-insensitive
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(text) for text in SKILL_DB] # change list of strings to list of Docs needed by matcher
matcher.add("SKILL_LIST", patterns)

# All degree patterns to search for
DEGREE_PATTERNS = [
    r"\bB\.?S\.?C?\b", r"\bB\.?A\.?\b", r"\bB\.?ENG\b", r"\bBACHELOR['S]*\b", 
    r"\bM\.?S\.?C?\b", r"\bM\.?A\.?\b", r"\bM\.?ENG\b", r"\bMASTER['S]*\b", r"\bMBA\b",
    r"\bPH\.?D\b", r"\bDOCTORATE\b", r"\bASSOCIATE\b", r"\bDIPLOMA\b", r"\bENGINEER\b", r"\bDEGREE\b"
]

# Keywords indicating educational institutions
UNI_KEYWORDS = ["UNIVERSITY", "COLLEGE", "SCHOOL", "INSTITUTE", "ACADEMY", "POLYTECHNIC", "FACULTY"]

# All possible fields of study
FIELDS_OF_STUDY_DB = [
    # IT, Data Science
    "Computer Science", "Information Technology", "Software Engineering", 
    "Computer Engineering", "Data Science", "Artificial Intelligence", 
    "Machine Learning", "Cyber Security", "Network Engineering", 
    "Information Systems", "Web Development", "Computer Applications",
    "Applied Computing", "Cloud Computing", "Big Data Analytics",
    "Human Computer Interaction", "Game Development", "Informatics",
    "Telecommunications", "Systems Engineering", "Robotics",
    
    # Business, Management
    "Business Administration", "Business Management", "Business Analytics",
    "Project Management", "International Business", "Marketing", 
    "Finance", "Accounting", "Economics", "Human Resources", 
    "Supply Chain Management", "Logistics", "Operations Management",
    "Entrepreneurship", "Commerce", "Management Information Systems",
    "Public Administration", "Banking","Investment Management", "Business"   

    # Engineering, Math
    "Electrical Engineering", "Mechanical Engineering", "Civil Engineering",
    "Industrial Engineering", "Electronic Engineering", "Mathematics",
    "Applied Mathematics", "Statistics", "Physics", "Mechatronics",
    "Biomedical Engineering", "Chemical Engineering",
    
    # Humanities, Social
    "Psychology", "Sociology", "Political Science", "English Literature",
    "History", "Philosophy", "Communications", "Journalism", "Linguistics",
    "International Relations", "Law", "Legal Studies", "Education",
    "Graphic Design", "Fine Arts", "Digital Media", "Multimedia"
]

# All jobs/roles considered in the system
ALL_ROLES = ["Business Analyst", "Data Scientist", "DevOps Engineer", "ETL Developer", 
             "Java Developer", "Python Developer", "React Developer", "SQL Developer", 
             "Web Developer", "SAP Developer", "Tester"]

# Scoring logic for roles, assigning scores based on source category and target role,
# If job is similar to target role, score is high
# If job role not listed, score is 0
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

# Mapping from various source categories to standardized target roles
# Because of different namings in CVs and data sources
CATEGORY_MAPPING = {
    "Business Analysis": "Business Analyst", "Data Science": "Data Scientist", "DevOps": "DevOps Engineer", 
    "ETL Developer": "ETL Developer", "Python Developer": "Python Developer", "Java Developer": "Java Developer",
    "React Developer": "React Developer", "SAP Developer": "SAP Developer", "SQL Developer": "SQL Developer", "Testing": "Tester"
}

# Degree bonus mapping, additional score points based on highest degree obtained
DEGREE_BONUS = {
    "PHD": 0.5,
    "DOCTORATE": 0.5,
    "MBA": 0.4,
    "MASTER": 0.3,
    "M.S.": 0.3,
    "M.A.": 0.3,
    "ENGINEER": 0.2,
    "BACHELOR": 0.0, # Standard
    "B.S.": 0.0,
    "ASSOCIATE": -0.5,
    "HIGH SCHOOL": -1.0,
    "UNKNOWN": -0.5 # no information about degree, maybe does not have one
}


# Cleaning and repairing text
def clean_and_repair_text(text):
    if not text: return ""
    text = ftfy.fix_text(text) # fix encoding issues
    text = re.sub(r'[^\x00-\x7F\n]+', ' ', text) # remove non-ASCII characters
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text) # fix word breaks at line endings
    text = re.sub(r'[ \t]+', ' ', text) # normalize spaces and tabs, replace with single space
    text = re.sub(r'\n\s*\n', '\n', text) # remove multiple newlines
    return text.strip()

# Prepare text for NLP processing: lowercase, remove stopwords/punctuation, lemmatize
def nlp_clean_text(text):
    if not text or len(text) < 5: return ""
    doc = nlp(text[:10000].lower()) # limit to first 10k chars for performance
    cleaned = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space] # lemmatize and filter, leave only meaningful words
    return " ".join(cleaned)

# Analyze degrees and return bonus score for highest degree found
def get_degree_bonus(education_list):
    max_bonus = -0.5
    
    if not education_list: return max_bonus

    for edu in education_list:
        degree_text = edu.get("degree")
        if not degree_text: continue
        
        deg_upper = degree_text.upper()
        
        for key, bonus in DEGREE_BONUS.items():
            if key in deg_upper: # use 'in' to match substrings, for example "MASTER OF SCIENCE" contains "MASTER"
                if bonus > max_bonus:
                    max_bonus = bonus
    
    return max_bonus

# Extract personal info: phone number, first name, last name
# Not mandatory, but useful for contact purposes
# Model does not use this info for predictions
def extract_personal_info(text):
    info = {"phone": None, "first_name": None, "last_name": None}
    
    phone_pattern = re.compile(r'[\+\(]?[0-9][0-9 .\-\(\)]{8,}[0-9]')
    phones = phone_pattern.findall(text)
    for p in phones:
        if len(re.findall(r'\d', p)) < 9: continue # too short
        if "20" in p and "-" in p and len(p) < 12: continue # probably a year range
        info["phone"] = p.strip()
        break

    doc_head = nlp(text[:500]) # analyze only the beginning of the CV for names
    forbidden_names = ["Resume", "Curriculum", "Vitae", "Java", "Python", "Developer", "Engineer", "Manager", "Street", "Lane", "Road", "Date", "Ph.D", "Master"]
    for ent in doc_head.ents:
        if ent.label_ == "PERSON":
            clean_name = ent.text.strip().replace("\n", " ")
            # Be sure it's a valid name (no digits, not email, at least 2 parts, not forbidden words)
            if (not any(char.isdigit() for char in clean_name) 
                and "@" not in clean_name 
                and len(clean_name.split()) >= 2
                and not any(bad in clean_name for bad in forbidden_names)):        
                parts = clean_name.split()
                info["first_name"] = parts[0]
                info["last_name"] = " ".join(parts[1:])
                break

    return info

# Parse CV into sections based on headers and extract skills
def parse_cv_sections(full_text):
    parsed_data = {"Sections": {"Other": ""}, "Extracted_Skills_List": []}
    
    # First, extract skills using PhraseMatcher
    doc = nlp(full_text[:5000])
    found_matches = matcher(doc)
    found_skills = set()
    for _, start, end in found_matches:
        found_skills.add(doc[start:end].text)
    parsed_data["Extracted_Skills_List"] = list(found_skills)

    # Now, split text into sections based on headers
    section_headers = ["SKILLS", "EDUCATION", "EXPERIENCE", "SUMMARY", "PROJECTS", "CERTIFICATIONS"]
    # Signals indicating education sections, even if not explicitly labeled
    edu_signals = ["UNIVERSITY", "COLLEGE", "INSTITUTE", "SCHOOL", "BACHELOR", "MASTER", "PHD", "DIPLOMA", "DEGREE"]
    
    # Build dynamic regex pattern
    # Search for lines that are all uppercase (section headers) or lines that start with known section titles
    pattern_upper = r'\b(' + '|'.join(section_headers) + r')\b'
    pattern_title = r'(?:\n|^)\s*(' + '|'.join([re.escape(w.title()) for w in section_headers + edu_signals]) + r')\b'
    combined_regex = f'{pattern_upper}|{pattern_title}'
    
    parts = re.split(combined_regex, full_text, flags=re.IGNORECASE)
    parsed_data["Sections"]["Other"] = parts[0].strip().replace("\n", " ")
    current_key = "Other"
    
    # Iterate over the parts and assign them to sections
    # Regex split returns: text, header, text, header...
    for part in parts[1:]:
        if not part: continue
        upper_part = part.upper().strip()
        # Determine if this part is a new section header
        new_key = None
        if "EDUC" in upper_part or any(s in upper_part for s in edu_signals): new_key = "Education"
        elif "SUMM" in upper_part or "ABOUT" in upper_part: new_key = "Summary"
        elif "EXPER" in upper_part or "WORK" in upper_part: new_key = "Experience"
        
        # If new section detected, switch current key
        if new_key and len(part.split()) < 5:
            current_key = new_key
            # If it's education-related, also check for edu signals in the next part
            if any(s in upper_part for s in edu_signals):
                if new_key not in parsed_data["Sections"]: parsed_data["Sections"][new_key] = ""
                parsed_data["Sections"][new_key] += " " + part.strip()
        else:
            # If it's not a header, append to current section
            if current_key not in parsed_data["Sections"]: parsed_data["Sections"][current_key] = ""
            parsed_data["Sections"][current_key] += " " + part.strip().replace("\n", " ")
    return parsed_data

# Search for entities related to education in text, sich as degrees, institutions, fields, years
def find_all_entities(text):
    entities = []
    text_processed = text[:10000]

    for m in re.finditer(r'\b((?:19|20)\d{2}(?:\s*[-–to]\s*(?:(?:19|20)\d{2}|Present|Current))?)\b', text_processed, re.IGNORECASE):
        entities.append({"type": "years", "val": m.group(1), "start": m.start(), "end": m.end()})

    JUST_DEGREE = ["DEGREE"]
    for pat in DEGREE_PATTERNS:
        for m in re.finditer(pat, text_processed, re.IGNORECASE):
            val = m.group(0).strip().upper()
            if val in JUST_DEGREE: continue
            entities.append({"type": "degree", "val": val, "start": m.start(), "end": m.end()})

    for field in FIELDS_OF_STUDY_DB:
        pattern = r'\b' + re.escape(field) + r'\b'
        for m in re.finditer(pattern, text_processed, re.IGNORECASE):
            entities.append({"type": "field", "val": field, "start": m.start(), "end": m.end()})

    doc = nlp(text_processed)
    for ent in doc.ents:
        if ent.label_ == "ORG" and any(k in ent.text.upper() for k in UNI_KEYWORDS):
            entities.append({"type": "institution", "val": ent.text, "start": ent.start_char, "end": ent.end_char})

    # Remove overlapping entities, keep only the first occurring
    entities.sort(key=lambda x: x['start'])

    cleaned_entities = []
    last_end = -1
    for ent in entities:
        if ent['start'] >= last_end: # if no overlap
            cleaned_entities.append(ent)
            last_end = ent['end']
    return cleaned_entities

# Convert flat list of entities into structured education records in JSON
# Using a stream-based approach: 
# grouping entities into records, starting new record when same type entity found again 
def structure_education_stream(full_text):
    if not full_text: return []
    found_entities = find_all_entities(full_text)
    records = []
    current_record = {"degree": None, "institution": None, "years": None, "field": None}
    
    for ent in found_entities:
        ent_type = ent['type']
        ent_val = ent['val']
        # Heuristic: If we already have this type in current record, start a new record
        if current_record.get(ent_type) is not None:
            if any(current_record.values()): records.append(current_record)
            current_record = {"degree": None, "institution": None, "years": None, "field": None}
        current_record[ent_type] = ent_val
    # Append last record if not empty
    if any(current_record.values()): records.append(current_record)
    
    # Filter out records with no useful info (no degree and no institution)
    final_output = []
    for r in records:
        if r["degree"] or r["institution"]:
            final_output.append({
                "degree": r["degree"],
                "institution": r["institution"],
                "years": r["years"],
                "field_of_study": r["field"] if r["field"] else "Unknown"
            })
    return final_output

# Merge education entries with institution details
def merge_education_fields(edu_list):
    if not edu_list: return []
    merged = []
    for entry in edu_list:
        inst = entry.get("institution")
        merged.append({
            "degree": entry.get("degree"),
            "years": entry.get("years"),
            "institution_details": inst if inst else "", 
            "field_of_study": entry.get("field_of_study")
        })
    return merged

# Simple education processing for external DB entries with unstructured education text
def process_education_simple(text):
    if not text or len(text) < 3: return []
    years = re.search(r'\b20\d{2}\b', text)
    field_found = "Unknown"
    for f in FIELDS_OF_STUDY_DB:
        if f.lower() in text.lower():
            field_found = f
            break
    return [{
        "degree": None,
        "years": years.group(0) if years else None,
        "institution_details": text.strip(),
        "field_of_study": field_found
    }]



# ------ ETL Pipeline Steps - logical steps to process raw CV data into structured dataset

# Raw -> Structured
# Clean text, parse sections, find personal information
def step_1_process_raw():
    print(f"\n[STEP 1] Parse and structure: {INPUT_FILE} -> {FILE_STRUCTURED}", flush=True)
    if not INPUT_FILE.exists(): return print(f"No file {INPUT_FILE}!")
    
    count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, open(FILE_STRUCTURED, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if not line.strip(): continue
            if i % 10 == 0:
                print(f"   -> Processing record number {i}...", end='\r', flush=True)
                
            try:
                rec = json.loads(line)
                raw = rec.get("cv_text", "")

                clean = clean_and_repair_text(raw)
                
                personal_info = extract_personal_info(clean)
                
                parsed = parse_cv_sections(clean)

                raw_edu = parsed["Sections"].get("Education", "")
                structured_edu = structure_education_stream(raw_edu)
                
                # Update record with parsed data
                rec["parsed_education"] = structured_edu 
                rec["extracted_skills"] = parsed["Extracted_Skills_List"]
                # If no summary, take first 500 chars of "Other" section
                rec["parsed_summary"] = parsed["Sections"].get("Summary", "") or parsed["Sections"].get("Other", "")[:500]
                
                rec["personal_info"] = personal_info 
                
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"\n [ERROR] Line {i}: {e}")
                continue
                
    print(f"\n[STEP 1] Finished. Number of records: {count}", flush=True)

# Structured -> Features + Scoring
# Format features for model input, calculate scoring based on logic (bonus for degrees)
def step_2_extract_features():
    print(f"\n[STEP 2] Format features and scoring: -> {FILE_FEATURES}", flush=True)
    if not FILE_STRUCTURED.exists(): return
    
    count = 0
    with open(FILE_STRUCTURED, 'r', encoding='utf-8') as f_in, open(FILE_FEATURES, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            rec = json.loads(line)
            
            personal = rec.get("personal_info", {})
            parsed_edu = rec.get("parsed_education", [])
            
            base_score = rec.get("score", 0)
            deg_bonus = get_degree_bonus(parsed_edu)
            final_score = base_score + deg_bonus
            final_score = max(1.0, min(5.0, final_score)) # Clamp 1-5
            final_score = round(final_score, 2)
    
            # Prepare features dictionary
            features = {
                "skills": rec.get("extracted_skills", []),
                "education": parsed_edu, 
                "summary_clean": nlp_clean_text(rec.get("parsed_summary", "")),
                "phone": personal.get("phone"),
                "first_name": personal.get("first_name"),
                "last_name": personal.get("last_name")
            }
            
            # Create new record with features and final score
            new_rec = {
                "cv_id": rec.get("cv_id"),
                "target_role": rec.get("target_role"),
                "score": final_score,
                "features": features
            }
            f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            count += 1
            if count % 50 == 0: print(f"   -> Formatting {count}...", end='\r')
    print("\n[STEP 2] Finished.")

# Merge external database entries into main dataset
def step_3_merge_external():
    print(f"\n[STEP 3] Merging: -> {FILE_MERGED}", flush=True)
    
    # First, copy existing features and merge education fields
    with open(FILE_FEATURES, 'r', encoding='utf-8') as f_in, open(FILE_MERGED, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            rec = json.loads(line)
            feats = rec.get("features", {})
            feats["education"] = merge_education_fields(feats.get("education", []))
            rec["features"] = feats
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
        # Add external DB entries if file exists
        if EXTERNAL_DB_FILE.exists():
            print("Adding external database", flush=True)
            with open(EXTERNAL_DB_FILE, 'r', encoding='utf-8') as f_ext:
                try:
                    data = json.load(f_ext)
                    if not isinstance(data, list): data = [data]
                except:
                    # Fallback for formant JSONL instead of JSON array
                    f_ext.seek(0)
                    data = [json.loads(l) for l in f_ext if l.strip()]

                ext_cnt = 0
                for item in data:
                    role = CATEGORY_MAPPING.get(item.get("Category", ""), item.get("Category", ""))
                    if role not in SCORING_LOGIC: continue
                    # Generate records FOR EACH possible target role with appropriate scoring
                    # This allows model to learn not only from positive examples (Java Dev for Java Dev role),
                    # but also negative (Java Dev for Data Scientist role), it also makes dataset larger
                    scores = SCORING_LOGIC[role]
                    for target in ALL_ROLES:
                        new_rec = {
                            "cv_id": f"EXT_{item.get('id', ext_cnt)}",
                            "target_role": target,
                            "score": scores.get(target, 0), # get score from correlation table
                            "features": {
                                "skills": str(item.get("Skills", "")).split(","),
                                "education": process_education_simple(str(item.get("Education", ""))),
                                "summary_clean": nlp_clean_text(str(item.get("Summary", ""))),
                                # For external data, we don't have to have personal info (cv_id is always provided)
                                "phone": None,
                                "first_name": None,
                                "last_name": None
                            }
                        }
                        f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    ext_cnt += 1
                print(f"{ext_cnt} records added from external DB.")

# Final cleaning: Merged -> Cleaned
# Delete not useful parts from summaries (emails or phones)
# Filter out records with too short summaries or no valid education entries
# Save final cleaned dataset to training the model
def step_4_final_clean():
    print(f"\n[STEP 4] Final cleaning: -> {FINAL_FILE}", flush=True)
    FINAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    kept, removed = 0, 0
    
    with open(FILE_MERGED, 'r', encoding='utf-8') as f_in, open(FINAL_FILE, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            rec = json.loads(line)
            feats = rec.get("features", {})
            
            # Clean summary: remove emails, phone numbers
            summ = feats.get("summary_clean", "")
            if summ:
                summ = re.sub(r'\S+@\S+', '', summ)
                summ = re.sub(r'[\+\(]?[0-9][0-9 .\-\(\)]{6,}[0-9]', '', summ)
                summ = re.sub(r'\s+', ' ', summ).strip()
            
            # Filter out too short summaries for quality
            if not summ or len(summ.split()) < 2:
                removed += 1
                continue
                
            feats["summary_clean"] = summ

            # Validate education entries
            valid_edu = []
            for edu in feats.get("education", []):
                if 'years' in edu: del edu['years']

                inst = re.sub(r'\d+', '', edu.get("institution_details", "")).strip()
                field = edu.get("field_of_study", "Unknown")
                
                if not field or field == "Unknown": continue 

                # Accept only entries with recognized institution keywords or reasonably long text
                if any(k in inst.upper() for k in UNI_KEYWORDS) or (inst and len(inst.split()) <= 20):
                    edu["institution_details"] = inst
                    valid_edu.append(edu)
            
            # If no valid education entries, skip record
            if not valid_edu:
                removed += 1
                continue
                
            feats["education"] = valid_edu
            rec["features"] = feats
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            
    print(f"[THE END] Saved: {kept}, Deleted: {removed}")


# Main ETL pipeline runner - execute all steps in order
def run_full_etl():
    print("=== ETL PIPELINE START===")
    step_1_process_raw()
    step_2_extract_features()
    step_3_merge_external()
    step_4_final_clean()
    print("=== ETL PIPELINE FINISHED ===")

if __name__ == "__main__":
    run_full_etl()