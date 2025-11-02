# 🏥 NUCC Specialty Standardizer (Hybrid Matching Engine)

### Overview
This repository contains an **offline NUCC provider specialty standardization tool** that maps **raw, unstandardized medical specialties** (e.g., “Cardio”, “ENT”, “OBGYN”) to official **NUCC Taxonomy Codes**.

It uses a **hybrid multi-algorithm pipeline** combining:
- String normalization & phonetic matching  
- Character n-gram TF-IDF scoring  
- Optional **Sentence-Transformer embeddings**  
- Trigram Jaccard fallback similarity  
- Direct **synonym and alias expansion**  

The system is designed for **robust data standardization** in healthcare claims, provider rosters, and insurance network files.

---

## 🚀 Features

- 🔡 **Text Normalization & Tokenization**  
  Cleans and standardizes text across casing, punctuation, and spacing.

- 🔊 **Phonetic Matching** (Soundex-based)  
  Handles abbreviations and pronunciation variants (e.g., *ENT → Otolaryngology*).

- 🤖 **Hybrid Similarity Engine**  
  Combines **TF-IDF**, **embedding cosine similarity**, and **trigram Jaccard** for fallback scoring.

- 🧩 **Synonym & Alias Dictionary**  
  Optional CSV for custom mappings (`alias`, `normalized`, `code`).

- 🧮 **Confidence-Based Decisioning**  
  Outputs confidence scores and classifies low-confidence entries as `"JUNK"`.

- 📊 **Batch CSV Processing**  
  Processes thousands of records with detailed explanation per specialty.

---

## 📦 Installation

Clone and install requirements:

```bash
git clone https://github.com/<your-username>/nucc-standardizer.git
cd nucc-standardizer
pip install -r requirements.txt
```

### Requirements
```text
pandas
numpy
scikit-learn
sentence-transformers   # optional, for embeddings
```

---

## 🧠 How It Works

1. **Build NUCC Candidates:**  
   - Reads taxonomy master file.  
   - Constructs normalized text per `code` using classification, specialization, or display name.

2. **Index Construction:**  
   - Builds TF-IDF matrix for character n-grams (3–5).  
   - Optionally loads a sentence-transformer (e.g. `all-MiniLM-L6-v2`) for embedding-based similarity.  
   - Caches phonetic encodings for faster lookup.

3. **Synonym Expansion (Optional):**  
   Loads custom `synonyms.csv` for known abbreviations and direct alias-to-code mapping.

4. **Mapping Phase:**  
   Each input specialty phrase goes through:
   - Synonym or direct code match  
   - Exact normalized surface match  
   - Hybrid similarity scoring (TF-IDF + embedding + trigram Jaccard)  
   - Threshold-based acceptance (default 0.6)

5. **Aggregation:**  
   Multi-part specialties (e.g. “Cardio / Neuro”) are split and merged with average confidence.

---

## ⚙️ Usage

```bash
python standardize_2.py \
  --nucc nucc_taxonomy_master.csv \
  --input input_specialties.csv \
  --out standardized_output.csv \
  --syn synonyms.csv \
  --threshold 0.6 \
  --max-candidates 6
```

### Arguments
| Flag | Description | Default |
|------|--------------|----------|
| `--nucc` | Path to NUCC taxonomy CSV | *required* |
| `--input` | Input CSV with `raw_specialty` column | *required* |
| `--out` | Output CSV path | *required* |
| `--syn` | Optional synonym dictionary CSV | None |
| `--threshold` | Confidence threshold | 0.6 |
| `--max-candidates` | Max number of top codes per phrase | 6 |

---

## 📁 Input Format

### NUCC Taxonomy (`nucc_taxonomy_master.csv`)
Must include columns:
```
code, classification, specialization, display_name
```

### Input Specialties (`input_specialties.csv`)
```csv
raw_specialty
Cardio
ENT Surgeon
OBGYN
Addiction Med.
Pediatric Surgery
```

### Optional Synonyms (`synonyms.csv`)
```csv
alias,normalized,code
obgyn,obstetrics and gynecology,
ent,otolaryngology,
cardio,cardiology,
```

---

## 📊 Output Format

| raw_specialty | nucc_codes | confidence | explain |
|----------------|-------------|-------------|----------|
| Cardio | 207RC0000X | 0.97 | synonym direct map for 'Cardio' |
| ENT Surgeon | 207YX0905X | 0.89 | hybrid TF-IDF + Jaccard |
| OBGYN | 207V00000X | 0.95 | exact match on 'obstetrics and gynecology' |
| xyz | JUNK | 0.00 | no candidates |

---

## 🧮 Algorithmic Components

| Stage | Function | Description |
|--------|------------|-------------|
| `normalize_text()` | Text preprocessing | Lowercasing, punctuation removal, whitespace normalization |
| `soundex()` | Phonetic encoding | Maps similar-sounding tokens |
| `build_tfidf()` | Character n-gram model | Uses scikit-learn TF-IDF vectorizer |
| `build_embeddings()` | Sentence transformer | Optional contextual similarity |
| `choose_codes_for_phrase()` | Hybrid scoring | Aggregates multiple similarity metrics |
| `trigram_jaccard_scores()` | Fallback | Works without sklearn or embeddings |
| `build_candidates()` | Candidate generation | Extracts canonical strings per NUCC code |

---

## 🧰 Architecture

```
standardize_2.py
│
├── normalize_text()            # Basic normalization
├── build_candidates()          # Generate searchable NUCC entries
├── build_tfidf()               # Create TF-IDF matrix
├── build_embeddings()          # Optional semantic embedding index
├── choose_codes_for_phrase()   # Core hybrid matching logic
├── load_synonyms()             # Read and register aliases
└── main()                      # CLI orchestrator
```

---

## ⚡ Example Run

```bash
python standardize_2.py \
  --nucc data/nucc_taxonomy_master.csv \
  --input data/input_specialties.csv \
  --out results/mapped_specialties.csv
```

Output sample:

```
Saved 5000 rows to results/mapped_specialties.csv
 raw_specialty     nucc_codes  confidence  explain
 Cardio            207RC0000X      0.97    synonym direct map for 'Cardio'
 ENT Surgeon       207YX0905X      0.89    hybrid TF-IDF + Jaccard
 xyz               JUNK           0.00    no candidates
```

---

## 🧑‍💻 Author

**Suyash Jindal**  
IIT Kanpur | Healthcare Data Standardization Project  
📧 [Contact via GitHub](https://github.com/<your-username>)

---

## 📜 License

MIT License © 2025 Suyash Jindal
