# 🏥 NUCC Provider Specialty Standardization Tool

### Overview
This tool intelligently maps **unstandardized healthcare provider specialties** (like “Cardio”, “ENT Surgeon”, “OBGYN”) to their corresponding **official NUCC Taxonomy Codes** as defined by the [National Uniform Claim Committee (NUCC)](https://www.nucc.org/).

The NUCC taxonomy is used across the U.S. healthcare system for:
- Provider credentialing  
- Claims processing  
- Network adequacy and data quality validation  

This system applies a **multi-stage matching pipeline** combining text normalization, phonetic encoding, fuzzy matching, and vector-based semantic similarity to ensure accurate mappings even when input data contains typos, abbreviations, or inconsistent wording.

---

## 🚀 Features

- **Multi-Stage Matching Pipeline**
  1. **Exact Match** (normalized lookup)  
  2. **Phonetic Match** (Soundex + Metaphone)  
  3. **Fuzzy Match** (Levenshtein + Jaro-Winkler)  
  4. **TF-IDF + Cosine Similarity** (character n-grams)  
  5. **N-Gram + Jaccard Word Similarity**

- **Synonym & Abbreviation Handling** (e.g., `ENT → Otolaryngology`, `OBGYN → Obstetrics Gynecology`)
- **Compound Specialty Splitting** (`"Cardio/Neuro"` → separate mappings for both)
- **Confidence Scoring & JUNK Detection**
- **Batch CSV Processing with Progress Updates**
- **Comprehensive Statistics Summary**

---

## 🧠 How It Works

For each input specialty:
1. The text is **normalized** — punctuation, stop words, and extra spaces removed.
2. The tool checks for an **exact match** in the NUCC taxonomy.
3. If no match is found, it applies phonetic, fuzzy, and semantic similarity algorithms.
4. Confidence scores are computed; results below the threshold are marked as `"JUNK"`.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/nucc-specialty-mapper.git
cd nucc-specialty-mapper
pip install -r requirements.txt
```

### Required Packages
```text
pandas
numpy
scikit-learn
jellyfish
```

(Install via `pip install pandas numpy scikit-learn jellyfish`)

---

## 📁 File Requirements

You will need:
1. **NUCC Taxonomy Master CSV**  
   Contains official NUCC taxonomy data.  
   Example columns:  
   `Code, Grouping, Classification, Specialization, Display_Name, Definition`

2. **Input CSV**  
   Must contain a column named:  
   `raw_specialty` (list of unstandardized provider specialties)

Example `input_specialties.csv`:
```csv
raw_specialty
Cardio
ENT Surgeon
OBGYN
Pediatric Surgery
Addiction Med.
```

---

## ⚙️ Usage

Run from the command line:

```bash
python nucc_mapper.py --nucc nucc_taxonomy_master.csv --input input_specialties.csv --output output_results.csv
```

Optional arguments:
```bash
--threshold   Confidence threshold (default: 0.70)
```

Example:
```bash
python nucc_mapper.py \
  --nucc nucc_taxonomy_master.csv \
  --input input_specialties.csv \
  --output standardized_output.csv \
  --threshold 0.75
```

---

## 📊 Output Format

The tool generates a CSV with the following columns:

| raw_specialty     | nucc_codes  | confidence | explain                         |
|-------------------|-------------|-------------|----------------------------------|
| Cardio            | 207RC0000X  | 0.98        | Exact match                      |
| ENT Surgeon       | 207YX0905X  | 0.87        | Phonetic match (Soundex)         |
| Addiction Med.    | 2084A0401X  | 0.79        | Fuzzy match (Levenshtein+JW)     |
| xyz               | JUNK        | 0.00        | No confident match found         |

At the end, a summary is printed:
```
✓ Processing complete!
  Total records: 5000
  Successfully mapped: 4500 (90.0%)
  Marked as JUNK: 500 (10.0%)
  Average confidence: 0.842
  Execution time: 37.42 seconds
```

---

## 🧩 Internal Logic Summary

| Stage | Algorithm | Confidence Weight | Purpose |
|-------|------------|------------------|----------|
| 1 | Exact Match | 1.00 | Perfect matches |
| 2 | Soundex + Metaphone | 0.85 | Pronunciation-based |
| 3 | Levenshtein + Jaro-Winkler | ≥0.80 | Handles typos, near matches |
| 4 | TF-IDF (char n-grams) | ≥0.40 | Contextual similarity |
| 5 | N-Gram + Jaccard | ≥0.60 | Partial overlap detection |

---

## 🧪 Example Run

```bash
python nucc_mapper.py --nucc data/nucc_taxonomy_master.csv \
                      --input data/input_specialties.csv \
                      --output results/nucc_mapped_output.csv
```

**Output Preview:**

```csv
raw_specialty,nucc_codes,confidence,explain
Cardio,207RC0000X,0.97,Exact match
OBGYN,207V00000X,0.88,Phonetic match (Soundex/Metaphone)
ENT Surgeon,207YX0905X,0.86,Fuzzy match (Levenshtein+Jaro-Winkler)
Addiction Med.,2084A0401X,0.83,TF-IDF cosine similarity
xyz,JUNK,0.00,No confident match found
```

---

## 🧰 Architecture

```
NUCCSpecialtyMapper
│
├── _normalize_text()         → cleans + expands synonyms
├── _build_indices()          → prepares TF-IDF & phonetic indices
├── _stage1_exact_match()     → direct lookup
├── _stage2_phonetic_match()  → soundex & metaphone
├── _stage3_fuzzy_match()     → edit-distance based
├── _stage4_tfidf_match()     → vector-based similarity
├── _stage5_ngram_match()     → set-based overlap
├── _handle_compound_specialties() → splits “Cardio/Neuro” cases
└── process_file()            → batch process and save output
```

---

## 📈 Performance Notes
- Ideal for input files up to **50,000+ specialties**
- Vectorization and phonetic indices improve runtime significantly
- Adjustable confidence threshold for stricter or looser mappings

---

## 🧑‍💻 Author
**Suyash Jindal**  
IIT Kanpur | Healthcare Data Standardization Project  
📧 [Contact via GitHub](https://github.com/<your-username>)

---

## 📜 License
MIT License © 2025 Suyash Jindal
