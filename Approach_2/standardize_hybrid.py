
"""
NUCC Provider Specialty Standardization Tool - HYBRID VERSION
Combines: Fuzzy Matching, TF-IDF, Phonetic Encoding, N-grams, Rule-based
Features: Synonym learning, adaptive thresholds, compound handling

Usage:
  python standardize_hybrid.py --nucc nucc_taxonomy_master.csv --input input.csv --out output.csv \
      [--synonyms synonyms.csv] [--threshold 0.65] [--learn-synonyms] [--save-learned learned_synonyms.csv]
"""

import argparse
import csv
import re
import unicodedata
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    import jellyfish
    JELLYFISH_AVAILABLE = True
except ImportError:
    JELLYFISH_AVAILABLE = False
    print("[WARN] jellyfish not installed. Phonetic matching disabled. Install: pip install jellyfish")

import difflib


def normalize_text(s: str) -> str:
    """Comprehensive text normalization"""
    if s is None or pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[\/\|\+\-_,;:]", " ", s)
    s = re.sub(r"[()\[\]{}]", " ", s)
    s = s.replace(".", "")
    s = s.replace("â€™", "'").replace("`", "'")
    s = s.replace("'", " ")
    s = re.sub(r"[^a-z0-9& ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    """Tokenize normalized text"""
    s = normalize_text(s)
    s = re.sub(r"\b&\b", " and ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split() if s else []


def split_segments(raw: str) -> List[str]:
    """Split compound specialties (e.g., 'Cardio/Diab' -> ['Cardio', 'Diab'])"""
    if not raw:
        return []
    parts = re.split(r"\s*(?:\/|,|\+|\band\b|\&|\|)\s*", str(raw), flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts if parts else [raw]

DEFAULT_SYNONYMS = {
    "ent": "otolaryngology",
    "obgyn": "obstetrics gynecology",
    "ob/gyn": "obstetrics gynecology",
    "ob-gyn": "obstetrics gynecology",
    "ob gyn": "obstetrics gynecology",
    "gyn": "gynecology",
    "gastro": "gastroenterology",
    "cardio": "cardiology",
    "endo": "endocrinology",
    "diab": "diabetes endocrinology",
    "fm": "family medicine",
    "fp": "family practice",
    "ped": "pediatrics",
    "peds": "pediatrics",
    "pediatric": "pediatrics",
    "ortho": "orthopedic surgery",
    "derm": "dermatology",
    "neuro": "neurology",
    "psych": "psychiatry",
    "neuropsych": "neuropsychiatry",
    "heme": "hematology",
    "onc": "oncology",
    "hemonc": "hematology oncology",
    "hem/onc": "hematology oncology",
    "optho": "ophthalmology",
    "ophtho": "ophthalmology",
    "ophthal": "ophthalmology",
    "rheum": "rheumatology",
    "pulm": "pulmonary disease",
    "nephro": "nephrology",
    "id": "infectious disease",
    "gi": "gastroenterology",
    "gu": "urology",
    "uro": "urology",
    "er": "emergency medicine",
    "ed": "emergency medicine",
    "im": "internal medicine",
    "anesthesia": "anesthesiology",
    "addiction med": "addiction medicine",
    "pain med": "pain medicine",
    "pain": "pain medicine",
    "pain and spine": "pain medicine",
    "pmr": "physical medicine rehabilitation",
    "pm&r": "physical medicine rehabilitation",
    "ot": "occupational therapy",
    "pt": "physical therapy",
    "pcp": "family medicine",
    "crit care": "critical care",
    "icu": "critical care",
    "path": "pathology",
    "rad": "radiology",
    "radio": "radiology",
    "ir": "interventional radiology",
    # Noise words (map to empty)
    "dept": "",
    "department": "",
    "clinic": "",
    "doctor": "",
    "md": "",
    "do": "",
    "doc": "",
    "services": "",
    "program": "",
    "unit": "",
    "center": "",
    "practice": "",
    "hospital": "",
    "group": "",
    "division": "",
}


def load_synonyms(path: str = None) -> Dict[str, str]:
    """Load synonyms from CSV or use defaults"""
    syn = dict(DEFAULT_SYNONYMS)
    if not path:
        return syn
    
    try:
        df = pd.read_csv(path)
        col_map = {c.lower().strip(): c for c in df.columns}
        key_col = col_map.get("key") or col_map.get("abbreviation")
        val_col = col_map.get("value") or col_map.get("expansion")
        
        if key_col and val_col:
            for _, row in df.iterrows():
                k = str(row[key_col]).strip()
                v = str(row[val_col]).strip()
                if k and not pd.isna(k):
                    syn[normalize_text(k)] = normalize_text(v)
            print(f"✓ Loaded {len(df)} custom synonyms from {path}")
    except Exception as e:
        print(f"[WARN] Failed to read synonyms from {path}: {e}. Using defaults.")
    
    return syn


def expand_synonyms(text: str, syn: Dict[str, str]) -> str:

    toks = tokenize(text)
    out = []
    for t in toks:
        nt = syn.get(t, t)
        if nt:
            out.extend(tokenize(nt))
    return " ".join(out)


def jaccard_similarity(a: List[str], b: List[str]) -> float:
   
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fuzzy_ratio(a: str, b: str) -> float:
 
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def ngram_similarity(s1: str, s2: str, n: int = 3) -> float:

    if not s1 or not s2:
        return 0.0
    
    def get_ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s)-n+1))
    
    ngrams1 = get_ngrams(s1, n)
    ngrams2 = get_ngrams(s2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 0.0


def levenshtein_similarity(s1: str, s2: str) -> float:
 
    if not JELLYFISH_AVAILABLE or not s1 or not s2:
        return 0.0
    distance = jellyfish.levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - (distance / max_len) if max_len > 0 else 0.0


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity (requires jellyfish)"""
    if not JELLYFISH_AVAILABLE or not s1 or not s2:
        return 0.0
    return jellyfish.jaro_winkler_similarity(s1, s2)


class NUCCIndex:
  
    
    def __init__(self, nucc_df: pd.DataFrame):
        self.df = self._prepare_dataframe(nucc_df)
        self.exact_match_index = {}
        self.phonetic_index = defaultdict(list)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        self._build_indices()
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
      
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        
      
        rename_map = {}
        if "provider_taxonomy_code" in df.columns and "code" not in df.columns:
            rename_map["provider_taxonomy_code"] = "code"
        if "displayname" in df.columns and "display_name" not in df.columns:
            rename_map["displayname"] = "display_name"
        df = df.rename(columns=rename_map)
        
        required = ["code", "classification", "specialization", "display_name", "grouping"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"NUCC CSV missing required column: {col}")
        
    
        if "status" in df.columns:
            status_series = df["status"].astype(str).str.lower()
            df = df[status_series.str.contains("active", na=False) |
                    (~status_series.str.contains("deprecated|inactive", na=False))]
        
      
        df["norm_classification"] = df["classification"].astype(str).apply(normalize_text)
        df["norm_specialization"] = df["specialization"].astype(str).apply(normalize_text)
        df["norm_display"] = df["display_name"].astype(str).apply(normalize_text)
        df["norm_grouping"] = df["grouping"].astype(str).apply(normalize_text)
        
    
        df["search_text"] = (
            df["norm_classification"] + " " +
            df["norm_specialization"] + " " +
            df["norm_display"]
        )
        
        return df
    
    def _build_indices(self):
    
        print("Building NUCC indices...")
        
      
        for idx, row in self.df.iterrows():
            for field in ["norm_classification", "norm_specialization", "norm_display"]:
                key = row[field]
                if key and len(key) > 0:
                    if key not in self.exact_match_index:
                        self.exact_match_index[key] = []
                    self.exact_match_index[key].append(idx)
        
    
        if JELLYFISH_AVAILABLE:
            for idx, row in self.df.iterrows():
                classification = row["norm_classification"]
                if classification:
                    first_word = classification.split()[0] if classification.split() else ""
                    if first_word:
                        try:
                            soundex = jellyfish.soundex(first_word)
                            metaphone = jellyfish.metaphone(first_word)
                            self.phonetic_index[soundex].append(idx)
                            self.phonetic_index[metaphone].append(idx)
                        except:
                            pass
        
       
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            min_df=1,
            max_features=10000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df["search_text"])
        
        print(f"✓ Indexed {len(self.df)} NUCC codes")

class HybridNUCCMapper:
    
    def __init__(self, nucc_index: NUCCIndex, synonyms: Dict[str, str], 
                 threshold: float = 0.65, topn: int = 5, learn_synonyms: bool = False):
        self.index = nucc_index
        self.synonyms = synonyms
        self.threshold = threshold
        self.topn = topn
        self.learn_synonyms = learn_synonyms
        self.learned_mappings = defaultdict(Counter) 
    
    def _stage1_exact_match(self, query_norm: str) -> Tuple[List[int], float, str]:
     
        if query_norm in self.index.exact_match_index:
            indices = self.index.exact_match_index[query_norm]
            return indices, 1.0, "exact_match"
        return [], 0.0, ""
    
    def _stage2_phonetic_match(self, query_norm: str) -> Tuple[List[int], float, str]:
      
        if not JELLYFISH_AVAILABLE or not query_norm:
            return [], 0.0, ""
        
        first_word = query_norm.split()[0] if query_norm.split() else ""
        if not first_word:
            return [], 0.0, ""
        
        indices = set()
        try:
            soundex = jellyfish.soundex(first_word)
            metaphone = jellyfish.metaphone(first_word)
            indices.update(self.index.phonetic_index.get(soundex, []))
            indices.update(self.index.phonetic_index.get(metaphone, []))
        except:
            pass
        
        if indices:
            return list(indices), 0.85, "phonetic_match"
        return [], 0.0, ""
    
    def _stage3_fuzzy_match(self, query: str, query_norm: str, query_tokens: List[str]) -> Tuple[List[int], float, str]:
    
        candidates = []
        
        for idx, row in self.index.df.iterrows():
            # Multi-metric scoring
            cls_norm = row["norm_classification"]
            
            # Fuzzy ratio
            fz = fuzzy_ratio(query_norm, cls_norm)
            
            # Jaccard token similarity
            cls_tokens = tokenize(cls_norm)
            jc = jaccard_similarity(query_tokens, cls_tokens)
            
            # N-gram similarity
            ng = ngram_similarity(query_norm, cls_norm, n=3)
            
            # Levenshtein (if available)
            lv = levenshtein_similarity(query_norm, cls_norm) if JELLYFISH_AVAILABLE else 0.0
            
            # Jaro-Winkler (if available)
            jw = jaro_winkler_similarity(query_norm, cls_norm) if JELLYFISH_AVAILABLE else 0.0
            
            # Weighted combination
            if JELLYFISH_AVAILABLE:
                score = 0.25*fz + 0.25*jc + 0.20*ng + 0.15*lv + 0.15*jw
            else:
                score = 0.35*fz + 0.35*jc + 0.30*ng
            
            # Exact classification bonus
            if cls_norm == query_norm and cls_norm:
                score += 0.2
            
            if score >= 0.5:
                candidates.append((idx, score))
        
        if candidates:
            candidates.sort(key=lambda x: -x[1])
            top_score = candidates[0][1]
            # Keep candidates within 5% of top score
            top_indices = [idx for idx, score in candidates if score >= top_score - 0.05]
            return top_indices, top_score, "fuzzy_match"
        
        return [], 0.0, ""
    
    def _stage4_tfidf_match(self, query_expanded: str) -> Tuple[List[int], float, str]:
        """Stage 4: TF-IDF cosine similarity"""
        if not query_expanded:
            return [], 0.0, ""
        
        query_vec = self.index.tfidf_vectorizer.transform([query_expanded])
        similarities = cosine_similarity(query_vec, self.index.tfidf_matrix).flatten()
        
        # Get top matches above absolute threshold
        TFIDF_THRESHOLD = 0.35
        valid_indices = np.where(similarities >= TFIDF_THRESHOLD)[0]
        
        if len(valid_indices) > 0:
            valid_scores = similarities[valid_indices]
            max_score = np.max(valid_scores)
            
            # Relative threshold: keep matches within 90% of best
            relative_cutoff = max_score * 0.9
            top_indices = valid_indices[valid_scores >= relative_cutoff]
            
            # Sort by score and limit
            sorted_idx = top_indices[np.argsort(-similarities[top_indices])][:self.topn]
            
            return sorted_idx.tolist(), float(max_score), "tfidf_match"
        
        return [], 0.0, ""
    
    def _check_general_classification(self, query_norm: str, indices: List[int]) -> Tuple[bool, List[int]]:
        """Check if query matches a general classification (return all subspecialties)"""
        if not indices:
            return False, []
        
        first_row = self.index.df.iloc[indices[0]]
        if query_norm == first_row["norm_classification"] and query_norm:
            # Find all codes with same classification
            cls_norm = first_row["norm_classification"]
            all_indices = self.index.df[self.index.df["norm_classification"] == cls_norm].index.tolist()
            return True, all_indices
        
        return False, indices
    
    def map_single_segment(self, segment: str) -> Dict[str, Any]:
        """Map a single specialty segment through the pipeline"""
        # Normalize and expand
        query_norm = normalize_text(segment)
        query_expanded = expand_synonyms(segment, self.synonyms)
        query_tokens = tokenize(query_expanded)
        query_expanded_str = " ".join(query_tokens)
        
        if not query_norm:
            return {
                "codes": [],
                "confidence": 0.0,
                "explain": "empty_after_normalization"
            }
        
        # Stage 1: Exact match
        indices, conf, method = self._stage1_exact_match(query_norm)
        if not indices:
            indices, conf, method = self._stage1_exact_match(query_expanded_str)
        
        # Stage 2: Phonetic match
        if not indices or conf < self.threshold:
            ph_indices, ph_conf, ph_method = self._stage2_phonetic_match(query_norm)
            if ph_conf > conf:
                indices, conf, method = ph_indices, ph_conf, ph_method
        
        # Stage 3: Fuzzy match
        if not indices or conf < self.threshold:
            fz_indices, fz_conf, fz_method = self._stage3_fuzzy_match(
                segment, query_norm, query_tokens
            )
            if fz_conf > conf:
                indices, conf, method = fz_indices, fz_conf, fz_method
        
        # Stage 4: TF-IDF match
        if not indices or conf < self.threshold:
            tf_indices, tf_conf, tf_method = self._stage4_tfidf_match(query_expanded_str)
            if tf_conf > conf:
                indices, conf, method = tf_indices, tf_conf, tf_method
        
        # Check for general classification match
        if indices:
            is_general, final_indices = self._check_general_classification(query_norm, indices)
            if is_general:
                indices = final_indices
                method = f"{method}+general_classification"
        
        # Filter by threshold
        if conf < self.threshold:
            return {
                "codes": [],
                "confidence": conf,
                "explain": f"low_confidence_{method}" if method else "no_match"
            }
        
        # Get codes
        codes = [self.index.df.iloc[idx]["code"] for idx in indices]
        
        # Learn synonyms
        if self.learn_synonyms and codes:
            self._record_mapping(query_norm, codes[0])
        
        return {
            "codes": codes,
            "confidence": conf,
            "explain": method
        }
    
    def map_specialty(self, raw_specialty: str) -> Dict[str, Any]:
        """Map raw specialty (handles compound specialties)"""
        if not raw_specialty or pd.isna(raw_specialty):
            return {
                "raw_specialty": raw_specialty,
                "nucc_codes": "JUNK",
                "confidence": 0.0,
                "explain": "empty_input"
            }
        
        raw_specialty = str(raw_specialty).strip()
        
        # Split compound specialties
        segments = split_segments(raw_specialty)
        
        all_codes = []
        confidences = []
        methods = []
        
        for seg in segments:
            if len(seg.strip()) < 2:
                continue
            
            result = self.map_single_segment(seg)
            
            if result["codes"]:
                all_codes.extend(result["codes"])
                confidences.append(result["confidence"])
                methods.append(f"[{seg}]:{result['explain']}")
            else:
                confidences.append(result["confidence"])
                methods.append(f"[{seg}]:{result['explain']}")
        
        # Deduplicate codes while preserving order
        seen = set()
        unique_codes = [c for c in all_codes if not (c in seen or seen.add(c))]
        
        if not unique_codes:
            return {
                "raw_specialty": raw_specialty,
                "nucc_codes": "JUNK",
                "confidence": max(confidences) if confidences else 0.0,
                "explain": "; ".join(methods) if methods else "no_segments"
            }
        
        # Aggregate confidence
        max_conf = max(confidences) if confidences else 0.0
        penalty = 0.05 if any(c < self.threshold for c in confidences) else 0.0
        agg_conf = max(0.0, min(1.0, max_conf - penalty))
        
        return {
            "raw_specialty": raw_specialty,
            "nucc_codes": " | ".join(unique_codes),
            "confidence": round(agg_conf, 4),
            "explain": "; ".join(methods)
        }
    
    def _record_mapping(self, query_norm: str, code: str):
        """Record successful mapping for synonym learning"""
        self.learned_mappings[query_norm][code] += 1
    
    def get_learned_synonyms(self, min_frequency: int = 3) -> Dict[str, str]:
        """Extract learned synonyms from mapping patterns"""
        learned = {}
        
        for query, code_counts in self.learned_mappings.items():
            if sum(code_counts.values()) >= min_frequency:
                # Get most common code
                most_common_code = code_counts.most_common(1)[0][0]
                
                # Get corresponding classification
                matching_rows = self.index.df[self.index.df["code"] == most_common_code]
                if not matching_rows.empty:
                    classification = matching_rows.iloc[0]["norm_classification"]
                    
                    # Only add if it's an abbreviation (shorter than classification)
                    if len(query) < len(classification) * 0.7:
                        learned[query] = classification
        
        return learned




def main():
    parser = argparse.ArgumentParser(
        description="NUCC Hybrid Mapper with Synonym Learning"
    )
    parser.add_argument("--nucc", required=True, help="Path to NUCC taxonomy master CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV with 'raw_specialty' column")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument("--synonyms", default=None, help="Optional synonyms CSV (columns: key, value)")
    parser.add_argument("--threshold", type=float, default=0.65, 
                       help="Confidence threshold (default: 0.65)")
    parser.add_argument("--topn", type=int, default=5, 
                       help="Max codes per match (default: 5)")
    parser.add_argument("--learn-synonyms", action="store_true",
                       help="Enable synonym learning from successful mappings")
    parser.add_argument("--save-learned", default=None,
                       help="Save learned synonyms to CSV")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  NUCC HYBRID SPECIALTY MAPPER")
    print("="*70)
    
    start_time = time.time()
    
    # Load NUCC data
    print(f"\n[1/5] Loading NUCC taxonomy from {args.nucc}...")
    nucc_df = pd.read_csv(args.nucc, dtype=str).fillna("")
    nucc_index = NUCCIndex(nucc_df)
    
    # Load synonyms
    print(f"\n[2/5] Loading synonyms...")
    synonyms = load_synonyms(args.synonyms)
    print(f"✓ Loaded {len(synonyms)} synonym mappings")
    
    # Initialize mapper
    print(f"\n[3/5] Initializing hybrid mapper...")
    mapper = HybridNUCCMapper(
        nucc_index=nucc_index,
        synonyms=synonyms,
        threshold=args.threshold,
        topn=args.topn,
        learn_synonyms=args.learn_synonyms
    )
    print(f"✓ Threshold: {args.threshold}, TopN: {args.topn}")
    if args.learn_synonyms:
        print("✓ Synonym learning: ENABLED")
    
    # Load input
    print(f"\n[4/5] Processing input from {args.input}...")
    inp_df = pd.read_csv(args.input, dtype=str).fillna("")
    
    # Normalize column names
    cols_lower = {c.strip().lower(): c for c in inp_df.columns}
    if "raw_specialty" not in cols_lower:
        raise ValueError("Input CSV must contain 'raw_specialty' column")
    
    raw_col = cols_lower["raw_specialty"]
    specialties = inp_df[raw_col].tolist()
    
    # Process all specialties
    results = []
    total = len(specialties)
    
    for i, raw in enumerate(specialties, 1):
        result = mapper.map_specialty(raw)
        results.append(result)
        
        if i % 1000 == 0 or i == total:
            print(f"  Progress: {i}/{total} ({i/total*100:.1f}%)", end="\r")
    
    print()  # New line after progress
    
    # Save output
    print(f"\n[5/5] Saving results to {args.out}...")
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    
    # Statistics
    elapsed = time.time() - start_time
    junk_count = (out_df["nucc_codes"] == "JUNK").sum()
    mapped_count = total - junk_count
    avg_conf = out_df[out_df["nucc_codes"] != "JUNK"]["confidence"].mean()
    
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    print(f"  Total records:       {total}")
    print(f"  Successfully mapped: {mapped_count} ({mapped_count/total*100:.1f}%)")
    print(f"  Marked as JUNK:      {junk_count} ({junk_count/total*100:.1f}%)")
    print(f"  Average confidence:  {avg_conf:.3f}")
    print(f"  Execution time:      {elapsed:.2f} seconds")
    print(f"  Throughput:          {total/elapsed:.1f} records/sec")
    print("="*70)
    
    # Save learned synonyms
    if args.learn_synonyms and args.save_learned:
        learned = mapper.get_learned_synonyms(min_frequency=3)
        if learned:
            learned_df = pd.DataFrame([
                {"key": k, "value": v} for k, v in learned.items()
            ])
            learned_df.to_csv(args.save_learned, index=False)
            print(f"\n✓ Saved {len(learned)} learned synonyms to {args.save_learned}")
    
    print(f"\n✓ Output saved to {args.out}\n")


if __name__ == "__main__":
    main()
