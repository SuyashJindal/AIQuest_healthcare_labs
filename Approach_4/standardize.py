"""
NUCC Provider Specialty Standardization Tool
Combines multiple algorithms: Fuzzy Matching, TF-IDF, Phonetic Encoding, N-grams
Author: HiLabs Challenge Solution
"""

import pandas as pd
import numpy as np
import re
import jellyfish
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import time

class NUCCSpecialtyMapper:
    def __init__(self, nucc_csv_path, confidence_threshold=0.70):
        """
        Initialize mapper with NUCC taxonomy data
        
        Args:
            nucc_csv_path: Path to NUCC master CSV
            confidence_threshold: Minimum confidence for valid mapping
        """
        self.confidence_threshold = confidence_threshold
        self.nucc_df = pd.read_csv(nucc_csv_path)
        
        # Standardize column names (handle case variations)
        self.nucc_df.columns = self.nucc_df.columns.str.strip()
        
        # Map to expected column names
        column_mapping = {
            'Code': 'code',
            'Grouping': 'grouping',
            'Classification': 'classification',
            'Specialization': 'specialization',
            'Display_Name': 'display_name',
            'Definition': 'definition',
            'Notes': 'notes',
            'Section': 'section'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in self.nucc_df.columns:
                self.nucc_df.rename(columns={old_col: new_col}, inplace=True)
        
        # Medical abbreviation dictionary
        self.synonyms = {
            'ent': 'otolaryngology',
            'obgyn': 'obstetrics gynecology',
            'ob/gyn': 'obstetrics gynecology',
            'ob-gyn': 'obstetrics gynecology',
            'cardio': 'cardiology',
            'peds': 'pediatrics',
            'neuro': 'neurology',
            'psych': 'psychiatry',
            'ortho': 'orthopedic surgery',
            'derm': 'dermatology',
            'gi': 'gastroenterology',
            'pulm': 'pulmonology',
            'endo': 'endocrinology',
            'heme': 'hematology',
            'onc': 'oncology',
            'nephro': 'nephrology',
            'rheum': 'rheumatology',
            'uro': 'urology',
            'ophthal': 'ophthalmology',
            'pm&r': 'physical medicine rehabilitation',
            'pmr': 'physical medicine rehabilitation',
            'er': 'emergency medicine',
            'im': 'internal medicine',
            'fp': 'family practice',
            'fm': 'family medicine',
            'addiction med': 'addiction medicine',
            'pain': 'pain medicine',
            'crit care': 'critical care',
            'icu': 'critical care',
            'anesthesia': 'anesthesiology',
            'path': 'pathology',
            'rad': 'radiology',
            'radio': 'radiology'
        }
        
        self._build_indices()
        print(f"✓ Loaded {len(self.nucc_df)} NUCC taxonomy codes")
    
    def _normalize_text(self, text):
        """Normalize text for matching"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Apply synonym replacements
        words = text.split()
        normalized_words = []
        for word in words:
            if word in self.synonyms:
                normalized_words.append(self.synonyms[word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def _build_indices(self):
        """Build all matching indices"""
        print("Building indices...")
        
        # Normalize NUCC data
        self.nucc_df['norm_classification'] = self.nucc_df['classification'].apply(
            self._normalize_text
        )
        self.nucc_df['norm_specialization'] = self.nucc_df['specialization'].apply(
            self._normalize_text
        )
        self.nucc_df['norm_display'] = self.nucc_df['display_name'].apply(
            self._normalize_text
        )
        
        # Combined search text
        self.nucc_df['search_text'] = (
            self.nucc_df['norm_classification'] + ' ' +
            self.nucc_df['norm_specialization'] + ' ' +
            self.nucc_df['norm_display']
        )
        
        # 1. Exact match indices
        self.exact_match = {}
        for idx, row in self.nucc_df.iterrows():
            for field in ['norm_classification', 'norm_specialization', 'norm_display']:
                key = row[field]
                if key and len(key) > 0:
                    if key not in self.exact_match:
                        self.exact_match[key] = []
                    self.exact_match[key].append({
                        'code': row['code'],
                        'display': row['display_name']
                    })
        
        # 2. Phonetic indices (Soundex + Double Metaphone)
        self.soundex_index = defaultdict(list)
        self.metaphone_index = defaultdict(list)
        
        for idx, row in self.nucc_df.iterrows():
            classification = row['norm_classification']
            if classification:
                # Soundex
                try:
                    soundex_code = jellyfish.soundex(classification.split()[0])
                    self.soundex_index[soundex_code].append({
                        'code': row['code'],
                        'display': row['display_name']
                    })
                except:
                    pass
                
                # Double Metaphone
                try:
                    primary, secondary = jellyfish.metaphone(classification.split()[0]), None
                    if primary:
                        self.metaphone_index[primary].append({
                            'code': row['code'],
                            'display': row['display_name']
                        })
                except:
                    pass
        
        # 3. TF-IDF vectorization with character n-grams
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),  # Character n-grams
            analyzer='char_wb',
            min_df=1,
            max_features=5000
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.nucc_df['search_text']
        )
        
        print("✓ Indices built successfully")
    
    def _levenshtein_similarity(self, s1, s2):
        """Calculate normalized Levenshtein similarity"""
        if not s1 or not s2:
            return 0.0
        distance = jellyfish.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len)
    
    def _jaro_winkler_similarity(self, s1, s2):
        """Calculate Jaro-Winkler similarity"""
        if not s1 or not s2:
            return 0.0
        return jellyfish.jaro_winkler_similarity(s1, s2)
    
    def _ngram_similarity(self, s1, s2, n=3):
        """Calculate character n-gram similarity (Jaccard)"""
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
    
    def _jaccard_word_similarity(self, s1, s2):
        """Calculate word-level Jaccard similarity"""
        if not s1 or not s2:
            return 0.0
        
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _stage1_exact_match(self, normalized_input):
        """Stage 1: Exact matching"""
        if normalized_input in self.exact_match:
            matches = self.exact_match[normalized_input]
            return [m['code'] for m in matches], 1.0, "Exact match"
        return None, 0.0, ""
    
    def _stage2_phonetic_match(self, normalized_input):
        """Stage 2: Phonetic matching (Soundex + Metaphone)"""
        if not normalized_input:
            return None, 0.0, ""
        
        first_word = normalized_input.split()[0]
        matches = []
        
        # Try Soundex
        try:
            soundex_code = jellyfish.soundex(first_word)
            if soundex_code in self.soundex_index:
                matches.extend(self.soundex_index[soundex_code])
        except:
            pass
        
        # Try Metaphone
        try:
            metaphone_code = jellyfish.metaphone(first_word)
            if metaphone_code in self.metaphone_index:
                matches.extend(self.metaphone_index[metaphone_code])
        except:
            pass
        
        if matches:
            # Deduplicate
            unique_codes = list(set(m['code'] for m in matches))
            return unique_codes, 0.85, "Phonetic match (Soundex/Metaphone)"
        
        return None, 0.0, ""
    
    def _stage3_fuzzy_match(self, normalized_input):
        """Stage 3: Fuzzy string matching (Levenshtein + Jaro-Winkler)"""
        best_matches = []
        
        for idx, row in self.nucc_df.iterrows():
            # Calculate similarities
            lev_sim = self._levenshtein_similarity(
                normalized_input, 
                row['norm_classification']
            )
            jw_sim = self._jaro_winkler_similarity(
                normalized_input,
                row['norm_classification']
            )
            
            # Combined score (weighted average)
            combined_score = 0.5 * lev_sim + 0.5 * jw_sim
            
            if combined_score >= 0.80:
                best_matches.append({
                    'code': row['code'],
                    'display': row['display_name'],
                    'score': combined_score
                })
        
        if best_matches:
            # Sort by score
            best_matches.sort(key=lambda x: x['score'], reverse=True)
            top_score = best_matches[0]['score']
            top_codes = [m['code'] for m in best_matches if m['score'] >= top_score - 0.05]
            return top_codes, top_score, "Fuzzy match (Levenshtein+Jaro-Winkler)"
        
        return None, 0.0, ""
    
    def _stage4_tfidf_match(self, normalized_input):
        """Stage 4: TF-IDF with cosine similarity"""
        if not normalized_input:
            return None, 0.0, ""
        
        # Vectorize input
        input_vec = self.tfidf_vectorizer.transform([normalized_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[-10:][::-1]
        top_scores = similarities[top_indices]
        
        # Filter by threshold
        valid_matches = []
        for idx, score in zip(top_indices, top_scores):
            if score >= 0.40:
                valid_matches.append({
                    'code': self.nucc_df.iloc[idx]['code'],
                    'display': self.nucc_df.iloc[idx]['display_name'],
                    'score': score
                })
        
        if valid_matches:
            top_score = valid_matches[0]['score']
            top_codes = [m['code'] for m in valid_matches if m['score'] >= top_score - 0.1]
            return top_codes, top_score, "TF-IDF cosine similarity"
        
        return None, 0.0, ""
    
    def _stage5_ngram_match(self, normalized_input):
        """Stage 5: N-gram similarity"""
        best_matches = []
        
        for idx, row in self.nucc_df.iterrows():
            # Calculate n-gram similarity (trigrams)
            ngram_sim = self._ngram_similarity(
                normalized_input,
                row['norm_classification'],
                n=3
            )
            
            # Word-level Jaccard
            jaccard_sim = self._jaccard_word_similarity(
                normalized_input,
                row['norm_classification']
            )
            
            # Combined score
            combined_score = 0.6 * ngram_sim + 0.4 * jaccard_sim
            
            if combined_score >= 0.60:
                best_matches.append({
                    'code': row['code'],
                    'display': row['display_name'],
                    'score': combined_score
                })
        
        if best_matches:
            best_matches.sort(key=lambda x: x['score'], reverse=True)
            top_score = best_matches[0]['score']
            top_codes = [m['code'] for m in best_matches if m['score'] >= top_score - 0.05]
            return top_codes, top_score, "N-gram + Jaccard similarity"
        
        return None, 0.0, ""
    
    def _handle_compound_specialties(self, raw_specialty):
        """Handle compound specialties like 'Cardio/Diab' or 'Pain & Spine'"""
        # Split on common delimiters
        delimiters = r'[/&,\-]|\sand\s'
        parts = re.split(delimiters, raw_specialty, flags=re.IGNORECASE)
        
        if len(parts) > 1:
            all_codes = []
            all_confidences = []
            
            for part in parts:
                part = part.strip()
                if len(part) >= 3:  # Minimum length
                    codes, conf, _ = self.map_specialty(part)
                    if codes and codes != ['JUNK']:
                        all_codes.extend(codes)
                        all_confidences.append(conf)
            
            if all_codes:
                # Remove duplicates while preserving order
                unique_codes = list(dict.fromkeys(all_codes))
                avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
                return unique_codes, avg_confidence, "Compound specialty split"
        
        return None, 0.0, ""
    
    def map_specialty(self, raw_specialty):
        """
        Map raw specialty to NUCC codes using multi-stage pipeline
        
        Returns:
            codes: List of NUCC codes or ['JUNK']
            confidence: Float between 0-1
            explanation: String explaining the match
        """
        if pd.isna(raw_specialty) or not str(raw_specialty).strip():
            return ['JUNK'], 0.0, "Empty input"
        
        raw_specialty = str(raw_specialty).strip()
        
        # Check if it's clearly junk (too short or random)
        if len(raw_specialty) < 3:
            return ['JUNK'], 0.0, "Input too short"
        
        # Check for compound specialties first
        compound_result = self._handle_compound_specialties(raw_specialty)
        if compound_result[0]:
            codes, conf, explain = compound_result
            if conf >= self.confidence_threshold:
                return codes, conf, explain
        
        # Normalize input
        normalized_input = self._normalize_text(raw_specialty)
        
        if not normalized_input:
            return ['JUNK'], 0.0, "Invalid input after normalization"
        
        # Stage 1: Exact Match
        codes, conf, explain = self._stage1_exact_match(normalized_input)
        if codes and conf >= self.confidence_threshold:
            return codes, conf, explain
        
        # Stage 2: Phonetic Match
        codes, conf, explain = self._stage2_phonetic_match(normalized_input)
        if codes and conf >= self.confidence_threshold:
            return codes, conf, explain
        
        # Stage 3: Fuzzy Match (Levenshtein + Jaro-Winkler)
        codes, conf, explain = self._stage3_fuzzy_match(normalized_input)
        if codes and conf >= self.confidence_threshold:
            return codes, conf, explain
        
        # Stage 4: TF-IDF Match
        codes, conf, explain = self._stage4_tfidf_match(normalized_input)
        if codes and conf >= self.confidence_threshold:
            return codes, conf, explain
        
        # Stage 5: N-gram Match
        codes, conf, explain = self._stage5_ngram_match(normalized_input)
        if codes and conf >= self.confidence_threshold:
            return codes, conf, explain
        
        # No confident match found
        return ['JUNK'], 0.0, "No confident match found"
    
    def process_file(self, input_csv_path, output_csv_path):
        """
        Process input CSV and generate output with mappings
        
        Args:
            input_csv_path: Path to input CSV with 'raw_specialty' column
            output_csv_path: Path to save output CSV
        """
        print(f"\nProcessing {input_csv_path}...")
        start_time = time.time()
        
        # Read input
        input_df = pd.read_csv(input_csv_path)
        
        if 'raw_specialty' not in input_df.columns:
            raise ValueError("Input CSV must have 'raw_specialty' column")
        
        results = []
        total = len(input_df)
        
        for idx, row in input_df.iterrows():
            raw_specialty = row['raw_specialty']
            
            # Map specialty
            codes, confidence, explanation = self.map_specialty(raw_specialty)
            
            # Format output
            nucc_codes = '|'.join(codes) if codes != ['JUNK'] else 'JUNK'
            
            results.append({
                'raw_specialty': raw_specialty,
                'nucc_codes': nucc_codes,
                'confidence': round(confidence, 4),
                'explain': explanation
            })
            
            # Progress indicator
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{total} records...")
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Save to CSV
        output_df.to_csv(output_csv_path, index=False)
        
        elapsed_time = time.time() - start_time
        
        # Statistics
        junk_count = len(output_df[output_df['nucc_codes'] == 'JUNK'])
        mapped_count = total - junk_count
        avg_confidence = output_df[output_df['nucc_codes'] != 'JUNK']['confidence'].mean()
        
        print(f"\n✓ Processing complete!")
        print(f"  Total records: {total}")
        print(f"  Successfully mapped: {mapped_count} ({mapped_count/total*100:.1f}%)")
        print(f"  Marked as JUNK: {junk_count} ({junk_count/total*100:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Execution time: {elapsed_time:.2f} seconds")
        print(f"  Output saved to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='NUCC Provider Specialty Standardization Tool'
    )
    parser.add_argument(
        '--nucc',
        required=True,
        help='Path to NUCC taxonomy master CSV'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input specialties CSV'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save output CSV'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.70,
        help='Confidence threshold (default: 0.70)'
    )
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = NUCCSpecialtyMapper(
        nucc_csv_path=args.nucc,
        confidence_threshold=args.threshold
    )
    
    # Process file
    mapper.process_file(
        input_csv_path=args.input,
        output_csv_path=args.output
    )


if __name__ == '__main__':
    main()
