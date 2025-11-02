import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, process
import string
from typing import List, Tuple, Dict, Any
import argparse
import json

class NuccTaxonomyMapper:
    def __init__(self, nucc_file_path: str):
        self.nucc_df = pd.read_csv(nucc_file_path)
        print(f"Columns in NUCC file: {list(self.nucc_df.columns)}")
        self._preprocess_data()
        self._build_synonym_dict()
        self.confidence_threshold = 0.7
        
    def _preprocess_data(self):
        self.nucc_df['search_text'] = self.nucc_df.apply(
            self._create_search_text, axis=1
        )
        
        self.search_terms = []
        for idx, row in self.nucc_df.iterrows():
            terms = self._extract_search_terms(row)
            self.search_terms.extend(terms)
        
        self.search_terms = list(set(self.search_terms))
        print(f"Loaded {len(self.search_terms)} unique search terms from {len(self.nucc_df)} taxonomy codes")
        
    def _create_search_text(self, row) -> str:
        fields = [
            str(row['Classification']).lower() if pd.notna(row['Classification']) else '',
            str(row['Specialization']).lower() if pd.notna(row['Specialization']) else '',
            str(row['Display_Name']).lower() if pd.notna(row['Display_Name']) else '',
            str(row['Grouping']).lower() if pd.notna(row['Grouping']) else ''
        ]
        return ' '.join([f for f in fields if f])
    
    def _extract_search_terms(self, row) -> List[str]:
        terms = []
        
        if pd.notna(row['Classification']):
            classification = str(row['Classification']).lower().strip()
            terms.append(classification)
            
            if '&' in classification:
                parts = [p.strip() for p in classification.split('&')]
                terms.extend(parts)
            if '/' in classification:
                parts = [p.strip() for p in classification.split('/')]
                terms.extend(parts)
        
        if pd.notna(row['Specialization']):
            specialization = str(row['Specialization']).lower().strip()
            terms.append(specialization)
            
            if '&' in specialization:
                parts = [p.strip() for p in specialization.split('&')]
                terms.extend(parts)
        
        if pd.notna(row['Display_Name']):
            display_name = str(row['Display_Name']).lower().strip()
            terms.append(display_name)
            
            clean_display = re.sub(r'\b(physician|doctor|specialist|medicine|surgery)\b', '', display_name)
            terms.append(clean_display.strip())
        
        return [t for t in terms if t and len(t) > 2]
    
    def _build_synonym_dict(self):
        self.synonyms = {
            'ent': 'otolaryngology',
            'obgyn': 'obstetrics gynecology',
            'peds': 'pediatrics',
            'cardio': 'cardiology',
            'ortho': 'orthopedic',
            'neuro': 'neurology',
            'psych': 'psychiatry',
            'gi': 'gastroenterology',
            'endo': 'endocrinology',
            'rheum': 'rheumatology',
            'pulm': 'pulmonary',
            'neph': 'nephrology',
            'heme': 'hematology',
            'onc': 'oncology',
            'derm': 'dermatology',
            'uro': 'urology',
            'plastics': 'plastic surgery',
            'vascular': 'vascular surgery',
            'ct': 'cardiothoracic',
            'critical care': 'critical care medicine',
            'er': 'emergency medicine',
            'em': 'emergency medicine',
            'fp': 'family practice',
            'im': 'internal medicine',
            'pmr': 'physical medicine rehabilitation',
            'pm&r': 'physical medicine rehabilitation',
            'anesthesiolgy': 'anesthesiology',
            'anestesiology': 'anesthesiology',
            'pediatrician': 'pediatrics',
            'pediatric': 'pediatrics',
            'gastro': 'gastroenterology',
            'gastroent': 'gastroenterology',
            'cardiology': 'cardiovascular disease',
            'cardiac': 'cardiovascular',
            'surgical': 'surgery',
            'medical': 'internal medicine',
            'gen surgery': 'general surgery',
            'gen practice': 'general practice',
            'ent surgeon': 'otolaryngology',
            'heart doctor': 'cardiology',
            'skin doctor': 'dermatology',
            'bone doctor': 'orthopedic',
            'brain doctor': 'neurology',
            'cancer doctor': 'oncology',
            'kidney doctor': 'nephrology',
            'lung doctor': 'pulmonary',
        }
        
        additional_synonyms = {}
        for abbr, full in self.synonyms.items():
            if ' ' not in abbr and len(abbr) <= 4:
                additional_synonyms[full] = full
        self.synonyms.update(additional_synonyms)
    
    def _preprocess_input(self, text: str) -> str:
        if pd.isna(text) or text == '':
            return ''
            
        text = str(text).lower().strip()
        
        text = re.sub(r'\b(dept|department|clinic|center|division|unit|service|specialty|specialist|doctor|physician|dr|md|do)\b', '', text)
        text = re.sub(r'[^\w\s&/]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        replaced_words = []
        for word in words:
            if word in self.synonyms:
                replaced_words.extend(self.synonyms[word].split())
            else:
                replaced_words.append(word)
        
        return ' '.join(replaced_words)
    
    def _calculate_confidence(self, score: float, match_type: str) -> float:
        base_confidence = score / 100.0
        
        if match_type == 'exact':
            return min(1.0, base_confidence + 0.3)
        elif match_type == 'partial':
            return min(1.0, base_confidence + 0.1)
        else:
            return base_confidence
    
    def _find_matches(self, processed_input: str) -> List[Tuple[str, float, str]]:
        if not processed_input:
            return []
        
        matches = []
        
        for term in self.search_terms:
            if processed_input == term:
                matching_codes = self._get_codes_for_term(term)
                for code in matching_codes:
                    matches.append((code, 100.0, 'exact'))
        
        for idx, row in self.nucc_df.iterrows():
            search_text = row['search_text']
            
            ratio_score = fuzz.ratio(processed_input, search_text)
            partial_score = fuzz.partial_ratio(processed_input, search_text)
            token_score = fuzz.token_sort_ratio(processed_input, search_text)
            
            best_score = max(ratio_score, partial_score, token_score)
            
            if best_score >= 70:
                match_type = 'exact' if best_score >= 95 else 'partial'
                confidence = self._calculate_confidence(best_score, match_type)
                matches.append((row['Code'], confidence, match_type))
        
        input_words = set(processed_input.split())
        for idx, row in self.nucc_df.iterrows():
            classification = str(row['Classification']).lower() if pd.notna(row['Classification']) else ''
            specialization = str(row['Specialization']).lower() if pd.notna(row['Specialization']) else ''
            
            classification_words = set(classification.split())
            specialization_words = set(specialization.split()) if specialization else set()
            
            class_overlap = len(input_words.intersection(classification_words)) / len(classification_words) if classification_words else 0
            spec_overlap = len(input_words.intersection(specialization_words)) / len(specialization_words) if specialization_words else 0
            
            overlap_score = max(class_overlap, spec_overlap)
            if overlap_score > 0.5:
                confidence = self._calculate_confidence(overlap_score * 100, 'component')
                matches.append((row['Code'], confidence, 'component'))
        
        return matches
    
    def _get_codes_for_term(self, term: str) -> List[str]:
        codes = []
        for idx, row in self.nucc_df.iterrows():
            if (pd.notna(row['Classification']) and term in str(row['Classification']).lower()) or \
               (pd.notna(row['Specialization']) and term in str(row['Specialization']).lower()) or \
               (pd.notna(row['Display_Name']) and term in str(row['Display_Name']).lower()):
                codes.append(row['Code'])
        return codes
    
    def map_specialty(self, raw_specialty: str) -> Dict[str, Any]:
        if pd.isna(raw_specialty) or str(raw_specialty).strip() == '':
            return {
                'raw_specialty': raw_specialty,
                'nucc_codes': 'JUNK',
                'confidence': 0.0,
                'explain': 'Empty input'
            }
        
        processed_input = self._preprocess_input(raw_specialty)
        
        if not processed_input:
            return {
                'raw_specialty': raw_specialty,
                'nucc_codes': 'JUNK',
                'confidence': 0.0,
                'explain': 'No meaningful content after preprocessing'
            }
        
        matches = self._find_matches(processed_input)
        
        if not matches:
            return {
                'raw_specialty': raw_specialty,
                'nucc_codes': 'JUNK',
                'confidence': 0.0,
                'explain': f'No matches found for "{processed_input}"'
            }
        
        code_confidences = {}
        for code, confidence, match_type in matches:
            if code not in code_confidences or confidence > code_confidences[code]:
                code_confidences[code] = confidence
        
        valid_matches = [(code, conf) for code, conf in code_confidences.items() 
                        if conf >= self.confidence_threshold]
        valid_matches.sort(key=lambda x: x[1], reverse=True)
        
        if not valid_matches:
            return {
                'raw_specialty': raw_specialty,
                'nucc_codes': 'JUNK',
                'confidence': max([conf for _, conf in code_confidences.items()]) if code_confidences else 0.0,
                'explain': f'No matches above confidence threshold ({self.confidence_threshold})'
            }
        
        nucc_codes = '|'.join([code for code, _ in valid_matches])
        avg_confidence = np.mean([conf for _, conf in valid_matches])
        
        match_codes = [code for code, _ in valid_matches[:3]]
        explanation = f"Mapped to {len(valid_matches)} code(s) including: {', '.join(match_codes)}"
        
        return {
            'raw_specialty': raw_specialty,
            'nucc_codes': nucc_codes,
            'confidence': round(avg_confidence, 3),
            'explain': explanation
        }

def main():
    parser = argparse.ArgumentParser(description='Map raw provider specialties to NUCC taxonomy codes')
    parser.add_argument('--nucc', required=True, help='Path to NUCC taxonomy master CSV file')
    parser.add_argument('--input', required=True, help='Path to input CSV file with raw_specialty column')
    parser.add_argument('--out', required=True, help='Path to output CSV file')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    print("Loading NUCC taxonomy data...")
    mapper = NuccTaxonomyMapper(args.nucc)
    mapper.confidence_threshold = args.confidence
    
    print("Loading input specialties...")
    input_df = pd.read_csv(args.input)
    
    if 'raw_specialty' not in input_df.columns:
        raise ValueError("Input CSV must contain 'raw_specialty' column")
    
    print("Mapping specialties to NUCC codes...")
    results = []
    total = len(input_df)
    
    for i, row in input_df.iterrows():
        if i % 100 == 0:
            print(f"Processed {i}/{total} specialties...")
        
        result = mapper.map_specialty(row['raw_specialty'])
        results.append(result)
    
    output_df = pd.DataFrame(results)
    
    output_df.to_csv(args.out, index=False)
    print(f"Results saved to {args.out}")
    
    junk_count = len(output_df[output_df['nucc_codes'] == 'JUNK'])
    mapped_count = len(output_df) - junk_count
    
    print(f"\nMapping Summary:")
    print(f"Total specialties processed: {len(output_df)}")
    print(f"Successfully mapped: {mapped_count} ({mapped_count/len(output_df)*100:.1f}%)")
    print(f"Marked as JUNK: {junk_count} ({junk_count/len(output_df)*100:.1f}%)")

if __name__ == "__main__":
    main()
