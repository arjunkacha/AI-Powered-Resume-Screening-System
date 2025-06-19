import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.models.resume_parser import ResumeParser
import spacy
from sklearn.model_selection import train_test_split

class DataCollector:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.resume_parser = ResumeParser()
        self.nlp = spacy.load("en_core_web_sm")
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_resumes(self, source_dir: str) -> None:
        source_path = Path(source_dir)
    
        pdf_dir = self.raw_dir / "pdf"
        docx_dir = self.raw_dir / "docx"
        pdf_dir.mkdir(exist_ok=True)
        docx_dir.mkdir(exist_ok=True)
    
        files_copied = 0
        for file in source_path.glob("**/*"):
            if file.suffix.lower() == '.pdf':
                target = pdf_dir / file.name
                if not target.exists():
                    target.write_bytes(file.read_bytes())
                    print(f"Copied PDF: {file.name}")
                    files_copied += 1
            elif file.suffix.lower() == '.docx':
                target = docx_dir / file.name
                if not target.exists():
                    target.write_bytes(file.read_bytes())
                    print(f"Copied DOCX: {file.name}")
                    files_copied += 1
        print(f"Total files copied: {files_copied}")

    
    def process_resumes(self) -> pd.DataFrame:
        """Process all resumes and create a structured dataset."""
        processed_data = []
        
        # Process PDF files
        for pdf_file in tqdm(list(self.raw_dir.glob("pdf/*.pdf")), desc="Processing PDFs"):
            try:
                result = self.resume_parser.parse_resume(str(pdf_file))
                result['file_path'] = str(pdf_file)
                result['file_type'] = 'pdf'
            
                # Only append if raw_text exists and is not empty
                if 'raw_text' in result and result['raw_text'].strip():
                    processed_data.append(result)
                else:
                    print(f"Skipping {pdf_file} due to missing raw_text")
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")

        # Process DOCX files
        for docx_file in tqdm(list(self.raw_dir.glob("docx/*.docx")), desc="Processing DOCX"):
            try:
                result = self.resume_parser.parse_resume(str(docx_file))
                result['file_path'] = str(docx_file)
                result['file_type'] = 'docx'
                
                # Only append if raw_text exists and is not empty
                if 'raw_text' in result and result['raw_text'].strip():
                    processed_data.append(result)
                else:
                    print(f"Skipping {docx_file} due to missing raw_text")
            except Exception as e:
                print(f"Error processing {docx_file}: {str(e)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        # Save processed data
        df.to_csv(self.processed_dir / "processed_resumes.csv", index=False)
        df.to_json(self.processed_dir / "processed_resumes.json", orient="records")
        
        return df
    
    def tag_resumes(self, df: pd.DataFrame, job_titles: List[str]) -> pd.DataFrame:
        """Tag resumes with relevant job titles based on skills and experience."""
        if 'raw_text' not in df.columns:
            print("No 'raw_text' column found! Cannot tag resumes.")
            return df
        # Create a simple rule-based tagging system
        def get_job_title(text: str) -> str:
            doc = self.nlp(text.lower())
            
            # Define keywords for different job titles
            job_keywords = {
                'software_engineer': ['software', 'developer', 'programming', 'code'],
                'data_scientist': ['data science', 'machine learning', 'ai', 'analytics'],
                'devops_engineer': ['devops', 'aws', 'azure', 'kubernetes', 'docker'],
                'frontend_developer': ['frontend', 'react', 'angular', 'vue', 'javascript'],
                'backend_developer': ['backend', 'api', 'server', 'database']
            }
            
            # Count keyword matches
            matches = {job: sum(1 for kw in keywords if kw in text.lower())
                      for job, keywords in job_keywords.items()}
            
            # Return the job title with most matches
            if matches:
                return max(matches.items(), key=lambda x: x[1])[0]
            return 'other'
        
        # Apply tagging
        df['job_title'] = df['raw_text'].apply(get_job_title)
        
        # Save tagged data
        df.to_csv(self.processed_dir / "tagged_resumes.csv", index=False)
        df.to_json(self.processed_dir / "tagged_resumes.json", orient="records")
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """Split data into train and test sets."""
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # Save splits
        train_df.to_csv(self.processed_dir / "train_resumes.csv", index=False)
        test_df.to_csv(self.processed_dir / "test_resumes.csv", index=False)
        
        return {
            'train': train_df,
            'test': test_df
        }
    
    def perform_eda(self, df: pd.DataFrame) -> None:
        """Perform exploratory data analysis and generate visualizations."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create EDA directory
        eda_dir = self.processed_dir / "eda"
        eda_dir.mkdir(exist_ok=True)
        
        # 1. Skills distribution
        all_skills = [skill for skills in df['skills'] for skill in skills]
        skill_counts = pd.Series(all_skills).value_counts().head(20)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=skill_counts.values, y=skill_counts.index)
        plt.title("Top 20 Skills Distribution")
        plt.tight_layout()
        plt.savefig(eda_dir / "skills_distribution.png")
        plt.close()
        
        # 2. Job title distribution
        plt.figure(figsize=(10, 6))
        df['job_title'].value_counts().plot(kind='bar')
        plt.title("Job Title Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(eda_dir / "job_title_distribution.png")
        plt.close()
        
        # 3. Experience length distribution
        df['experience_length'] = df['experience'].apply(len)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='experience_length')
        plt.title("Experience Length Distribution")
        plt.tight_layout()
        plt.savefig(eda_dir / "experience_length_distribution.png")
        plt.close()
        
        # Save summary statistics
        summary_stats = {
            'total_resumes': len(df),
            'unique_skills': len(set(all_skills)),
            'avg_experience_length': df['experience_length'].mean(),
            'job_title_distribution': df['job_title'].value_counts().to_dict()
        }
        
        with open(eda_dir / "summary_stats.json", 'w') as f:
            json.dump(summary_stats, f, indent=4)

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Collect resumes from a source directory
    collector.collect_resumes("D:/Internship weekly tasks/week-1/resume-screening-system/resumes")
    # Process resumes
    df = collector.process_resumes()
    
    # Tag resumes with job titles
    job_titles = ['software_engineer', 'data_scientist', 'devops_engineer', 
                 'frontend_developer', 'backend_developer']
    df = collector.tag_resumes(df, job_titles)
    
    # Split data
    splits = collector.split_data(df)
    
    # Perform EDA
    collector.perform_eda(df)
