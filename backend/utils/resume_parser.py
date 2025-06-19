from pathlib import Path
from typing import Dict, Any
import pdfminer.high_level
import docx
import re

class ResumeParser:
    def __init__(self):
        self.skills_pattern = re.compile(r'\b(python|java|javascript|react|node\.js|sql|aws|docker|kubernetes|machine learning|ai|data science)\b', re.IGNORECASE)
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')

    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            return pdfminer.high_level.extract_text(file_path)
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")

    def extract_skills(self, text: str) -> list:
        """Extract skills from text."""
        return list(set(self.skills_pattern.findall(text.lower())))

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from text."""
        email = self.email_pattern.search(text)
        phone = self.phone_pattern.search(text)
        
        return {
            "email": email.group(0) if email else None,
            "phone": phone.group(0) if phone else None
        }

    def parse_resume(self, file_path: Path) -> Dict[str, Any]:
        """Parse resume and extract relevant information."""
        # Determine file type and extract text
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == '.docx':
            text = self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Extract information
        skills = self.extract_skills(text)
        contact_info = self.extract_contact_info(text)

        return {
            "filename": file_path.name,
            "skills": skills,
            "contact_info": contact_info,
            "raw_text": text
        }

# Example usage
if __name__ == "__main__":
    parser = ResumeParser()
    # Test with a sample resume
    # result = parser.parse_resume(Path("path/to/resume.pdf"))
    # print(json.dumps(result, indent=2)) 