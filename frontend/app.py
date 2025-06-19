import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://localhost:8000"

def parse_resume(file):
    files = {"file": file}
    response = requests.post(f"{API_BASE_URL}/parse-resume", files=files)
    return response.json()

def rank_resumes(job_description, resumes):
    data = {
        "job_description": {"text": job_description},
        "resumes": resumes
    }
    response = requests.post(f"{API_BASE_URL}/rank-resumes", json=data)
    return response.json()

def main():
    st.title("AI-Powered Resume Screening System")
    st.write("Upload resumes and job descriptions to screen candidates automatically.")

    # Sidebar for job description
    st.sidebar.header("Job Description")
    job_description = st.sidebar.text_area(
        "Enter the job description:",
        height=300,
        help="Paste the job description here to screen resumes against it."
    )

    # Main content area
    st.header("Upload Resumes")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose resume files (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files and job_description:
        if st.button("Screen Resumes"):
            with st.spinner("Processing resumes..."):
                # Parse resumes
                parsed_resumes = []
                for file in uploaded_files:
                    result = parse_resume(file)
                    parsed_resumes.append(result)

                # Rank resumes
                ranked_results = rank_resumes(job_description, parsed_resumes)

                # Display results
                st.header("Ranked Candidates")

                # Create DataFrame for better display
                results_data = []
                for idx, result in enumerate(ranked_results, 1):
                    resume = result["resume"]
                    score = result["similarity_score"]

                    results_data.append({
                        "Rank": idx,
                        "Name": resume.get("name", "N/A"),
                        "Skills": ", ".join(resume.get("skills", [])),
                        "Similarity Score": f"{score:.2%}"
                    })

                df = pd.DataFrame(results_data)
                st.dataframe(df)

                # Display detailed view
                st.header("Detailed Analysis")
                for idx, result in enumerate(ranked_results, 1):
                    resume = result["resume"]
                    score = result["similarity_score"]

                    with st.expander(f"{idx}. {resume.get('name', 'N/A')} - {score:.2%}"):
                        st.subheader("Skills")
                        st.write(", ".join(resume.get("skills", [])))

                        st.subheader("Experience")
                        for exp in resume.get("experience", []):
                            st.write(f"**{exp['date_range']}**")
                            st.write(exp["description"])
                            st.write("---")

if __name__ == "__main__":
    main() 