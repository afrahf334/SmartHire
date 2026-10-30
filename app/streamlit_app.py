import streamlit as st
from pypdf import PdfReader
from docx import Document
import joblib
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(
    page_title="SmartHire",
    page_icon="💼",
    layout="wide"
)
job_tfidf = joblib.load("models/job_tfidf_vectorizer_new.pkl")
st.write("TF-IDF vocabulary size:", len(job_tfidf.vocabulary_))
job_kmeans = joblib.load("models/job_kmeans_new.pkl")
jobs = pd.read_csv("data/processed/processed_jobs_new.csv")
resume_tfidf = joblib.load("models/tfidf_vectorizer.pkl")
classifier = joblib.load("models/classifier.pkl")
st.write("TF-IDF vocabulary size:", len(job_tfidf.vocabulary_))
st.write("Job dataset shape:", jobs.shape)
st.write("Job dataset columns:", jobs.columns.tolist())
st.write("Job vectorizer vocabulary size:", len(job_tfidf.vocabulary_))
st.write("Resume vectorizer vocabulary size:", len(resume_tfidf.vocabulary_))
st.write("Job text sample:", jobs["Job_Text"].iloc[0][:300])

st.title("💼 SmartHire")
st.subheader("Resume-to-Job Matching & Career Guidance Engine")

st.write(
    "Upload your resume to discover suitable job categories, "
    "matching job opportunities, and skill gaps."
)

st.success("SmartHire app started successfully!")
st.divider()

st.header("📄 Upload Your Resume")

resume_text = ""
uploaded_file = st.file_uploader(
    "Choose your resume",
    type=["pdf", "docx"]
)
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            resume_text = "\n".join(
                page.extract_text() or ""
                for page in pdf_reader.pages
            )

        else:
            document = Document(uploaded_file)
            resume_text = "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        if resume_text.strip():
            st.success("Resume uploaded successfully!")

            st.subheader("Extracted Resume Text")

            st.text_area(
                "Resume content",
                resume_text,
                height=250
            )
        else:
            st.warning(
                "No readable text found. "
                "Please try a text-based PDF or DOCX file."
            )

    except Exception as e:
        st.error(f"Could not read the resume: {e}")    

st.subheader("💼 Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=250,
    placeholder="Paste the job description..."
)

if job_description:
    st.success("Job description added successfully!")

    if uploaded_file is not None and job_description and resume_text.strip():

        st.write(
            "Resume non-zero TF-IDF features:",
            job_tfidf.transform([resume_text]).nnz
        )

        st.write(
            "Job description non-zero TF-IDF features:",
            job_tfidf.transform([job_description]).nnz
        )

        resume_vector = job_tfidf.transform([resume_text])
        job_vector = job_tfidf.transform([job_description])

        resume_terms = set(job_tfidf.inverse_transform(resume_vector)[0])
        job_terms = set(job_tfidf.inverse_transform(job_vector)[0])

        st.write("Shared terms:", sorted(resume_terms & job_terms))
        st.write("Number of shared terms:", len(resume_terms & job_terms))
        similarity_score = cosine_similarity(
            resume_vector,
            job_vector
        )[0][0]

        st.write("Raw cosine similarity:", similarity_score)

        match_percentage = round(similarity_score * 100, 2)

        st.subheader("🎯 Resume ↔ Job Match")
        st.metric("Match Score", f"{match_percentage}%")

if uploaded_file is not None and resume_text.strip():

    resume_features = resume_tfidf.transform([resume_text])
    predicted_category = classifier.predict(resume_features)[0]

    st.subheader("🎓 Predicted Job Category")
    st.success(predicted_category)

    resume_lower = resume_text.lower()

    resume_job_vector = job_tfidf.transform([resume_text])

    job_similarities = cosine_similarity(
        resume_job_vector,
        job_tfidf.transform(jobs["Job_Text"])
    ).flatten()

    hybrid_scores = []

    category_keywords = {
        "Data Science": [
            "data scientist", "data science", "machine learning engineer",
            "data analyst", "analytics engineer", "statistician", "ai engineer"
        ],
        "Python Developer": ["python developer", "python engineer", "django developer"],
        "Java Developer": ["java developer", "java engineer"],
        "Automation Testing": ["automation testing", "test automation", "qa automation"],
        "Business Analyst": ["business analyst", "business analysis"],
        "Database": ["database administrator", "database developer", "sql developer"],
        "DevOps Engineer": ["devops", "site reliability", "cloud engineer", "platform engineer"],
        "Web Designing": ["web designer", "web design", "ui designer", "ux designer"],
        "Electrical Engineering": ["electrical engineer", "electrical engineering"],
        "Mechanical Engineer": ["mechanical engineer", "mechanical design engineer"],
        "Civil Engineer": ["civil engineer", "structural engineer"],
        "Blockchain": ["blockchain", "web3", "smart contract"],
        "Hadoop": ["hadoop", "big data engineer", "spark engineer"],
        "ETL Developer": ["etl developer", "data integration"],
        "Network Security Engineer": ["network security", "cybersecurity", "security engineer"],
        "DotNet Developer": ["dotnet developer", ".net developer", "asp.net developer"],
        "SAP Developer": ["sap developer", "sap consultant", "sap abap"],
        "Testing": ["software tester", "test engineer", "quality assurance"],
        "Advocate": ["lawyer", "advocate", "attorney", "legal counsel"],
        "Arts": ["artist", "illustrator", "graphic designer"],
        "HR": ["human resources", "hr manager", "recruiter"],
        "Sales": ["sales representative", "sales manager", "account executive"],
        "Operations Manager": ["operations manager", "operations director"],
        "PMO": ["project management office", "pmo", "program manager"],
        "Health and fitness": ["fitness trainer", "personal trainer", "health coach"]
    }

    known_skills = [
        "machine learning", "deep learning",
        "natural language processing", "artificial intelligence",
        "data analysis", "data mining", "data science",
        "information retrieval", "pattern recognition",
        "web services", "core java", "javascript",
        "python", "java", "sql", "nosql", "c++",
        "hadoop", "hbase", "linux", "angular", "django",
        "html", "css", "git", "android", "mysql",
        "pl sql", "pyspark", "pandas", "numpy",
        "scikit-learn", "tensorflow", "pytorch"
    ]

    for index in range(len(jobs)):
        text_score = job_similarities[index]
        skills = str(jobs.iloc[index]["Required Skills"]).lower()

        job_skills = [
            skill for skill in known_skills
            if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", skills)
        ]

        matched_skills = [
            skill for skill in job_skills
            if re.search(
                r"(?<!\w)" + re.escape(skill) + r"(?!\w)",
                resume_lower
            )
        ]

        if job_skills:
            skill_score = len(matched_skills) / len(job_skills)
        else:
            skill_score = 0

        job_title = str(jobs.iloc[index]["Job Title"]).lower()

        category_terms = category_keywords.get(
            predicted_category, [predicted_category.lower()]
        )
        category_score = float(
            any(term in job_title for term in category_terms)
        )

        hybrid_score = (
            0.55 * text_score
            + 0.25 * skill_score
            + 0.20 * category_score
        )

        hybrid_scores.append(hybrid_score)

    hybrid_scores = pd.Series(hybrid_scores)
    top_job_indices = hybrid_scores.nlargest(5).index

    st.subheader("💼 Top 5 Recommended Jobs")
    for rank, index in enumerate(top_job_indices, start=1):
        job = jobs.iloc[index]
        st.write( f"**{rank}. {job['Job Title']}** — " f"{job['Company']} | {job['Location']}" )
        st.write( f"SmartMatch Score: **{hybrid_scores[index] * 100:.2f}%**" )
        st.write( f"Required Skills: {job['Required Skills']}" )
        st.divider()

if uploaded_file is not None and resume_text.strip():

    resume_lower = resume_text.lower()
    print("DEBUG resume text:", resume_lower[:500])

    recommended_jobs = jobs.iloc[top_job_indices]


    import re

    skill_vocabulary = [
        "machine learning",
        "deep learning",
        "natural language processing",
        "artificial intelligence",
        "data analysis",
        "data mining",
        "data science",
        "information retrieval",
        "pattern recognition",
        "web services",
        "core java",
        "javascript",
        "jquery",
        "postgres",
        "python",
        "java",
        "sql",
        "nosql",
        "hadoop",
        "hbase",
        "linux",
        "angular",
        "django",
        "html",
        "css",
        "c++",
        "git",
        "android",
        "mysql",
        "pl sql",
        "pyspark",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pytorch",
    ]

    skill_counts = {}

    for skills in recommended_jobs["Required Skills"].dropna():
        text = str(skills).lower()
        found_in_job = set()

        for skill in sorted(skill_vocabulary, key=len, reverse=True):
            pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

            if re.search(pattern, text):
                found_in_job.add(skill)

        for skill in found_in_job:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    top_cluster_skills = sorted(
        skill_counts,
        key=skill_counts.get,
        reverse=True
    )[:10]

    print("DEBUG skills:", list(skill_counts.keys())[:20])

    skills_found = []
    skills_missing = []

    for skill in top_cluster_skills:
        skill = str(skill).strip().lower()

        if not skill:
            continue

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, resume_lower):
            if skill not in skills_found:
                skills_found.append(skill)
        else:
            if skill not in skills_missing:
                skills_missing.append(skill)


    st.subheader("🧩 Skill Gap Analysis")

    if skills_found:
        st.write(
            "**Skills found in your resume:** "
            + ", ".join(skills_found)
        )
    else:
        st.write("No matching skills were identified in your resume.")

    if skills_missing:
        st.warning(
            "**Skills to improve for recommended jobs:** "
            + ", ".join(skills_missing)
        )
    else:
        st.success(
            "Your resume covers the main skills identified "
            "in the recommended jobs."
        )

    recommended_jobs = jobs.iloc[top_job_indices]
    top_roles = recommended_jobs["Job Title"].value_counts().head(3)

    st.subheader("🔎 Recommended Job Areas")
    st.info("These job areas are based on your top matching jobs.")

    st.write("**Job titles among your recommendations:**")

    for role in top_roles.index:
        st.write(f"• {role}")
