
import os
import re
import fitz  # PyMuPDF
import docx
import pandas as pd
from datetime import datetime
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the folder path
resume_folder = r"C:\Users\KAPSHARM\Downloads\OneDrive_2025-10-13\SAP FI 13th Jun 2025"

# SAP keywords
sap_keywords = [
    "SAP FI", "SAP FICO", "SAP CO", "SAP S/4HANA", "SAP Finance", "SAP MM", "SAP SD", "SAP PP", "SAP QM",
    "SAP PM", "SAP PS", "SAP HCM", "SAP HR", "SAP SuccessFactors", "SAP BW", "SAP BI", "SAP BO", "SAP BPC",
    "SAP ABAP", "SAP BASIS", "SAP Fiori", "SAP UI5", "SAP CRM", "SAP SCM", "SAP APO", "SAP Ariba", "SAP IBP",
    "SAP Hybris", "SAP PI", "SAP PO", "SAP Solution Manager", "SAP NetWeaver", "SAP Leonardo", "SAP Cloud Platform"
]

cert_keywords = ["SAP Certified", "Certification", "Certified", "ACCA", "CPA", "CMA", "CFA", "MBA", "CA"]

email_pattern = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
years_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:years|year|yrs|yr)", re.IGNORECASE)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception:
        pass
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception:
        pass
    return text

def extract_email_and_context(text):
    match = email_pattern.search(text)
    if match:
        email = match.group()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if email in line:
                start = max(0, i - 5)
                end = min(len(lines), i + 5)
                context = "\n".join(lines[start:end])
                return email, context
    return "", ""

def extract_candidate_name(text, email, file_name):
    lines = text.splitlines()
    for line in lines[:30]:
        line = line.strip()
        if line and not any(x in line.lower() for x in ["summary", "objective", "experience", "skills", "resume"]):
            if not any(char.isdigit() for char in line) and len(line.split()) <= 4:
                return line
    if email:
        local = email.split("@")[0]
        name = re.sub(r"[._\-]", " ", local)
        return name.title()
    name = os.path.splitext(file_name)[0]
    name = re.sub(r"[._\-]", " ", name)
    return name.title()

def extract_sap_skills(text):
    found = []
    for keyword in sap_keywords:
        if keyword.lower() in text.lower():
            found.append(keyword)
    return ", ".join(found) if found else "Not Found"

def extract_certifications(text):
    found = set()
    for line in text.splitlines()[:200]:
        for keyword in cert_keywords:
            if keyword.lower() in line.lower():
                found.add(keyword)
    return ", ".join(found) if found else "Not Found"

def extract_experience_lines(text):
    sap_line = ""
    total_line = ""
    total_years = ""

    # First pass: regex and keyword-based
    lines = text.splitlines()
    for line in lines:
        if not sap_line and "sap" in line.lower() and "experience" in line.lower():
            sap_line = line.strip()
        if not total_line and any(x in line.lower() for x in ["total experience", "overall experience", "years of experience"]):
            total_line = line.strip()
        if not total_years:
            match = re.search(r"(\\d+(?:\\.\\d+)?)\\s*(?:years|year|yrs|yr)", line, re.IGNORECASE)
            if match:
                total_years = float(match.group(1))

    # NLP fallback
    if not sap_line or not total_line or not total_years:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for sent in doc.sents:
            if not sap_line and "sap" in sent.text.lower() and "experience" in sent.text.lower():
                sap_line = sent.text.strip()
            if not total_line and any(x in sent.text.lower() for x in ["total experience", "overall experience", "years of experience"]):
                total_line = sent.text.strip()
            if not total_years:
                match = re.search(r"(\\d+(?:\\.\\d+)?)\\s*(?:years|year|yrs|yr)", sent.text, re.IGNORECASE)
                if match:
                    total_years = float(match.group(1))

    # Final fallback if SAP skills are found
    if sap_line == "" or total_line == "" or total_years == "":
        sap_line = sap_line or "SAP experience mentioned but not clearly stated"
        total_line = total_line or "Total experience mentioned but not clearly stated"
        total_years = total_years or 0.0

    return sap_line, total_line, total_years
    sap_line = ""
    total_line = ""
    total_years = ""

    lines = text.splitlines()
    for line in lines:
        if not sap_line and "sap" in line.lower() and "experience" in line.lower():
            sap_line = line.strip()
        if not total_line and any(x in line.lower() for x in ["total experience", "overall experience", "years of experience"]):
            total_line = line.strip()
        if not total_years:
            match = years_pattern.search(line)
            if match:
                total_years = float(match.group(1))

    # NLP fallback
    if not sap_line or not total_line or not total_years:
        doc = nlp(text)
        for sent in doc.sents:
            if not sap_line and "sap" in sent.text.lower() and "experience" in sent.text.lower():
                sap_line = sent.text.strip()
            if not total_line and any(x in sent.text.lower() for x in ["total experience", "overall experience", "years of experience"]):
                total_line = sent.text.strip()
            if not total_years:
                match = years_pattern.search(sent.text)
                if match:
                    total_years = float(match.group(1))

    return sap_line if sap_line else "SAP experience mentioned but not clearly stated", \
           total_line if total_line else "Total experience mentioned but not clearly stated", \
           total_years if total_years else 0.0

# Process resumes
data = []
sr_no = 1
for file_name in os.listdir(resume_folder):
    file_path = os.path.join(resume_folder, file_name)
    if not os.path.isfile(file_path):
        continue
    if not file_name.lower().endswith((".pdf", ".docx")):
        continue

    text = extract_text_from_pdf(file_path) if file_name.lower().endswith(".pdf") else extract_text_from_docx(file_path)
    if not text.strip():
        continue

    email, context = extract_email_and_context(text)
    fast_text = context if context else "\n".join(text.splitlines()[:60])
    candidate_name = extract_candidate_name(fast_text, email, file_name)
    sap_skills = extract_sap_skills(text)
    certifications = extract_certifications(fast_text)

    sap_exp_line, total_exp_line, total_exp_years = extract_experience_lines(text)

    data.append([
        sr_no,
        file_name,
        candidate_name,
        email if email else "Not Found",
        sap_skills,
        sap_exp_line,
        total_exp_line,
        total_exp_years,
        certifications
    ])
    sr_no += 1

# Save to Excel
columns = [
    "Sr No", "File Name", "Candidate Name", "Candidate Email",
    "Relevant SAP Skills", "SAP Experience (line)",
    "Total Experience (line)", "Total Exp (years)", "Certifications"
]
df = pd.DataFrame(data, columns=columns)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(resume_folder, f"Resume_Simplified_Extract_{timestamp}.xlsx")
df.to_excel(output_file, index=False)

print(f"Resume data extracted and saved to: {output_file}")