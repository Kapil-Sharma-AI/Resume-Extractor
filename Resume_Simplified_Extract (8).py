import os
import re
import docx2txt
import PyPDF2
import pandas as pd

# -------------------------------
# CONFIGURATION
# -------------------------------
FOLDER_PATH = r"c:\users\kapsharm\downloads\onedrive_2025-10-13\SAP FI 13th jun 2025"
OUTPUT_FILE = "Resume_Extract.xlsx"
ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.doc']

# -------------------------------
# TEXT EXTRACTION FUNCTION
# -------------------------------
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        elif ext in ['.docx', '.doc']:
            text = docx2txt.process(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

# -------------------------------
# TOTAL EXPERIENCE EXTRACTION
# -------------------------------
def extract_total_exp(text, file_name):
    patterns = [
        r'Total\s*Experience[:\s]*([\d\.\+]+)\s*(years|yrs)?',
        r'([\d\.\+]+)\s*(years|yrs)\s*of\s*experience',
        r'Experience[:\s]*([\d\.\+]+)\s*(years|yrs)?'
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 0 < val < 60:  # realistic range
                    return val
            except:
                continue
    # fallback: check file name
    match = re.search(r'(\d{1,2})\s*(yrs|years)', file_name, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 0 < val < 60:
            return val
    return ""  # Not found

# -------------------------------
# SAP RELEVANT EXPERIENCE
# -------------------------------
def extract_sap_exp(text):
    sap_years = ""
    sap_desc = ""
    matches = re.findall(r'(SAP\s+\w+).*?(\d+(\.\d+)?)\s*(years|yrs)', text, re.IGNORECASE)
    if matches:
        sap_years_list = []
        desc_list = []
        for m in matches:
            module, years = m[0], float(m[1])
            if years > 0 and years < 60:
                sap_years_list.append(years)
                desc_list.append(f"{module} ({years} yrs)")
        if sap_years_list:
            sap_years = round(sum(sap_years_list), 1)
            sap_desc = "; ".join(desc_list)
    elif re.search(r'SAP', text, re.IGNORECASE):
        sentences = re.split(r'[.\n]', text)
        sap_sentences = [s.strip() for s in sentences if 'SAP' in s]
        if sap_sentences:
            sap_desc = "; ".join(sap_sentences[:3])
            sap_years = ""
    return sap_years, sap_desc

# -------------------------------
# TOTAL EXPERIENCE INFO (BRIEF)
# -------------------------------
def extract_total_info(text):
    sentences = re.split(r'[.\n]', text)
    info_list = [s.strip() for s in sentences if re.search(r'experience|worked', s, re.IGNORECASE)]
    return "; ".join(info_list[:3])

# -------------------------------
# PROCESS FOLDER
# -------------------------------
def process_folder(folder_path):
    data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
                path = os.path.join(root, file)
                text = extract_text(path)
                total_exp = extract_total_exp(text, file)
                sap_years, sap_desc = extract_sap_exp(text)
                total_info = extract_total_info(text)
                data.append({
                    "File Name": file,
                    "SAP Relevant Exp (yrs)": sap_years,
                    "SAP Relevant Description": sap_desc,
                    "SAP Total Exp Info": total_info,
                    "Total Years Exp": total_exp
                })
    return pd.DataFrame(data)

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    df = process_folder(FOLDER_PATH)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Extraction complete! Data saved to {OUTPUT_FILE}")
