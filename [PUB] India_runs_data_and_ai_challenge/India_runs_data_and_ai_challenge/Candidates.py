import json
import docx2txt
import pandas as pd

def load_job_description(file_path="job_description.docx"):
    # Extracts text directly from the docx file
    text = docx2txt.process(file_path)
    return text

def load_candidates(file_path="candidates.jsonl"):
    # Reads JSON Lines format into a Pandas DataFrame
    candidates = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return pd.DataFrame(candidates)

if __name__ == "__main__":
    jd = load_job_description()
    df = load_candidates()
    print(f"Loaded Job Description Length: {len(jd)} characters.")
    print(f"Loaded {len(df)} candidates from dataset.")