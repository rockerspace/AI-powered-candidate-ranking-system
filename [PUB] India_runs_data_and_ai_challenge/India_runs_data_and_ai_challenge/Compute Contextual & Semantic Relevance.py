from sentence_transformers import SentenceTransformer, util
from Candidates import load_job_description, load_candidates

def compute_semantic_scores():
    # 1. Load data
    jd_text = load_job_description()
    df_candidates = load_candidates()
    
    # 2. Load semantic model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Handle the text column (assuming 'resume_text' or 'experience_description' exists)
    # Check your candidate_schema.json to ensure the column name matches!
    text_column = 'resume_text' if 'resume_text' in df_candidates.columns else df_candidates.columns[2] 
    
    print("Computing semantic embeddings...")
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embeddings = model.encode(df_candidates[text_column].tolist(), convert_to_tensor=True)
    
    # 4. Calculate similarities
    cosine_scores = util.cos_sim(jd_embedding, resume_embeddings)[0].tolist()
    df_candidates['semantic_score'] = cosine_scores
    
    return df_candidates

if __name__ == "__main__":
    df = compute_semantic_scores()
    print(df[['semantic_score']].head())