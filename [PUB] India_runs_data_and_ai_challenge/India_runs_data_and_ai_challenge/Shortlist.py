import pandas as pd
from Signal_Integration_Multi_Attribute_Scoring import integrate_signals

def main():
    print("--- Executing AI Recruiter Ranking System ---")
    df_shortlist = integrate_signals()
    
    print("\n=== Final Expertly Ranked Shortlist ===")
    for index, row in df_shortlist.head(10).iterrows(): # Show top 10
        # Dynamically fetch name/id depending on dataset headers
        name = row.get('name', row.get('id', f"Candidate_{index}"))
        print(f"Rank {index+1}: {name} | Score: {row['final_rank_score']:.4f} (Semantic Fit: {row['semantic_score']:.4f})")
    
    # Build your submission output to match your sample_submission.csv format
    # Usually requires candidate_id and final rank or score
    submission = pd.DataFrame({
        'candidate_id': df_shortlist.get('id', df_shortlist.index),
        'rank': range(1, len(df_shortlist) + 1),
        'score': df_shortlist['final_rank_score']
    })
    
    submission.to_csv("final_submission.csv", index=False)
    print("\n[SUCCESS] final_submission.csv successfully generated for submission!")

if __name__ == "__main__":
    main()