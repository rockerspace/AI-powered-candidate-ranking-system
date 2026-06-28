import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Compute_Contextual_Semantic_Relevance import compute_semantic_scores  # Note: ensure filename matches perfectly

def integrate_signals():
    # Get dataframe with semantic scores
    df = compute_semantic_scores()
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Dynamically find available numeric signal keys from your jsonl
    # Adjust these strings based on what keys are listed in your candidate_schema.json
    signal_cols = [col for col in ['years_experience', 'profile_completion_pct', 'engagement_score'] if col in df.columns]
    
    if signal_cols:
        scaled_signals = scaler.fit_transform(df[signal_cols])
        for i, col in enumerate(signal_cols):
            df[f'norm_{col}'] = scaled_signals[:, i]
            
        # Calculate final weighted score
        # Give 60% weight to semantic relevance, split remaining among signals
        df['final_rank_score'] = df['semantic_score'] * 0.60
        weight_per_signal = 0.40 / len(signal_cols)
        for col in signal_cols:
            df['final_rank_score'] += df[f'norm_{col}'] * weight_per_signal
    else:
        # Fallback if no specific behavioral metrics exist in the data schema
        df['final_rank_score'] = df['semantic_score']
        
    return df.sort_values(by='final_rank_score', ascending=False).reset_index(drop=True)