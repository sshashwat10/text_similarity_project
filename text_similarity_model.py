# text_similarity_model.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(text1, text2):
    # Encode texts
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    # Compute cosine similarity
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return round(score.item(), 4)

def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['similarity_score'] = df.apply(lambda row: compute_similarity(row['text1'], row['text2']), axis=1)
    return df

if __name__ == "__main__":
    csv_path = r"C:\Users\sshas\Downloads\Compressed\DataNeuron_DataScience_Task1\DataNeuron_Text_Similarity.csv"
    result_df = process_csv(csv_path)
    result_df.to_csv("similarity_scored_output.csv", index=False)
    print("Similarity scores saved to similarity_scored_output.csv")
