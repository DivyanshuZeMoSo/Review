import pandas as pd

def load_movie_data(filepath):
    df = pd.read_excel(filepath)
    df.dropna(subset=['Title', 'Description', 'Genre', 'Director'], inplace=True)
    return df
