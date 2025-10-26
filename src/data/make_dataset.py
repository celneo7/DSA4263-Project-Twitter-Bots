from dotenv import load_dotenv
import fasttext
from huggingface_hub import hf_hub_download
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import os
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import pipeline

def safe_word_count(desc):
    if pd.isna(desc) or not isinstance(desc, str):
        return 0
    return len(desc.split())


def safe_mean_word_length(desc):
    if pd.isna(desc) or not isinstance(desc, str):
        return 0
    words = desc.split()
    if len(words) == 0:
        return 0
    return np.mean([len(word) for word in words])


def safe_mean_sent_length(desc):
    if pd.isna(desc) or not isinstance(desc, str):
        return 0
    sents = sent_tokenize(desc)
    if len(sents) == 0:
        return 0
    return np.mean([len(sent) for sent in sents])


def count_hashtags(desc):
    if isinstance(desc, str):
        return desc.count("#")
    return 0


def count_handles(desc):
    if isinstance(desc, str):
        return desc.count("@")
    return 0


def count_urls(desc):
    pattern = r"htt[s?://\S+]"
    if isinstance(desc, str):
        matches = re.findall(pattern, desc)
        return len(matches)
    return 0


def detect_language(df):
    """Predict language used in `description` using FastText model."""

    def clean_text(text):
        """Remove emoji and links from text to make prediction faster"""
        if not isinstance(text, str):
            return ""
        
        # Emoji & link cleaner
        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"
                u"\U0001F300-\U0001F5FF"
                u"\U0001F680-\U0001F6FF"
                u"\U0001F1E0-\U0001F1FF"
                "]+", flags=re.UNICODE)
        
        text = re.sub(r"http\S+|www\S+|@\S+", "", text)
        text = emoji_pattern.sub(r"", text)
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()


    # load fastText model
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model = fasttext.load_model(model_path)
    
    def detect_text_language(text):
        text = clean_text(text)
        if not text:
            return "unknown"
        labels, probs = model.predict(text)
        probs = np.asarray(probs)  # fix for NumPy 2.x
        return labels[0].replace("__label__", "")

    df["description_language"] = df["description"].apply(detect_text_language)

    return df
    

def translate_language(
        df,
        translation_model_map={
        "yue_Hant": pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en"),
        "kor_Hang": pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en"),
        "spa_Latn": pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
    },
        chunk_size=1000):
    
        df["description_en"] = ""

        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]

            for i, row in chunk.iterrows():
                lang = row["description_language"]
                text = row["description"]

                if lang == "eng_Latn":
                    translated = text
                elif lang == "unknown" or lang not in translation_model_map:
                    translated = ""
                else:
                    translator = translation_model_map[lang]
                    translated = translator(text)

                df.at[i, "description_en"] = translated

        return df
    

def custom_preprocessor(text):
    """
    Remove translated pattern from descriptions that were translated to english.
    Normalize  by replacing handles and URLs with placeholders, lowercase and lemmatize.
    """
    translated_pattern = r"'translation_text': '([^']*)'"
    match = re.search(translated_pattern, text)
    if match:
        text = match[1]

    # replace URLs with placeholder
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # remove newlines, tabs, and extra spaces
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    text = text.lower().strip()

    stop_words = set(nltk.corpus.stopwords.words("english"))
    filtered = [word for word in text.split() if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
    return " ".join(lemmatized)


def embed_descriptions(description_df, hf_api):
    model = SentenceTransformer('all-MiniLM-L6-v2', token=hf_api)
    embeddings_lst = []

    for i, row in description_df.iterrows():
        if row["description_en"]:
            embeddings = model.encode(row["description_en"])
        else:
            embeddings = np.zeros(384, dtype=float)  # default dim=384
        embeddings_lst.append(embeddings)

        if not i % 1000:
            print(f"progress: {i}")

    description_df["description_en_embeddings"] = embeddings_lst
    
    return description_df


def main(df):
    load_dotenv()
    df["length"] = df["description"].str.len().fillna(0)
    df["word_count"] = df["description"].apply(safe_word_count)
    df["mean_word_length"] = df["description"].apply(safe_mean_word_length)
    df["mean_sent_length"] = df["description"].apply(safe_mean_sent_length)
    df["hashtag_count"] = df["description"].apply(count_hashtags)
    df["handle_count"] = df["description"].apply(count_handles)

    df = detect_language(df)
    df = translate_language(df)
    df = embed_descriptions(df, os.getenv("HF_API_KEY") )
    df.to_csv("twitter_human_bots_description_engineered.csv")

if __name__=="__main__":
    filename = os.path.join(os.getcwd(), "twitter_human_bots_dataset.csv")
    main(pd.read_csv(filename, index_col=0))