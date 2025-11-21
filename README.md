# DSA4263 Project Twitter Bot Detection & Analysis
Data Science solution for detection of Twitter Bots

## **Project Overview**

This project explores behavioural, linguistic, and profile-based features of Twitter accounts to distinguish genuine human users from automated bot accounts. With bots now exceeding 50% of global web traffic and a significant portion classified as malicious, detecting automated activity has become crucial for online safety, platform integrity, and fraud prevention.

Insights from this project can support:

* Social media platforms in identifying suspicious accounts
* E-commerce and advertising systems in mitigating bot-driven fraud
* Researchers studying online manipulation and inauthentic behaviour


## Objectives

The main goals of this project are to:

1. **Explore** behavioural and linguistic patterns associated with human vs bot accounts
2. **Engineer new features** that improve classification performance
3. **Build machine learning models** to detect bots
4. **Identify fraud-related patterns** that could guide real-world bot-detection strategies

## Repository Structure

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks providing a reproducible record of
│                         data preparation, exploratory analysis, model construction,
│                         and results.
│
└── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                         generated with `pip freeze > requirements.txt`

```


---

## **Getting started**

1. Ensure you have Python (Ver 3.12.3) and Jupyter notebook installed on your system.

2. Clone the repository:

```bash
git clone https://github.com/celneo7/DSA4263-Project-Twitter-Bots.git
cd DSA4263-Project-Twitter-Bots
```

3. Install Python dependencies in a virtual environment:

For MacOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Windows
```bash
python3 -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```
## Data
The dataset is not stored in this repository due to its large size.
All notebooks will automatically download, preprocess, and save the data
to the appropriate folders for full reproducibility.

Datasets (raw, interim, preprocessed) can also be accessed directly via:
[Google Drive](https://drive.google.com/drive/folders/1td_xBh4Ssk-_aVDtupbQ986e_wg0Royf)

## Usage

Open and run Jupyter notebooks in following order:

```
notebooks/preprocessing.ipynb
notebooks/data_exploration.ipynb
notebooks/hypothesis.ipynb
notebooks/Models.ipynb
```

## Data Dictionary 

| Column Name                   | Type            | Description |
|------------------------------|-----------------|-------------|
| **default_profile**          | Boolean         | Indicates whether the account uses Twitter’s default profile settings (True = default, False = customised). |
| **default_profile_image**    | Boolean         | Indicates whether the user is using the default profile image (often associated with low-effort or bot accounts). |
| **description**              | String          | Original profile description (bio) provided by the user, before cleaning or translation. |
| **favourites_count**         | Integer         | Total number of tweets the account has liked. |
| **followers_count**          | Integer         | Number of followers the account has. |
| **friends_count**            | Integer         | Number of accounts the user is following. |
| **geo_enabled**              | Boolean         | Indicates whether the account has enabled geographic tagging for tweets. |
| **id**                       | Integer         | Unique numeric identifier for the Twitter account. |
| **lang**                     | String          | Twitter’s original language field inferred from Tweets or UI settings; often empty or unreliable for bios. |
| **verified**                 | Boolean         | Indicates whether the account is verified by Twitter (badge status). |
| **average_tweets_per_day**   | Float           | Average number of tweets posted per day since account creation. |
| **account_age_days**         | Integer         | Number of days since the account was created. |
| **account_type**             | Categorical     | Target variable indicating whether the account is labelled as human or bot. |
| **word_count**               | Integer         | Number of words in the cleaned profile description. |
| **mean_word_length**         | Float           | Average word length in the description (proxy for writing complexity). |
| **hashtag_count**            | Integer         | Number of hashtags (#) present in the profile description. |
| **handle_count**             | Integer         | Number of user mentions (@username) present in the description. |
| **url_count**                | Integer         | Number of URLs present in the description. |
| **description_language**     | String          | Language of the profile description as predicted by the FastText language identification model. |
| **description_en**           | String          | English-translated version of the description using Helsinki-NLP models, or an empty string for unsupported languages. |
| **description_en_embeddings**| List (384-d)    | 384-dimensional semantic embedding of `description_en` generated using SentenceTransformer (MiniLM-L6-v2). |
| **log_followers_friends_ratio** | Float      | Log-transformed ratio of followers to friends, capturing relational asymmetry while preventing division-by-zero issues. |
| **cluster_id**               | Integer         | Topic cluster assigned to each description using BERTopic; represents semantic grouping of bios. |
