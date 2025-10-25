import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# --- DATA PATH ---
DATA_PATH_CELEBRITY = "../datasets/Celebrity/df_celebrity.csv"
DATA_PATH_CIDII = "../datasets/CIDII/df_cidii.csv"
DATA_PATH_FAKES = "../datasets/FaKES/df_fakes.csv"
DATA_PATH_FAKEVSATIRE = "../datasets/FakeVsSatire/df_fakevssatire.csv"
DATA_PATH_HORNE = "../datasets/Horne/df_horne.csv"
DATA_PATH_INFODEMIC = "../datasets/Infodemic/df_infodemic.csv"
DATA_PATH_ISOT = "../datasets/ISOT/df_isot.csv"
DATA_PATH_KAGGLE_CLEMENT = "../datasets/Kaggle_clement/df_kaggle_clement.csv"
DATA_PATH_KAGGLE_MEG = "../datasets/Kaggle_meg/df_kaggle_meg.csv"
DATA_PATH_LIAR_PLUS = "../datasets/LIAR_PLUS/df_liarplus.csv"
DATA_PATH_POLITIFACT = "../datasets/Politifact/df_politifact.csv"
DATA_PATH_UNIPI_NDF = "../datasets/Unipi_NDF/df_ndf.csv"

# --- TOPICS ---
TOPIC_CELEBRITY = "gossip"
TOPIC_CIDII = "islam"
TOPIC_FAKES = "syria"
TOPIC_FAKEVSATIRE = "politics"
TOPIC_HORNE = "politics"
TOPIC_INFODEMIC = "covid"
TOPIC_ISOT = "politics"
TOPIC_KAGGLE_CLEMENT = "politics"
TOPIC_KAGGLE_MEG = "general"
TOPIC_LIAR_PLUS = "politics"
TOPIC_POLITIFACT = "politics"
TOPIC_UNIPI_NDF = "notredame"



# ---------------------------
# Count labels (real vs fake)
# ---------------------------
def count_labels(df):
    """
    Count total number of samples and number of samples per label in the given dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'labels' column.
    """

    # print total length
    print("Total number of rows:", len(df))
    # print number of 0 labels
    print("Number of Real News:", (df['labels'] == 0).sum())
    # print number of 1 labels
    print("Number of Fake News:", (df['labels'] == 1).sum())



# ------------------------------
# Data loading and preprocessing
# ------------------------------
def data_loading():
    """
    Preprocess and load all datasets into a dictionary of DataFrames.

    Returns:
        dict: A dictionary where keys are dataset names and values are preprocessed DataFrames.
    """

    # --- CELEBRITY ---
    dfCelebrity = pd.read_csv(DATA_PATH_CELEBRITY, sep="\t", encoding="utf-8")
    # texts and labels
    dfCelebrity = dfCelebrity[['texts', 'labels']] # keep only relevant columns
    dfCelebrity = dfCelebrity.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfCelebrity['date'] = pd.NaT # add empty date column to match other datasets
    # topic
    dfCelebrity['topic'] = TOPIC_CELEBRITY

    # --- CIDII ---
    dfCidii = pd.read_csv(DATA_PATH_CIDII, sep="\t", encoding="utf-8")
    # texts and labels
    dfCidii = dfCidii[['texts', 'labels']] # keep only relevant columns
    dfCidii = dfCidii.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfCidii['date'] = pd.NaT # add empty date column to match other datasets
    # topic
    dfCidii['topic'] = TOPIC_CIDII

    # --- FAKES ---
    dfFakes = pd.read_csv(DATA_PATH_FAKES, sep="\t", encoding="utf-8")
    # texts and labels
    dfFakes = dfFakes[['texts', 'labels']] # keep only relevant columns
    dfFakes = dfFakes.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfFakesDates = pd.read_csv("../datasets/FaKES/date_df_fakes.csv", encoding="ISO-8859-1") # load dataset with dates
    dfFakesDates = dfFakesDates.rename(columns={'article_content': 'texts'}) # rename column to match
    dfFakes = pd.merge(dfFakes, dfFakesDates[['texts', 'date']], on="texts", how="left") # merge to add dates
    dfFakes['date'] = pd.to_datetime(dfFakes['date'], errors='coerce') # convert date column to datetime, coerce errors to NaT
    # topic
    dfFakes['topic'] = TOPIC_FAKES

    # --- FAKEVSATIRE ---
    dfFakeVsSatire = pd.read_csv(DATA_PATH_FAKEVSATIRE, sep="\t", encoding="utf-8")
    # texts and labels
    dfFakeVsSatire = dfFakeVsSatire[['texts', 'labels']] # keep only relevant columns
    dfFakeVsSatire = dfFakeVsSatire.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfFakeVsSatire['date'] = pd.NaT # add empty date column to match other datasets
    # topic
    dfFakeVsSatire['topic'] = TOPIC_FAKEVSATIRE
    
    # --- HORNE ---
    dfHorne = pd.read_csv(DATA_PATH_HORNE, sep="\t", encoding="utf-8")
    # texts and labels
    dfHorne = dfHorne[['texts', 'labels']] # keep only relevant columns
    dfHorne = dfHorne.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfHorne['date'] = pd.to_datetime("2016-01-01") # add static date column to match other datasets
    # topic
    dfHorne['topic'] = TOPIC_HORNE

    # --- INFODEMIC ---
    dfInfodemic = pd.read_csv(DATA_PATH_INFODEMIC, sep="\t", encoding="utf-8")
    # texts and labels
    dfInfodemic = dfInfodemic[['texts', 'labels']] # keep only relevant columns
    dfInfodemic = dfInfodemic.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfInfodemic['date'] = pd.to_datetime("2020-01-01") # add static date column to match other datasets
    # topic
    dfInfodemic['topic'] = TOPIC_INFODEMIC

    # --- ISOT ---
    dfIsot = pd.read_csv(DATA_PATH_ISOT, sep="\t", encoding="utf-8")
    # texts and labels
    dfIsot = dfIsot[['texts', 'labels']] # keep only relevant columns
    dfIsot = dfIsot.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfIsot['date'] = pd.to_datetime("2016-01-01") # add static date column to match other datasets
    # topic
    dfIsot['topic'] = TOPIC_ISOT

    # --- KAGGLE_CLEMENT ---
    dfKaggleClement = pd.read_csv(DATA_PATH_KAGGLE_CLEMENT, encoding="utf-8")
    # texts and labels
    dfKaggleClement['texts'] = dfKaggleClement['title'].astype(str) + " " + dfKaggleClement['text'].astype(str) # merge title and text
    dfKaggleClement = dfKaggleClement[['texts', 'labels', 'date']] # keep only relevant columns
    dfKaggleClement = dfKaggleClement.drop_duplicates(subset=['texts', 'labels'])
    dfKaggleClement = dfKaggleClement.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfKaggleClement['date'] = pd.to_datetime(dfKaggleClement['date'], errors='coerce') # convert date column to datetime, coerce errors to NaT
    # topic
    dfKaggleClement['topic'] = TOPIC_KAGGLE_CLEMENT

    # --- KAGGLE_MEG ---
    dfKaggleMeg = pd.read_csv(DATA_PATH_KAGGLE_MEG, encoding="utf-8")
    # texts and labels
    dfKaggleMeg['texts'] = dfKaggleMeg['title'].astype(str) + " " + dfKaggleMeg['text'].astype(str) # merge title and text
    dfKaggleMeg['labels'] = dfKaggleMeg['spam_score'].apply(lambda x: 1 if x > 0.5 else 0) # create binary labels based on spam_score
    dfKaggleMeg = dfKaggleMeg[['texts', 'labels', 'published']] # keep only relevant columns
    dfKaggleMeg = dfKaggleMeg.drop_duplicates(subset=['texts', 'labels'])
    dfKaggleMeg = dfKaggleMeg.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfKaggleMeg = dfKaggleMeg.rename(columns={'published': 'date'}) # rename published to date
    dfKaggleMeg['date'] = pd.to_datetime(dfKaggleMeg['date'], errors='coerce') # convert date column to datetime, coerce errors to NaT
    dfKaggleMeg['date'] = dfKaggleMeg['date'].fillna(pd.to_datetime("2016-01-01")) # fill missing dates with static date to match other datasets
    # topic
    dfKaggleMeg['topic'] = TOPIC_KAGGLE_MEG

    # --- LIAR_PLUS ---
    dfLiarPlus = pd.read_csv(DATA_PATH_LIAR_PLUS, sep="\t", encoding="utf-8")
    # texts and labels
    dfLiarPlus = dfLiarPlus[['texts', 'labels']] # keep only relevant columns
    dfLiarPlus = dfLiarPlus.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfLiarPlus['date'] = pd.NaT # add empty date column to match other datasets
    # topic
    dfLiarPlus['topic'] = TOPIC_LIAR_PLUS

    # --- POLITIFACT ---
    dfPolitifact = pd.read_csv(DATA_PATH_POLITIFACT, sep="\t", encoding="utf-8")
    # texts and labels
    dfPolitifact = dfPolitifact[['texts', 'labels']] # keep only relevant columns
    dfPolitifact = dfPolitifact.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfPolitifact['date'] = pd.NaT # add empty date column to match other datasets
    # topic
    dfPolitifact['topic'] = TOPIC_POLITIFACT

    # --- UNIPI_NDF ---
    dfNDF = pd.read_csv(DATA_PATH_UNIPI_NDF, sep="\t", encoding="utf-8")
    # texts and labels
    dfNDF = dfNDF[['texts', 'labels']] # keep only relevant columns
    dfNDF = dfNDF.dropna(subset=['texts', 'labels']) # remove rows where texts OR labels are NaN
    # date
    dfNDF['date'] = pd.to_datetime("2019-01-01") # add static date column to match other datasets
    # topic
    dfNDF['topic'] = TOPIC_UNIPI_NDF

    return {
        "Celebrity": dfCelebrity,
        "CIDII": dfCidii,
        "FaKES": dfFakes,
        "FakeVsSatire": dfFakeVsSatire,
        "Horne": dfHorne,
        "Infodemic": dfInfodemic,
        "ISOT": dfIsot,
        "Kaggle_clement": dfKaggleClement,
        "Kaggle_meg": dfKaggleMeg,
        "LIAR_PLUS": dfLiarPlus,
        "Politifact": dfPolitifact,
        "Unipi_NDF": dfNDF
    }



# ----------------------------
# Data by topic/date functions
# ----------------------------
def data_by_topic():
    """
    Organize datasets by topic

    Returns:
        dict: A dictionary where keys are topics and values are DataFrames containing all samples for that
    """

    df_dict = data_loading()

    combined_df = pd.concat(df_dict.values(), ignore_index=True).reset_index(drop=True) # combine all datasets

    combined_df = combined_df.dropna(subset=['topic']) # drop rows where topic is NaN

    grouped = combined_df.groupby('topic', sort=False) # group by topic without sorting

    df_dict_by_topic = {topic: group for topic, group in grouped} # create dict from groupby

    # order topics by frequency
    topic_order = combined_df['topic'].value_counts().index
    df_dict_by_topic = {t: df_dict_by_topic[t] for t in topic_order}

    return df_dict_by_topic


def data_by_date():
    """
    Organize datasets by date

    Returns:
        dict: A dictionary where keys are years and values are DataFrames containing all samples for that
    """

    df_dict = data_loading()

    combined_df = pd.concat(df_dict.values(), ignore_index=True) # combine all datasets

    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce') # assure date format
    combined_df = combined_df.dropna(subset=['date']) # drop rows where date is NaT
    combined_df = combined_df.sort_values(by='date').reset_index(drop=True) # sort by date

    combined_df['year'] = combined_df['date'].dt.year # extract year for grouping

    # Considering only the year, create a dict of dataframes per year
    grouped = combined_df.groupby('year', sort=False)
    df_dict_by_date = {year: group for year, group in grouped}

    # merge "2011", "2012" and "2013" into "2011-2013", but "2011-2013" should come first in order
    years_to_merge = [2011, 2012, 2013]
    df_dict_by_date['2011-2013'] = pd.concat([df_dict_by_date[year] for year in years_to_merge], ignore_index=True)
    for year in years_to_merge:
        del df_dict_by_date[year]

    # order years
    df_dict_by_date = dict(sorted(df_dict_by_date.items(), key=lambda x: (int(x[0].split('-')[0]) if isinstance(x[0], str) and '-' in x[0] else x[0])))

    return df_dict_by_date



# --------------------------
# Dataset splitting function
# --------------------------

def split_dataset(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        df (pd.DataFrame): DataFrame containing 'texts' and 'labels'
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary with keys 'train', 'val', 'test' and values as tuples of (X, y)
    """

    X = df['texts'].astype(str)
    y = df['labels']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }



# --------------------------------
# Training and evaluation function
# --------------------------------

def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    """
    Trains the model and evaluates it on the validation set using weighted F1-score.

    Args:
        model (Pipeline): The scikit-learn Pipeline model to train.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.

    Returns:
        float: Weighted F1-score on the validation set.
    """

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="weighted") # weighted F1-score: average for label imbalance
    return f1