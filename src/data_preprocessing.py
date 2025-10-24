import pandas as pd

# --- DATA PATH ---
DATA_PATH_CELEBRITY = "datasets/Celebrity/df_celebrity.csv"
DATA_PATH_CIDII = "datasets/CIDII/df_cidii.csv"
DATA_PATH_FAKES = "datasets/FaKES/df_fakes.csv"
DATA_PATH_FAKEVSATIRE = "datasets/FakeVsSatire/df_fakevssatire.csv"
DATA_PATH_HORNE = "datasets/Horne/df_horne_all.csv"
DATA_PATH_INFODEMIC = "datasets/Infodemic/df_infodemic.csv"
DATA_PATH_ISOT = "datasets/ISOT/df_isot.csv"
DATA_PATH_KAGGLE_CLEMENT = "datasets/Kaggle_clement/df_kaggle_clement_def.csv"
DATA_PATH_KAGGLE_MEG = "datasets/Kaggle_meg/fake.csv"
DATA_PATH_LIAR_PLUS = "datasets/LIAR_PLUS/df_liar_plus_complete.csv"
DATA_PATH_POLITIFACT = "datasets/Politifact/df_politifact.csv"
DATA_PATH_UNIPI_NDF = "datasets/Unipi_NDF/df_ndf.csv"

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


def data_loading():
    """
    Preprocess and load all datasets into a dictionary of DataFrames.

    Returns:
        dict: A dictionary where keys are dataset names and values are preprocessed DataFrames.
    """

    # --- CELEBRITY ---
    dfCelebrity = pd.read_csv(DATA_PATH_CELEBRITY, sep="\t", encoding="utf-8")
    # texts and labels
    dfCelebrity = dfCelebrity[["texts", "labels"]] # keep only relevant columns
    dfCelebrity = dfCelebrity.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfCelebrity["date"] = pd.NaT # add empty date column to match other datasets
    # topic
    dfCelebrity["topic"] = TOPIC_CELEBRITY

    # --- CIDII ---
    dfCidii = pd.read_csv(DATA_PATH_CIDII, sep="\t", encoding="utf-8")
    # texts and labels
    dfCidii = dfCidii[["texts", "labels"]] # keep only relevant columns
    dfCidii = dfCidii.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfCidii["date"] = pd.NaT # add empty date column to match other datasets
    # topic
    dfCidii["topic"] = TOPIC_CIDII

    # --- FAKES ---
    dfFakes = pd.read_csv(DATA_PATH_FAKES, sep="\t", encoding="utf-8")
    # texts and labels
    dfFakes = dfFakes[["texts", "labels"]] # keep only relevant columns
    dfFakes = dfFakes.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfFakesDates = pd.read_csv("datasets/FaKES/date-FA-KES-Dataset.csv", encoding="ISO-8859-1") # load dataset with dates
    dfFakesDates = dfFakesDates.rename(columns={"article_content": "texts"}) # rename column to match
    dfFakes = pd.merge(dfFakes, dfFakesDates[['texts', 'date']], on="texts", how="left") # merge to add dates
    dfFakes['date'] = pd.to_datetime(dfFakes['date'], errors='coerce') # convert date column to datetime, coerce errors to NaT
    # topic
    dfFakes["topic"] = TOPIC_FAKES

    # --- FAKEVSATIRE ---
    dfFakeVsSatire = pd.read_csv(DATA_PATH_FAKEVSATIRE, sep="\t", encoding="utf-8")
    # texts and labels
    dfFakeVsSatire = dfFakeVsSatire[["texts", "labels"]] # keep only relevant columns
    dfFakeVsSatire = dfFakeVsSatire.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfFakeVsSatire["date"] = pd.NaT # add empty date column to match other datasets
    # topic
    dfFakeVsSatire["topic"] = TOPIC_FAKEVSATIRE
    
    # --- HORNE ---
    dfHorne = pd.read_csv(DATA_PATH_HORNE, sep="\t", encoding="utf-8")
    # texts and labels
    dfHorne = dfHorne[["texts", "labels"]] # keep only relevant columns
    dfHorne = dfHorne.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfHorne["date"] = pd.to_datetime("2016-01-01") # add static date column to match other datasets
    # topic
    dfHorne["topic"] = TOPIC_HORNE

    # --- INFODEMIC ---
    dfInfodemic = pd.read_csv(DATA_PATH_INFODEMIC, sep="\t", encoding="utf-8")
    # texts and labels
    dfInfodemic = dfInfodemic[["texts", "labels"]] # keep only relevant columns
    dfInfodemic = dfInfodemic.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfInfodemic["date"] = pd.to_datetime("2020-01-01") # add static date column to match other datasets
    # topic
    dfInfodemic["topic"] = TOPIC_INFODEMIC

    # --- ISOT ---
    dfIsot = pd.read_csv(DATA_PATH_ISOT, sep="\t", encoding="utf-8")
    # texts and labels
    dfIsot = dfIsot[["texts", "labels"]] # keep only relevant columns
    dfIsot = dfIsot.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfIsot["date"] = pd.to_datetime("2016-01-01") # add static date column to match other datasets
    # topic
    dfIsot["topic"] = TOPIC_ISOT

    # --- KAGGLE_CLEMENT ---
    dfKaggleClement = pd.read_csv(DATA_PATH_KAGGLE_CLEMENT, encoding="utf-8")
    # texts and labels
    dfKaggleClement["texts"] = dfKaggleClement["title"].astype(str) + " " + dfKaggleClement["text"].astype(str) # merge title and text
    dfKaggleClement = dfKaggleClement[["texts", "labels"]] # keep only relevant columns
    dfKaggleClement = dfKaggleClement.drop_duplicates(subset=['text', 'labels'])
    dfKaggleClement = dfKaggleClement.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfKaggleClement['date'] = pd.to_datetime(dfKaggleClement['date'], errors='coerce') # convert date column to datetime, coerce errors to NaT
    # topic
    dfKaggleClement["topic"] = TOPIC_KAGGLE_CLEMENT

    # --- KAGGLE_MEG ---
    dfKaggleMeg = pd.read_csv(DATA_PATH_KAGGLE_MEG, encoding="utf-8")
    # texts and labels
    dfKaggleMeg["texts"] = dfKaggleMeg["title"].astype(str) + " " + dfKaggleMeg["text"].astype(str) # merge title and text
    dfKaggleMeg["labels"] = dfKaggleMeg["spam_score"].apply(lambda x: 1 if x > 0.5 else 0) # create binary labels based on spam_score
    dfKaggleMeg = dfKaggleMeg[["texts", "labels"]] # keep only relevant columns
    dfKaggleMeg = dfKaggleMeg.drop_duplicates(subset=['texts', 'labels'])
    dfKaggleMeg = dfKaggleMeg.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfKaggleMeg['date'] = pd.to_datetime(dfKaggleMeg['date'], errors='coerce') # convert date column to datetime, coerce errors to NaT
    dfKaggleMeg['date'] = dfKaggleMeg['date'].fillna(pd.to_datetime("2016-01-01")) # fill missing dates with static date to match other datasets
    # topic
    dfKaggleMeg["topic"] = TOPIC_KAGGLE_MEG

    # --- LIAR_PLUS ---
    dfLiarPlus = pd.read_csv(DATA_PATH_LIAR_PLUS, sep="\t", encoding="utf-8")
    # texts and labels
    dfLiarPlus = dfLiarPlus[["texts", "labels"]] # keep only relevant columns
    dfLiarPlus = dfLiarPlus.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfLiarPlus["date"] = pd.NaT # add empty date column to match other datasets
    # topic
    dfLiarPlus["topic"] = TOPIC_LIAR_PLUS

    # --- POLITIFACT ---
    dfPolitifact = pd.read_csv(DATA_PATH_POLITIFACT, sep="\t", encoding="utf-8")
    # texts and labels
    dfPolitifact = dfPolitifact[["texts", "labels"]] # keep only relevant columns
    dfPolitifact = dfPolitifact.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfPolitifact["date"] = pd.NaT # add empty date column to match other datasets
    # topic
    dfPolitifact["topic"] = TOPIC_POLITIFACT

    # --- UNIPI_NDF ---
    dfNDF = pd.read_csv(DATA_PATH_UNIPI_NDF, sep="\t", encoding="utf-8")
    # texts and labels
    dfNDF = dfNDF[["texts", "labels"]] # keep only relevant columns
    dfNDF = dfNDF.dropna(subset=["texts", "labels"]) # remove rows where texts OR labels are NaN
    # date
    dfNDF["date"] = pd.to_datetime("2019-01-01") # add static date column to match other datasets
    # topic
    dfNDF["topic"] = TOPIC_UNIPI_NDF

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



def data_by_topic(dataframes_dict):
    # --------------------------
    # Merge dei dataset per topic
    # --------------------------

    # Raggruppa i dataset per topic
    from collections import defaultdict
    topic_groups = defaultdict(list)

    for name, df in dataframes_dict.items():
        topic = df["topic"].iloc[0]  # Assumi che il topic sia lo stesso per tutto il df
        topic_groups[topic].append(df)

    # Crea un dataframe unito per ogni topic
    df_by_topic = {topic: pd.concat(dfs, ignore_index=True) for topic, dfs in topic_groups.items()}

    # Ordina i topic in base alla numerosità (più frequente → meno frequente)
    topic_counts = {topic: len(df) for topic, df in df_by_topic.items()}
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

    # Ora hai una lista ordinata: [(topic, num_articoli), ...]
    print("Topic ordinati per numero di articoli:")
    for topic, count in sorted_topics:
        print(f"{topic}: {count} articoli")
