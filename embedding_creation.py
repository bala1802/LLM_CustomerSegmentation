import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

'''
'''
def compile_text(x):

    text = f"""customer's birth year : {x['Year_Birth']}, 
    education qualification of customer : {x['Education']}, 
    marital status of customer: {x['Marital_Status']}, 
    customer's yearly household income: {x['Income']},
    number of children in customer's household: {x['Kidhome']},
    number of teenagers in customer's household: {x['Teenhome']},
    date of customer's enrollment with the company: {x['Dt_Customer']},
    number of days since customer's last purchase: {x['Recency']},
    amount spent on wine: {x['MntWines']},
    amount spent on fruits: {x['MntFruits']},
    amount spent on meat products: {x['MntMeatProducts']},
    amount spent on fish products: {x['MntFishProducts']},
    amount spent on sweet products: {x['MntSweetProducts']},
    amount spent on gold products: {x['MntGoldProds']},
    deal based transactions: {x['NumDealsPurchases']},
    online transactions: {x['NumWebPurchases']},
    catalog based transactions: {x['NumCatalogPurchases']},
    transactions done in the stores: {x['NumStorePurchases']},
    website visits per month: {x['NumWebVisitsMonth']},
    customer accepted the marketing campaign 3: {x['AcceptedCmp3']},
    customer accepted the marketing campaign 4: {x['AcceptedCmp4']},
    customer accepted the marketing campaign 5: {x['AcceptedCmp5']},
    customer accepted the marketing campaign 1: {x['AcceptedCmp1']},
    customer accepted the marketing campaign 2: {x['AcceptedCmp2']},
    customer gave complaint: {x['Complain']},
    cost associated with reaching out to the customer as part of marketing: {x['Z_CostContact']},
    revenue from the customer: {x['Z_Revenue']},
    customer's response: {x['Response']},
    """
    
    return text

'''
'''
def encode_data(trainDf):
    sentences = trainDf.apply(lambda x: compile_text(x), axis=1).tolist()
    model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")
    encodedData = model.encode(sentences=sentences, show_progress_bar= True, normalize_embeddings  = True)
    return encodedData

'''
'''
def store_embeddings(embeddings):
    pd.DataFrame(embeddings).to_csv("data/embeddings.csv")

'''
'''
def create_embeddings():
    trainDf = pd.read_csv("data/train.csv")
    trainDf = trainDf.dropna()
    embeddings = encode_data(trainDf=trainDf)
    store_embeddings(embeddings=embeddings)
