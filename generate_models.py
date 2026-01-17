import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Check for data
if os.path.exists('True.csv') and os.path.exists('Fake.csv'):
    print("Loading datasets...")
    try:
        real = pd.read_csv("True.csv")
        fake = pd.read_csv("Fake.csv")
        real['Category'] = 1 # Real
        fake['Category'] = 0 # Fake
        
        # Merge
        dataset = pd.concat([real, fake]).reset_index(drop=True)
        # Handle missing values if any
        dataset['title'] = dataset['title'].fillna('')
        dataset['text'] = dataset['text'].fillna('')
        dataset['final_text'] = dataset['title'] + " " + dataset['text']
        
        # Train
        print("Training model...")
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X = vectorizer.fit_transform(dataset['final_text'])
        y = dataset['Category']
        
        model = LogisticRegression()
        model.fit(X, y)
        print("Model trained on full dataset.")
    except Exception as e:
        print(f"Error training on dataset: {e}")
        exit(1)
    
else:
    print("Datasets not found. Creating dummy model for demonstration...")
    # Dummy data
    data = {
        'text': [
            "The economy is growing at a steady pace according to the latest reports.", 
            "Scientists have discovered a new species of bird in the Amazon rainforest.", 
            "The parliament voted largely in favor of the new education bill.",
            "Aliens have landed in Washington DC and are demanding to speak to the leader.",
            "You have won the lottery! Click this link to claim your million dollar prize immediately.",
            "Doctors conceal this one weird trick to cure all diseases instantly."
        ],
        'title': [
            "Economic Growth Report", 
            "New Bird Species Found", 
            "Education Bill Passed", 
            "Alien Invasion Confirmed", 
            "Lottery Winner Alert", 
            "Miracle Cure Revealed"
        ],
        'Category': [1, 1, 1, 0, 0, 0] # 1: Real, 0: Fake
    }
    dataset = pd.DataFrame(data)
    dataset['final_text'] = dataset['title'] + " " + dataset['text']
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(dataset['final_text'])
    y = dataset['Category']
    
    model = LogisticRegression()
    model.fit(X, y)

# Save
print("Saving models...")
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
print("Done. model.pkl and vectorizer.pkl created.")
