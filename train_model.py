# Import pandas for data manipulation.
import pandas as pd

# Import TF-IDF vectorizer for converting product titles into numerical features.
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the Multinomial Naive Bayes classifier.
from sklearn.naive_bayes import MultinomialNB

# Import pickle for saving the trained model and vectorizer.
import pickle


# Load the product dataset.
data = pd.read_csv("products.csv")

print("Dataset loaded successfully.")
print("Original number of rows:", len(data))


# Remove unnecessary spaces from column names.
data.columns = data.columns.str.strip()


# Remove rows with missing product titles or category labels.
data = data.dropna(subset=["Product Title", "Category Label"])


# Standardize inconsistent category labels.
category_mapping = {
    "fridge": "Fridges",
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones"
}

data["Category Label"] = data["Category Label"].replace(category_mapping)


# Define the input feature and target variable.
X = data["Product Title"]
y = data["Category Label"]


# Create the TF-IDF vectorizer.
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)


# Transform the product titles into numerical features.
X_tfidf = tfidf_vectorizer.fit_transform(X)


# Create the final Multinomial Naive Bayes model.
naive_bayes_model = MultinomialNB()


# Train the model using the complete cleaned dataset.
naive_bayes_model.fit(X_tfidf, y)


# Store the vectorizer and model together.
model_package = {
    "vectorizer": tfidf_vectorizer,
    "model": naive_bayes_model
}


# Save the vectorizer and trained model.
with open("product_category_model.pkl", "wb") as file:
    pickle.dump(model_package, file)


print("Training completed successfully.")
print("Number of training samples:", len(X))
print("Number of categories:", y.nunique())
print("Model saved as: product_category_model.pkl")
