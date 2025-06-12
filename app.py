import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
st.title("Sentiment Analysis On Customer Reviews ")
df=pd.read_csv("combined_sentiment_data.csv")
st.write(df.head())
if "sentence" in df.columns and "sentiment" in df.columns:
    X=df["sentence"]
    y=df["sentiment"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train) 
    X_test_bow = vectorizer.transform(X_test) 

    model=MultinomialNB()
    model.fit(X_train_bow,y_train)
    y_pred=model.predict(X_test_bow)
    accuracy=accuracy_score(y_test,y_pred)
    st.write("### Model Accuracy:",accuracy)
    user_input=st.text_area("Enter a sentence for sentiment Prediction:")
    if user_input:
        user_input_bow=vectorizer.transform([user_input])
        prediction = model.predict(user_input_bow)[0]
        st.write("### Predicted Sentiment:",prediction)
        st.balloons()
else:
        st.error("""The dataset must contain "sentence" and "sentiment" columns.""")

    