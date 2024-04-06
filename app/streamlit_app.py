# streamlit_app.py

import streamlit as st
from model.Inference import *  # Import FastAPI function

# Streamlit app logic
def main():
    st.title("Two-tower Document prediction")

    # Add Streamlit components
    query = st.text_input("Enter your query:")
    if st.button("Predict"):
        documents = predict_passages(query)
        st.header("Relevant Documents:")
        for idx, doc in enumerate(documents, 1):
            st.write(f"{idx}. {doc}")

if __name__ == "__main__":
    main()
