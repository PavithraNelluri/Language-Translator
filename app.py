import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECTNAME"] = os.getenv("LANGCHAIN_PROJECTNAME")

# Initialize LLM
llm = ChatGroq(model='gemma2-9b-it', groq_api_key=groq_api_key)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the input from {source_language} to {target_language}.Only give the translation sentences no need of explanation."),
    ("user", "Question: {question}")
])
output_parser = StrOutputParser()

# Create the translation chain
chain = prompt | llm | output_parser

# Streamlit UI
st.title("Language Translator")
st.write("Translate any language to your required language using AI.")

# User input fields
source_language = st.text_input("Enter source language:")
target_language = st.text_input("Enter target language:")
question = st.text_area("Enter text to translate:")

if st.button("Translate"):
    if source_language and target_language and question:
        response = chain.invoke({
            "source_language": source_language,
            "target_language": target_language,
            "question": question
        })
        st.subheader("Translated Text:")
        st.write(response)
    else:
        st.warning("Please fill in all fields before translating.")


