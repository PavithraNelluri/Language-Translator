import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import HumanMessage, AIMessage


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
memory = StreamlitChatMessageHistory()
llm = ChatGroq(
    model="llama3-8b-8192", 
    groq_api_key=groq_api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly fitness expert. "
    "Answer only fitness-related questions like workouts, food, supplements, or wellness. "
    "Kindly refuse unrelated topics."
    "Remember, don't give too long answers."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}")
])
chain = prompt | llm | StrOutputParser()
for msg in memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

user_query = st.chat_input("Ask a fitness question...")
if user_query:
    st.chat_message("user").markdown(user_query)
    memory.add_user_message(user_query)
    response = chain.invoke({
        "question": user_query,
        "history": memory.messages
    })
    st.chat_message("assistant").markdown(response)
    memory.add_ai_message(response)
