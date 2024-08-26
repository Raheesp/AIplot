import streamlit as st
import plotly.express as px
import pandas as pd
import nltk
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp 
from langchain.embeddings import HuggingFaceBgeEmbeddings 
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chat_models import ChatOpenAI
from streamlit_option_menu import option_menu
from threading import Thread, Event
from utils import generate_insights, speak_insights, stop_event
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.csv_loader import CSVLoader
import os
import tempfile
from streamlit_chat import message
from langchain.llms import CTransformers

DB_FAISS_PATH = "vectorstore/db_faiss"


def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def streamlit_ui():
    with st.sidebar:
        choice = option_menu('Navigation', ['Home', 'Data analysis', 'Chat with CSV'], default_index=0)

    if choice == 'Home':
        st.title("Hi, Welcome to AIplot \n Where you can Analyse Data and Also Communicate with the Document")

    elif choice == 'Data analysis':
        st.title("Data Analysis Dashboard")
        uploaded_file = st.file_uploader("Upload Data", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Filename:", uploaded_file.name)
            st.write("Data Preview:", df.head())

            categorical_cols = df.select_dtypes(include=['object']).columns
            date_cols = df.select_dtypes(include=['datetime64']).columns
            numerical_cols = df.select_dtypes(include=['number']).columns

            st.sidebar.subheader("Filter Data")

            selected_categorical_col = st.sidebar.selectbox("Select Categorical Column", categorical_cols)
            if selected_categorical_col:
                unique_values = df[selected_categorical_col].unique()
                selected_value = st.sidebar.selectbox("Select Value", unique_values)
                df = df[df[selected_categorical_col] == selected_value]

            selected_date_col = st.sidebar.selectbox("Select Date Column", date_cols)
            if selected_date_col:
                start_date = st.sidebar.date_input("Start Date", df[selected_date_col].min())
                end_date = st.sidebar.date_input("End Date", df[selected_date_col].max())
                df = df[(df[selected_date_col] >= pd.to_datetime(start_date)) & (df[selected_date_col] <= pd.to_datetime(end_date))]

            graph_type = st.selectbox("Select Graph Type", ['Histogram', 'Box Plot', 'Pie Chart', 'Heatmap', 'Scatter Plot', 'Bar Chart', 'Line Plot', 'Area Chart'])

            if len(numerical_cols) > 0:
                if graph_type == 'Histogram':
                    fig = px.histogram(df, x=numerical_cols[0], title="Histogram")
                elif graph_type == 'Box Plot':
                    fig = px.box(df, y=numerical_cols[0], title="Box Plot")
                elif graph_type == 'Pie Chart':
                    fig = px.pie(df, names=df.columns[0], values=df[numerical_cols[0]], title="Pie Chart")
                elif graph_type == 'Heatmap':
                    if len(numerical_cols) > 1:
                        corr_matrix = df[numerical_cols].corr()
                        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Heatmap")
                    else:
                        fig = None
                elif graph_type == 'Scatter Plot':
                    if len(numerical_cols) > 1:
                        fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1], title="Scatter Plot")
                    else:
                        fig = None
                elif graph_type == 'Bar Chart':
                    if len(numerical_cols) > 0:
                        fig = px.bar(df, x=df.columns[0], y=numerical_cols[0], title="Bar Chart")
                    else:
                        fig = None
                elif graph_type == 'Line Plot':
                    if len(numerical_cols) > 1:
                        fig = px.line(df, x=numerical_cols[0], y=numerical_cols[1], title="Line Plot")
                    else:
                        fig = None
                elif graph_type == 'Area Chart':
                    if len(numerical_cols) > 1:
                        fig = px.area(df, x=numerical_cols[0], y=numerical_cols[1], title="Area Chart")
                    else:
                        fig = None
                
                if fig:
                    st.plotly_chart(fig)

            insights = generate_insights(df)
            st.subheader("Insights")
            for insight in insights:
                st.write(insight)

            if st.button("Explain Summary"):
                stop_event.clear()
                speak_insights(insights)
            
            if st.button("Stop Reading"):
                stop_event.set()

            nltk.download('vader_lexicon')

    elif choice == 'Chat with CSV':
        st.title('Chat with CSV using Llama 🦙')
        uploaded_file = st.sidebar.file_uploader("Upload Your Data",type = "csv")

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            loader = CSVLoader(file_path=tmp_file_path, encoding= "utf-8", csv_args ={
                'delimiter': ','
            })
            data =loader.load()
            st.json(data)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
            db = FAISS.from_documents(data, embeddings)
            db.save_local(DB_FAISS_PATH)
            llm = load_llm()
            chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

            def conversational_chat(query):
                result = chain.invoke({"question": query, "chat_history" :st.session_state['history']})
                st.session_state['history'].append({query, result["answer"]})
                return result["answer"]
            
            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Hello, Ask me anything about "+ uploaded_file.name + "😃"]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey ! :👋🏻: "]

            response_container = st.container()

            container = st.container()

            with container:
                with st.form(key = "my_form", clear_on_submit=True):
                    user_input = st.text_input("query", placeholder = "Talk To Your CSV Data here (:",
                    key='input')
                    sumbit_button = st.form_submit_button(label="chat")
                
                if sumbit_button and user_input:
                    output = conversational_chat(user_input)

                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user',
                        avatar_style = "big-smile")
                        message(st.session_state['generated'][i], key=str(i), avatar_style = "thumbs")

                    

if __name__ == "__main__":
    streamlit_ui()
