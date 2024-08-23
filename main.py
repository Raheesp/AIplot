import streamlit as st
import plotly.express as px
import pandas as pd
import nltk
import tempfile
from pathlib import Path
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chat_models import ChatOpenAI
from streamlit_option_menu import option_menu
from threading import Thread, Event
from utils import generate_insights, speak_insights, stop_event

TMP_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

header = st.container()

# Define RAG function
def RAG(docs, query):
    for source_docs in docs:
        file_name = source_docs.name  # Assuming source_docs has a name attribute
        suffix = Path(file_name).suffix.lower()  # Extract suffix and make it lowercase
        with tempfile.NamedTemporaryFile(delete=True, dir=TMP_DIR.as_posix(), suffix=suffix, mode='wb') as temp_file:
            temp_file.write(source_docs.read())
            temp_file.flush()  # Ensure content is written to disk
            
            # Process the document based on its type
            temp_file.seek(0)  # Go back to the beginning of the file to read it
            if suffix == '.pdf':
                process_pdf(temp_file)
            elif suffix == '.xls':
                process_excel(temp_file)
            elif suffix == '.doc':
                process_doc(temp_file)
            else:
                print(f"Unsupported file type: {suffix}")

def process_pdf(temp_file):
    print("Processing PDF...")

def process_excel(temp_file):
    print("Processing Excel...")

def process_doc(temp_file):
    print("Processing DOC...")

    loader = DirectoryLoader(TMP_DIR.as_posix(), glob=f'**/*.{suffix}', showprogress=True) # type: ignore
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)

    DB_FAISS_PATH = 'vectorestore_lmstudio/faiss'
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence_transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'gpu'})
    db = FAISS.from_documents(text, embeddings)
    db.save.local(DB_FAISS_PATH)

    llm = ChatOpenAI(base_url="http://192.168.1.8:1234", api_key="lm_studio")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True
    )

    chat_history = []
    result = qa_chain({'question': query, 'chat_history': chat_history}) # type: ignore
    st.write(result['answer'])
    chat_history.append((query, result['answer'])) # type: ignore

# Define Streamlit UI function
def streamlit_ui():
    with st.sidebar:
        choice = option_menu('Navigation', ['Home', 'Data analysis', 'Chat with Documents'], default_index=0)

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

    elif choice == 'Chat with Documents':
        with header:
            st.title("Chat with Documents")
            st.write('Upload a Document that you want to Chat with!')
            source_docs = st.file_uploader(label="Upload a Document", type=["exel", "pdf", "Doc"], accept_multiple_files=True)
            if not source_docs:
                st.warning("Please Upload a Document")
            else:
                query = st.chat_input()
                RAG(source_docs, query)

streamlit_ui()

            

    
  