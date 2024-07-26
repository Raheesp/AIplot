import streamlit as st
import plotly.express as px
import pandas as pd
import nltk
from threading import Thread, Event
from utils import generate_insights, speak_insights, stop_event

nltk.download('vader_lexicon')

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

