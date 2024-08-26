# AIplot
This project is a comprehensive data analysis dashboard built using Streamlit and Plotly, designed to handle various data types such as sales forecasting and market analysis. The dashboard features dynamic graph generation, including histograms, box plots, pie charts, heatmaps, scatter plots, bar charts, line plots, and area charts.

TheBloke/Llama-2-7B-Chat-GGML

# Data Analysis Dashboard with Voice Integration

## Overview

This project is a Data Analysis Dashboard built with [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/python/). It allows users to upload CSV files, visualize data with various types of charts, and generate insights. The dashboard also includes voice integration for enhanced interactivity and control.

## Features

- **Dynamic Graph Generation:** Create various types of charts including histograms, box plots, pie charts, heatmaps, scatter plots, bar charts, line plots, and area charts.
- **Automatic Graph Naming:** The dashboard automatically names graphs and charts based on the data being visualized.
- **Text-to-Speech Integration:** Uses [pyttsx3](https://pypi.org/project/pyttsx3/) to read out data insights.
- **Voice Commands:** Control the dashboard using voice commands to open websites (e.g., YouTube, GitHub) and generate insights.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/data-analysis-dashboard.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd data-analysis-dashboard
    ```

3. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. **Install required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app:**

    ```bash
    streamlit run main.py
    ```

2. **Upload a CSV file:**
   - Navigate to the dashboard in your web browser (usually `http://localhost:8501`).
   - Use the file uploader in the sidebar to upload a CSV file.

3. **Interact with the dashboard:**
   - Select the type of graph you want to generate.
   - Click on the "Explain Summary" button to get a spoken summary of the data insights.
   - Use voice commands to interact with the dashboard (e.g., "generate insight", "open YouTube").

## File Structure

- `main.py`: The main Streamlit application file.
- `utils.py`: Contains utility functions for generating insights and text-to-speech.
- `voice_recognition.py`: Handles voice recognition and command processing.
- `requirements.txt`: Lists the project dependencies.

## Requirements

- Python 3.8+
- Streamlit
- Plotly
- Pandas
- pyttsx3
- SpeechRecognition
- nltk
- Wikipedia-API
- DuckDB

## Troubleshooting

- **Columns have mixed types warning:** This may occur when importing CSV files with inconsistent data types. Use `dtype` argument in `pd.read_csv()` or set `low_memory=False`.
- **Voice Recognition Issues:** Ensure that your microphone is properly set up and recognized by the system.

## Contributing

Feel free to fork the repository and submit pull requests. For significant changes, please open an issue first to discuss the proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [pyttsx3](https://pypi.org/project/pyttsx3/)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
- [nltk](https://www.nltk.org/)
- [Wikipedia-API](https://pypi.org/project/wikipedia-api/)
- [DuckDB](https://duckdb.org/)


