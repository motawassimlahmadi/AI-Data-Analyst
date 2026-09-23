# 🤖 AI Data Analyst Agent

An autonomous, LLM-powered Data Analysis Assistant built with Python and Streamlit. 
This agent leverages the Gemini API and the concept of **Function Calling (Tool Use)** to interact with tabular datasets dynamically. It allows users to perform Exploratory Data Analysis (EDA), visualize distributions, and clean data using natural language.

## ✨ Features

* **💬 Conversational Interface:** Chat naturally with your dataset. The agent remembers the context of the conversation.
* **📊 Automated EDA:** Instantly get dataset profiles, missing value reports, correlation matrices, and statistical metrics.
* **📈 Dynamic Visualizations:** Asks for a chart, and the agent generates it (Scatter plots, Bar charts, Histograms, Boxplots, Time series) using Matplotlib and Seaborn.
* **🧹 Data Cleaning & Prep:** The agent can actively modify the dataset by dropping useless columns, handling high cardinality, and detecting outliers (IQR method).

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Visualizations:** Matplotlib, Seaborn
* **AI / LLM:** Google Gemini API (via `google-genai` SDK)
* **UI Framework:** Streamlit

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ai-data-analyst.git](https://github.com/yourusername/ai-data-analyst.git)
   cd ai-data-analyst
   ```
2. **Create a virtual environment and install dependencies:**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```
3. **Set up your API Key:**
Create a .env file in the root directory and add your Google Gemini API key
  ```bash
  GEMINI_API_KEY=your_api_key_here
  ```
4. **Run the application:**
  ```bash
streamlit run app.py
