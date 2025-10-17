import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import re
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG & SETUP ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Survey Chatbot", layout="wide")
st.title("📊 Interactive Survey Analyst (Gemini Powered)")
st.markdown("Ask any question related to your survey data and get instant answers and visualizations!")

# --- DATA LOADING ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/1y11HDYSm5JE_leMW7jlVOkQ8CnnguzY_EQ8iz1ePf_U/export?format=csv"

@st.cache_data(ttl=300)
def load_survey():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.replace(r'[\(\)\?\[\]\.\,]', '', regex=True).str.strip().str.replace(' ', '_')
    df.dropna(how="all", inplace=True)
    return df

df = load_survey()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = df # Store DataFrame in session state

st.sidebar.success(f"✅ Loaded {len(st.session_state.df)} latest responses successfully!")

# --- NEW: PANDAS EXECUTION TOOL (Added a chart generation function) ---
def analyze_data_with_pandas(code_string: str) -> str:
    """Safely executes a single-line Pandas operation."""
    try:
        # Check for plotting request (e.g., using .plot(), sns., or plt.)
        is_plot_request = any(keyword in code_string.lower() for keyword in ['.plot(', 'sns.', 'plt.'])
        
        # Execute the code
        result = eval(code_string, {'df': st.session_state.df, 'pd': pd, 'sns': sns, 'plt': plt, '__builtins__': {}})
        
        if is_plot_request:
            # If the code successfully generated a plot, display it.
            st.session_state.chart_data = plt.gcf()
            plt.close() # Close the figure to free memory
            return "Analysis Result: A chart was successfully generated."
        
        # Format the result for the LLM
        if isinstance(result, (pd.Series, pd.DataFrame)):
            return f"Analysis Result:\n{result.to_string()}"
        return f"Analysis Result:\n{str(result)}"
        
    except Exception as e:
        # Return the error to the LLM so it can try again or explain the issue
        return f"PANDAS EXECUTION ERROR: {e}"

# --- MODIFIED CHATBOT LOGIC (Router + Tool Use) ---
def ask_gemini(question, df):
    # Prepare context for Gemini
    columns = df.columns.tolist()
    
    # Tool description for the LLM (updated for visualization)
    tool_prompt = f"""
    You have access to a Python function `analyze_data_with_pandas(code_string)` to perform calculations and visualizations on the DataFrame named 'df'.
    
    DataFrame Columns (USE THESE EXACTLY): {columns}
    
    Example Usage for Calculations: `analyze_data_with_pandas("df['Age'].mean()")`
    Example Usage for Charts: If the user asks for a chart, use Matplotlib/Seaborn.
    - Bar Chart of Preferences: `analyze_data_with_pandas("sns.countplot(y='Preferred_Shopping_Mode', data=df).figure")`
    - Histogram of Age: `analyze_data_with_pandas("df['Age'].hist().figure")`
    
    If the question requires a calculation or visualization, you MUST output ONLY the Python function call.
    If the question is descriptive or requires high-level synthesis (e.g., 'Summarize the findings'), use the final answer format.
    """
    
    # 1. Ask Gemini to generate code
    code_generation_prompt = f"{tool_prompt}\n\nUser Question: {question}"
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    response = model.generate_content(code_generation_prompt)
    
    # Check if the response contains a function call (Code Interpreter/Tool Use)
    if 'analyze_data_with_pandas(' in response.text:
        match = re.search(r'analyze_data_with_pandas\("(.+?)"\)', response.text, re.DOTALL)
        if match:
            code_to_run = match.group(1).replace('\n', ' ').strip()
            
            # Execute the generated code
            pandas_result = analyze_data_with_pandas(code_to_run)
            
            # 3. Send the result back to Gemini for final formatting
            final_prompt = f"User Question: {question}\n\nAnalysis Result:\n{pandas_result}\n\nBased ONLY on the provided result, give a concise, professional answer."
            final_response = model.generate_content(final_prompt)
            return final_response.text.strip(), code_to_run
        
    # Fallback to the original descriptive method (for non-statistical questions)
    else:
        # Use the original prompt structure for descriptive/synthesis questions
        columns_list = df.columns.tolist()
        info = f"The dataset has {len(df)} responses and the following columns: {columns_list}."
        summary = df.describe(include='all').to_string()

        fallback_prompt = f"""
        You are a professional data analyst chatbot.
        Use the information provided below to answer the question clearly and directly.
        Do NOT mention any dataset, sources, summaries, or reasoning steps in your reply.
        Just provide the concise, factual answer.

        Dataset Info:
        {info}

        Statistical Summary:
        {summary}

        Example Rows:
        {df.sample(min(5, len(df))).to_csv(index=False)}

        User Question:
        {question}

        Give only the final answer in a short, professional sentence or paragraph.
        """
        fallback_response = model.generate_content(fallback_prompt)
        return fallback_response.text.strip(), None
    
    return "Error: Could not process request.", None # Default error return

# --- MODIFIED CHAT INTERFACE ---

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chart"):
            # Recreate the plot object from the stored data if necessary, or just display
            st.pyplot(message["chart"])
        if message.get("code"):
             st.code(message["code"], language='python')

# 2. Input Field (main chat input)
user_query = st.chat_input("💬 Ask a question about your survey:")

# 3. Process Input
if user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get chatbot response
    with st.spinner("Analyzing data..."):
        answer, code_run = ask_gemini(user_query, st.session_state.df)

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)
        
        message_data = {"role": "assistant", "content": answer}

        # Check for and display a generated chart
        if 'chart_data' in st.session_state and st.session_state.chart_data is not None:
            st.pyplot(st.session_state.chart_data)
            message_data["chart"] = st.session_state.chart_data
            del st.session_state.chart_data # Clear chart after display
        
        # Display the executed code for transparency and debug
        if code_run:
            with st.expander("Show Executed Pandas Code"):
                st.code(code_run, language='python')
                message_data["code"] = code_run

        st.session_state.messages.append(message_data)

# import streamlit as st
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# # ---------- LOAD .env ----------
# load_dotenv()  # Make sure .env contains GOOGLE_API_KEY
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     st.error("⚠️ Google API Key not found in .env")
#     st.stop()

# genai.configure(api_key=api_key)

# # ---------- STREAMLIT PAGE CONFIG ----------
# st.set_page_config(page_title="Survey Chatbot", layout="wide")
# st.title("📊 Dynamic Survey Chatbot")
# st.markdown("Ask any question about your survey responses!")

# # ---------- LOAD SURVEY DATA ----------
# SHEET_URL = "https://docs.google.com/spreadsheets/d/1y11HDYSm5JE_leMW7jlVOkQ8CnnguzY_EQ8iz1ePf_U/gviz/tq?tqx=out:csv"

# @st.cache_data(ttl=600)
# def load_data():
#     df = pd.read_csv(SHEET_URL)
#     df.dropna(how='all', inplace=True)
#     return df

# df = load_data()

# # ---------- EMBEDDINGS FOR DESCRIPTIVE QUERIES ----------
# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# model = load_embedding_model()

# @st.cache_data(ttl=600)
# def compute_embeddings(df):
#     df['text'] = df.apply(lambda r: " ".join(map(str, r.values)), axis=1)
#     df['embedding'] = df['text'].apply(lambda x: model.encode(x))
#     return df

# df = compute_embeddings(df)

# # ---------- RETRIEVAL FUNCTION ----------
# def retrieve_context(query, df, top_k=3):
#     q_vec = model.encode(query)
#     df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([x], [q_vec])[0][0])
#     top = df.sort_values("similarity", ascending=False).head(top_k)
#     return " ".join(top['text'].values)

# # ---------- QUERY HANDLER ----------
# def handle_query(query, df):
#     query_lower = query.lower()

#     # Check if query is numeric/statistical
#     if any(word in query_lower for word in ["how many", "count", "average", "percentage", "total"]):
#         # Example: count females preferring online
#         if "female" in query_lower and "online" in query_lower and "preference" in df.columns:
#             if 'Gender' in df.columns and 'Preference' in df.columns:
#                 count = df[(df['Gender'].str.lower()=='female') & 
#                            (df['Preference'].str.lower()=='online')].shape[0]
#                 return f"✅ {count} females prefer online shopping."
#         # Generic numeric fallback
#         return "🔢 Numeric/statistical query detected. Please refine your question using column names."

#     # Descriptive / open-ended queries handled by LLM
#     context = retrieve_context(query, df)
#     prompt = f"Survey context:\n{context}\n\nAnswer the following question concisely:\n{query}"
#     model_g = genai.GenerativeModel("gemini-2.5-flash")
#     response = model_g.generate_content(prompt)
#     return response.text

# # ---------- STREAMLIT UI ----------
# query = st.text_input("Your question:")

# if query:
#     with st.spinner("Generating answer..."):
#         answer = handle_query(query, df)
#         st.subheader("🤖 Chatbot Answer")
#         st.write(answer)
