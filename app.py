import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Unidil AI Production Dashboard", layout="wide")
st.title("🏭 Unidil AI Production Dashboard")

# --- DATA PROCESSING FUNCTIONS ---
@st.cache_data 
def load_and_process_data():
    df = pd.read_csv("Daily Production Data_NEW Format - UNIDIL.csv", header=None)
    date_row = df.iloc[2]
    data = []
    current_section = None
    
    for i in range(3, len(df)):
        label = str(df.iloc[i, 0]).strip().lower()
        if "planned" in label: current_section = "Planned"; continue
        elif "actual" in label: current_section = "Actual"; continue
        elif "yield" in label: current_section = "Yield"; continue
        elif "rejection" in label: current_section = "Rejections"; continue
        elif "stoppage" in label: current_section = "Stoppages"; continue
            
        if current_section and pd.notna(df.iloc[i, 0]):
            process = None
            if "corrugator" in label: process = "Corrugator"
            elif "tuber" in label: process = "Tuber"
            
            if not process: continue 
                
            for col in range(1, len(df.columns)):
                val = df.iloc[i, col]
                date_val = date_row[col]
                if pd.notna(val) and pd.notna(date_val):
                    val_str = str(val).strip()
                    if val_str not in ["-", "", "nan"]:
                        clean_val = val_str.replace(",", "")
                        try:
                            data.append({
                                "Date": str(date_val).strip(),
                                "Process": process,
                                "Metric": current_section,
                                "Value": float(clean_val)
                            })
                        except ValueError:
                            pass 

    clean_df = pd.DataFrame(data)
    final_df = clean_df.pivot_table(
        index=["Date", "Process"], columns="Metric", values="Value", aggfunc="first"
    ).reset_index()

    final_df.columns.name = None
    for col in ["Planned", "Actual", "Yield", "Rejections", "Stoppages"]:
        if col not in final_df.columns:
            final_df[col] = 0
        else:
            final_df[col] = final_df[col].fillna(0)
            
    final_df["Efficiency"] = np.where(final_df["Planned"] > 0, (final_df["Actual"] / final_df["Planned"]) * 100, 0)
    final_df["Production_Loss"] = final_df["Planned"] - final_df["Actual"]
    final_df["Production_Loss"] = final_df["Production_Loss"].clip(lower=0)
    
    return final_df

def generate_insights(df):
    insights = []
    for _, row in df.tail(20).iterrows(): 
        if row["Efficiency"] > 0 and row["Efficiency"] < 90:
            insights.append(f"[{row['Date']}] {row['Process']}: Low efficiency ({row['Efficiency']:.1f}%).")
    return insights

df = load_and_process_data()
insights = generate_insights(df)

# --- SIDEBAR: AI SETUP ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    st.markdown("[Get a free API key here](https://aistudio.google.com/)")

# --- UI TABS ---
tab1, tab2 = st.tabs(["📊 Production Charts", "🤖 AI Assistant"])

with tab1:
    st.header("Daily Production Trends")
    
    st.subheader("Corrugator: Actual vs Planned")
    corr_df = df[df["Process"] == "Corrugator"].set_index("Date")
    st.line_chart(corr_df[["Actual", "Planned"]])
    
    st.subheader("Tuber: Actual vs Planned")
    tuber_df = df[df["Process"] == "Tuber"].set_index("Date")
    st.line_chart(tuber_df[["Actual", "Planned"]])
    
    st.subheader("Raw Data Table")
    st.dataframe(df)

with tab2:
    st.header("Chat with your Factory Data")
    
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to activate the chat.")
    else:
        genai.configure(api_key=api_key)
        
        context_data = df.tail(20).to_markdown(index=False)
        context_insights = "\n".join(insights)
        system_prompt = f"""
        You are an expert industrial AI assistant for the Unidil factory.
        Here is the most recent production data:
        {context_data}
        
        Recent Issues:
        {context_insights}
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_prompt)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about Corrugator efficiency, recent downtime, or yield..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                chat = model.start_chat(history=[])
                
                for msg in st.session_state.messages[:-1]:
                    role = "user" if msg["role"] == "user" else "model"
                    chat.history.append({"role": role, "parts": [msg["content"]]})
                
                response = chat.send_message(prompt)
                st.markdown(response.text)
                
            st.session_state.messages.append({"role": "assistant", "content": response.text})