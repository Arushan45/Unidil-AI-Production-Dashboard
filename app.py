import os
from datetime import datetime

import google.generativeai as genai
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Unidil AI Production Dashboard", layout="wide")
st.title("Unidil AI Production Dashboard")

# --- DATA PROCESSING FUNCTIONS ---
GOOGLE_SHEET_ID = "1c5K4f49a_HLaRel0xp-sah9MAEyqpSz8zg4KmHCKjqs"
GOOGLE_SHEET_GID = "428263152"
GOOGLE_SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export?format=csv&gid={GOOGLE_SHEET_GID}"
)


def _get_gemini_api_key():
    # Streamlit Cloud secrets (support both key names).
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except StreamlitSecretNotFoundError:
        pass
    # Local env fallbacks.
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY")
    return os.environ.get("GOOGLE_API_KEY")


def _normalize_date_label(date_label):
    label = str(date_label).strip()
    parsed = pd.to_datetime(
        f"{label} {datetime.now().year}",
        format="%b %d %Y",
        errors="coerce",
    )
    if pd.isna(parsed):
        parsed = pd.to_datetime(label, errors="coerce")
    return label, parsed


def _format_date_label(ts):
    return f"{ts.strftime('%b')} {ts.day}"


def _extract_unidil_rows(raw_df):
    first_col = raw_df.iloc[:, 0].astype(str).str.strip().str.upper()
    unidil_rows = first_col[first_col == "UNIDIL"]

    if unidil_rows.empty:
        return [raw_df]

    blocks = []
    start_indexes = list(unidil_rows.index)
    for idx, start_idx in enumerate(start_indexes):
        end_idx = start_indexes[idx + 1] if idx + 1 < len(start_indexes) else len(raw_df)
        block = raw_df.iloc[start_idx:end_idx].reset_index(drop=True)
        blocks.append(block)
    return blocks


def _parse_production_sheet(raw_df):
    if raw_df.shape[0] < 4:
        return pd.DataFrame(columns=["Date", "Process", "Metric", "Value"])

    date_row = raw_df.iloc[2]
    data = []
    current_section = None

    for i in range(3, len(raw_df)):
        label = str(raw_df.iloc[i, 0]).strip().lower()
        if "planned" in label:
            current_section = "Planned"
            continue
        if "actual" in label:
            current_section = "Actual"
            continue
        if "yield" in label:
            current_section = "Yield"
            continue
        if "rejection" in label:
            current_section = "Rejections"
            continue
        if "stoppage" in label:
            current_section = "Stoppages"
            continue

        if not current_section or pd.isna(raw_df.iloc[i, 0]):
            continue

        process = None
        if "corrugator" in label:
            process = "Corrugator"
        elif "tuber" in label:
            process = "Tuber"

        if not process:
            continue

        for col in range(1, len(raw_df.columns)):
            val = raw_df.iloc[i, col]
            date_val = date_row[col]
            if pd.isna(val) or pd.isna(date_val):
                continue

            val_str = str(val).strip()
            if val_str in ["-", "", "nan"]:
                continue

            # Handle formatting glitches such as 110..51.
            clean_val = val_str.replace(",", "").replace("..", ".")
            try:
                numeric_val = float(clean_val)
            except ValueError:
                continue

            data.append(
                {
                    "Date": str(date_val).strip(),
                    "Process": process,
                    "Metric": current_section,
                    "Value": numeric_val,
                }
            )

    return pd.DataFrame(data)


@st.cache_data
def load_and_process_data():
    local_raw_df = pd.read_csv("Daily Production Data_NEW Format - UNIDIL.csv", header=None)
    local_long_df = _parse_production_sheet(local_raw_df)

    google_long_df = pd.DataFrame(columns=["Date", "Process", "Metric", "Value"])
    try:
        google_raw_df = pd.read_csv(GOOGLE_SHEET_CSV_URL, header=None)
        unidil_blocks = _extract_unidil_rows(google_raw_df)
        parsed_blocks = [_parse_production_sheet(block) for block in unidil_blocks]
        google_long_df = pd.concat(parsed_blocks, ignore_index=True)
    except Exception:
        # Fallback to local file when sheet is unavailable.
        pass

    clean_df = pd.concat([local_long_df, google_long_df], ignore_index=True)
    clean_df = clean_df.drop_duplicates(subset=["Date", "Process", "Metric"], keep="last")

    final_df = clean_df.pivot_table(
        index=["Date", "Process"], columns="Metric", values="Value", aggfunc="first"
    ).reset_index()

    final_df.columns.name = None
    for col in ["Planned", "Actual", "Yield", "Rejections", "Stoppages"]:
        if col not in final_df.columns:
            final_df[col] = 0
        else:
            final_df[col] = final_df[col].fillna(0)

    final_df["Efficiency"] = np.where(
        final_df["Planned"] > 0,
        (final_df["Actual"] / final_df["Planned"]) * 100,
        0,
    )
    final_df["Production_Loss"] = final_df["Planned"] - final_df["Actual"]
    final_df["Production_Loss"] = final_df["Production_Loss"].clip(lower=0)

    normalized = final_df["Date"].apply(_normalize_date_label)
    final_df["Date_Parsed"] = [item[1] for item in normalized]
    final_df = final_df.dropna(subset=["Date_Parsed"]).copy()

    # Build a continuous daily calendar so charts show Jan 1, Jan 2, Jan 3... with no gaps in ordering.
    all_dates = pd.date_range(final_df["Date_Parsed"].min(), final_df["Date_Parsed"].max(), freq="D")
    all_processes = sorted(final_df["Process"].dropna().unique().tolist())
    calendar_df = pd.MultiIndex.from_product([all_dates, all_processes], names=["Date_Parsed", "Process"]).to_frame(index=False)

    final_df = calendar_df.merge(
        final_df.drop(columns=["Date"]),
        on=["Date_Parsed", "Process"],
        how="left",
    )

    for col in ["Planned", "Actual", "Yield", "Rejections", "Stoppages"]:
        final_df[col] = final_df[col].fillna(0)

    final_df["Date"] = final_df["Date_Parsed"].apply(_format_date_label)
    final_df["Efficiency"] = np.where(
        final_df["Planned"] > 0,
        (final_df["Actual"] / final_df["Planned"]) * 100,
        0,
    )
    final_df["Production_Loss"] = (final_df["Planned"] - final_df["Actual"]).clip(lower=0)
    final_df = final_df.sort_values(by=["Date_Parsed", "Process"]).reset_index(drop=True)

    return final_df


def generate_insights(df):
    insights = []
    for _, row in df.tail(20).iterrows():
        if row["Efficiency"] > 0 and row["Efficiency"] < 90:
            insights.append(f"[{row['Date']}] {row['Process']}: Low efficiency ({row['Efficiency']:.1f}%).")
    return insights


def _prepare_chart_df(df, process_name):
    chart_df = df[df["Process"] == process_name].copy()
    chart_df = chart_df.dropna(subset=["Date_Parsed"]).sort_values("Date_Parsed")
    # Plot only meaningful daily production points for this process.
    chart_df = chart_df[(chart_df["Actual"] > 0) | (chart_df["Planned"] > 0)]
    chart_df = chart_df.drop_duplicates(subset=["Date_Parsed"], keep="last")
    # Keep true datetime index so Streamlit draws Jan -> Feb -> Mar in exact order.
    return chart_df.set_index("Date_Parsed")[["Actual", "Planned"]]


df = load_and_process_data()
insights = generate_insights(df)

# --- UI TABS ---
tab1, tab2 = st.tabs(["Production Charts", "AI Assistant"])

with tab1:
    st.header("Daily Production Trends")

    st.subheader("Corrugator: Actual vs Planned")
    corr_df = _prepare_chart_df(df, "Corrugator")
    st.line_chart(corr_df[["Actual", "Planned"]])

    st.subheader("Tuber: Actual vs Planned")
    tuber_df = _prepare_chart_df(df, "Tuber")
    st.line_chart(tuber_df[["Actual", "Planned"]])

    st.subheader("Raw Data Table")
    display_df = df.drop(columns=["Date_Parsed"]).copy()
    st.dataframe(display_df)

with tab2:
    st.header("Chat with your Factory Data")
    api_key = _get_gemini_api_key()
    if not api_key:
        st.error("Google Gemini API Key is not set. Add GOOGLE_API_KEY in Streamlit Secrets.")
    else:
        genai.configure(api_key=api_key)
        context_data = df.tail(20).drop(columns=["Date_Parsed"]).to_markdown(index=False)
        context_insights = "\n".join(insights)
        system_prompt = f"""
        You are an expert industrial AI assistant for the Unidil factory.
        Here is the most recent production data:
        {context_data}

        Recent Issues:
        {context_insights}
        """
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_prompt)

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
