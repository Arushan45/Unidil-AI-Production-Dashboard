import os
import re
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
MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


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


def _extract_unit_from_label(label_text):
    match = re.search(r"\(([^)]+)\)", str(label_text))
    if not match:
        return ""
    return match.group(1).strip()


def _build_unit_lookup(clean_df):
    lookup = {}
    if clean_df.empty or "Unit" not in clean_df.columns:
        return lookup
    unit_df = (
        clean_df[["Process", "Metric", "Unit"]]
        .dropna()
        .drop_duplicates(subset=["Process", "Metric"], keep="last")
    )
    for _, row in unit_df.iterrows():
        lookup[(row["Process"], row["Metric"])] = str(row["Unit"]).strip()
    return lookup


def _metric_unit(row, metric):
    if metric == "Efficiency":
        return "%"
    return str(row.get(f"{metric}_Unit", "") or "").strip()


def _format_value_with_unit(value, unit):
    if unit:
        return f"{value:.2f} {unit}"
    return f"{value:.2f}"


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
        return pd.DataFrame(columns=["Date", "Process", "Metric", "Value", "Unit"])

    date_row = raw_df.iloc[2]
    data = []
    current_section = None

    for i in range(3, len(raw_df)):
        raw_label = str(raw_df.iloc[i, 0]).strip()
        label = raw_label.lower()
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
                    "Unit": _extract_unit_from_label(raw_label),
                }
            )

    return pd.DataFrame(data)


@st.cache_data
def load_and_process_data():
    local_raw_df = pd.read_csv("Daily Production Data_NEW Format - UNIDIL.csv", header=None)
    local_long_df = _parse_production_sheet(local_raw_df)

    google_long_df = pd.DataFrame(columns=["Date", "Process", "Metric", "Value", "Unit"])
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
    unit_lookup = _build_unit_lookup(clean_df)

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
        final_df[f"{col}_Unit"] = final_df.apply(
            lambda r: unit_lookup.get((r["Process"], col), ""),
            axis=1,
        )

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


def _extract_query_date(query_text):
    text = query_text.lower()

    # Handles: jan21, jan 21, january 21
    match = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*([0-9]{1,2})\b",
        text,
    )
    if not match:
        return None

    month_token = match.group(1)
    day = int(match.group(2))
    month = MONTH_MAP.get(month_token, MONTH_MAP.get(month_token[:3]))
    if not month:
        return None

    year = datetime.now().year
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def _extract_query_dates(query_text):
    text = query_text.lower()
    matches = re.finditer(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*([0-9]{1,2})\b",
        text,
    )
    dates = []
    seen = set()
    year = datetime.now().year
    for match in matches:
        month_token = match.group(1)
        day = int(match.group(2))
        month = MONTH_MAP.get(month_token, MONTH_MAP.get(month_token[:3]))
        if not month:
            continue
        try:
            ts = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
        key = (ts.month, ts.day)
        if key not in seen:
            dates.append(ts)
            seen.add(key)
    return dates


def _extract_query_process(query_text):
    text = query_text.lower()
    if "tuber" in text:
        return "Tuber"
    if "corrugator" in text:
        return "Corrugator"
    return None


def _extract_query_metric(query_text):
    text = query_text.lower()
    if any(k in text for k in ["downtime", "stoppage", "stoppages"]):
        return "Stoppages"
    if "rejection" in text:
        return "Rejections"
    if "yield" in text:
        return "Yield"
    if "actual" in text:
        return "Actual"
    if "planned" in text or "plan" in text:
        return "Planned"
    if "efficiency" in text:
        return "Efficiency"
    return None


def _is_reasoning_query(query_text):
    text = query_text.lower()
    reasoning_keywords = [
        "why",
        "reason",
        "cause",
        "root cause",
        "what happened",
        "issue",
        "problem",
        "improve",
        "how to reduce",
    ]
    return any(k in text for k in reasoning_keywords)


def _build_reasoning_answer(df, query_text):
    if not _is_reasoning_query(query_text):
        return None

    q_date = _extract_query_date(query_text)
    q_process = _extract_query_process(query_text)
    q_metric = _extract_query_metric(query_text)

    if q_date is None:
        return None

    scoped = df.copy()
    if q_process:
        scoped = scoped[scoped["Process"] == q_process]
    scoped = scoped.sort_values("Date_Parsed")

    row_df = scoped[
        (scoped["Date_Parsed"].dt.month == q_date.month)
        & (scoped["Date_Parsed"].dt.day == q_date.day)
    ]
    if row_df.empty:
        return None

    row = row_df.iloc[0]
    process = row["Process"]
    date_label = row["Date"]

    prev_window = scoped[scoped["Date_Parsed"] < row["Date_Parsed"]].tail(5)
    if prev_window.empty:
        prev_window = scoped.head(5)

    prev_means = prev_window[["Yield", "Rejections", "Stoppages", "Actual", "Planned", "Efficiency"]].mean()

    clues = []
    actions = []

    if row["Stoppages"] > max(prev_means["Stoppages"] * 1.2, 60):
        clues.append(
            f"Stoppages were high ({_format_value_with_unit(row['Stoppages'], _metric_unit(row, 'Stoppages'))}) "
            f"vs recent average ({_format_value_with_unit(prev_means['Stoppages'], _metric_unit(row, 'Stoppages'))}), "
            "which likely reduced stable production flow."
        )
        actions.append("Check machine downtime logs and top stoppage reasons for that shift/day.")

    if row["Rejections"] > max(prev_means["Rejections"] * 1.2, 50):
        clues.append(
            f"Rejections were elevated ({_format_value_with_unit(row['Rejections'], _metric_unit(row, 'Rejections'))}) "
            f"vs recent average ({_format_value_with_unit(prev_means['Rejections'], _metric_unit(row, 'Rejections'))}), "
            "which can reduce effective yield/output."
        )
        actions.append("Review defect categories, material lot quality, and setup/changeover parameters.")

    if row["Actual"] < row["Planned"]:
        loss = row["Planned"] - row["Actual"]
        clues.append(
            f"Actual was below planned by {_format_value_with_unit(loss, _metric_unit(row, 'Actual'))} "
            f"({_format_value_with_unit(row['Actual'], _metric_unit(row, 'Actual'))} vs "
            f"{_format_value_with_unit(row['Planned'], _metric_unit(row, 'Planned'))}), "
            "indicating execution shortfall on that day."
        )
        actions.append("Validate manpower, speed losses, and micro-stops during peak planned hours.")

    if row["Yield"] < prev_means["Yield"]:
        drop = prev_means["Yield"] - row["Yield"]
        clues.append(
            f"Yield was lower than recent trend by {_format_value_with_unit(drop, _metric_unit(row, 'Yield'))} "
            f"({_format_value_with_unit(row['Yield'], _metric_unit(row, 'Yield'))} vs "
            f"{_format_value_with_unit(prev_means['Yield'], _metric_unit(row, 'Yield'))})."
        )
        actions.append("Audit process settings around that date and compare with the previous 3-5 days.")

    if not clues:
        clues.append("No strong anomaly signal was detected from stoppages/rejections/plan gap on that date.")
        actions.append("Check shift notes and maintenance remarks for non-quantified events.")

    metric_line = ""
    if q_metric and q_metric in row.index:
        metric_line = (
            f"{process} {q_metric} on {date_label}: "
            f"{_format_value_with_unit(row[q_metric], _metric_unit(row, q_metric))}\n\n"
        )

    return (
        f"{metric_line}Likely reasons for {process} on {date_label}:\n"
        + "\n".join([f"- {c}" for c in clues])
        + "\n\nRecommended checks/actions:\n"
        + "\n".join([f"- {a}" for a in actions[:3]])
    )


def _build_factory_ops_answer(df, query_text):
    text = query_text.lower()
    q_process = _extract_query_process(query_text)
    scoped = df[df["Process"] == q_process].copy() if q_process else df.copy()
    scoped = scoped.sort_values("Date_Parsed")

    if "highest stoppage" in text or "max stoppage" in text:
        row = scoped.loc[scoped["Stoppages"].idxmax()]
        return (
            f"Highest stoppage: {row['Process']} on {row['Date']} = "
            f"{_format_value_with_unit(row['Stoppages'], _metric_unit(row, 'Stoppages'))}"
        )

    if "lowest yield" in text or "worst yield" in text:
        non_zero = scoped[scoped["Yield"] > 0]
        if not non_zero.empty:
            row = non_zero.loc[non_zero["Yield"].idxmin()]
            return (
                f"Lowest yield: {row['Process']} on {row['Date']} = "
                f"{_format_value_with_unit(row['Yield'], _metric_unit(row, 'Yield'))}"
            )

    if "best day" in text and ("actual" in text or "production" in text):
        row = scoped.loc[scoped["Actual"].idxmax()]
        return (
            f"Best actual production day: {row['Process']} on {row['Date']} = "
            f"{_format_value_with_unit(row['Actual'], _metric_unit(row, 'Actual'))}"
        )

    if "summary" in text:
        month_date = _extract_query_date(query_text)
        summary_df = scoped
        if month_date is not None:
            summary_df = summary_df[summary_df["Date_Parsed"].dt.month == month_date.month]
        grouped = summary_df.groupby("Process", as_index=False)[["Planned", "Actual", "Stoppages", "Rejections"]].sum()
        if grouped.empty:
            return None
        lines = []
        for _, r in grouped.iterrows():
            eff = (r["Actual"] / r["Planned"] * 100) if r["Planned"] > 0 else 0
            sample = summary_df[summary_df["Process"] == r["Process"]].iloc[0]
            lines.append(
                f"{r['Process']}: Planned {_format_value_with_unit(r['Planned'], _metric_unit(sample, 'Planned'))}, "
                f"Actual {_format_value_with_unit(r['Actual'], _metric_unit(sample, 'Actual'))}, "
                f"Efficiency {_format_value_with_unit(eff, _metric_unit(sample, 'Efficiency'))}, "
                f"Stoppages {_format_value_with_unit(r['Stoppages'], _metric_unit(sample, 'Stoppages'))}, "
                f"Rejections {_format_value_with_unit(r['Rejections'], _metric_unit(sample, 'Rejections'))}"
            )
        return "Factory summary:\n" + "\n".join([f"- {x}" for x in lines])

    return None


def _direct_data_answer(df, query_text):
    if _is_reasoning_query(query_text):
        return None

    q_date = _extract_query_date(query_text)
    q_process = _extract_query_process(query_text)
    q_metric = _extract_query_metric(query_text)

    if q_date is None:
        return None

    filtered = df.copy()
    filtered = filtered[
        (filtered["Date_Parsed"].dt.month == q_date.month)
        & (filtered["Date_Parsed"].dt.day == q_date.day)
    ]
    if q_process:
        filtered = filtered[filtered["Process"] == q_process]

    if filtered.empty:
        date_label = _format_date_label(q_date)
        if q_process:
            return f"No {q_process} data found for {date_label}."
        return f"No data found for {date_label}."

    row = filtered.iloc[0]
    date_label = row["Date"]
    process_label = row["Process"]

    if q_metric and q_metric in row.index:
        value = row[q_metric]
        return (
            f"{process_label} {q_metric} on {date_label}: "
            f"{_format_value_with_unit(value, _metric_unit(row, q_metric))}"
        )

    return (
        f"{process_label} on {date_label} -> "
        f"Planned: {_format_value_with_unit(row['Planned'], _metric_unit(row, 'Planned'))}, "
        f"Actual: {_format_value_with_unit(row['Actual'], _metric_unit(row, 'Actual'))}, "
        f"Yield: {_format_value_with_unit(row['Yield'], _metric_unit(row, 'Yield'))}, "
        f"Rejections: {_format_value_with_unit(row['Rejections'], _metric_unit(row, 'Rejections'))}, "
        f"Stoppages: {_format_value_with_unit(row['Stoppages'], _metric_unit(row, 'Stoppages'))}, "
        f"Efficiency: {_format_value_with_unit(row['Efficiency'], _metric_unit(row, 'Efficiency'))}"
    )


def _build_compare_answer(df, query_text):
    text = query_text.lower()
    dates = _extract_query_dates(query_text)
    if len(dates) < 2:
        return None
    if not any(k in text for k in ["compare", "vs", "difference", "diff"]):
        return None

    date_a, date_b = dates[0], dates[1]
    q_process = _extract_query_process(query_text)

    compare_df = df[
        ((df["Date_Parsed"].dt.month == date_a.month) & (df["Date_Parsed"].dt.day == date_a.day))
        | ((df["Date_Parsed"].dt.month == date_b.month) & (df["Date_Parsed"].dt.day == date_b.day))
    ].copy()
    if q_process:
        compare_df = compare_df[compare_df["Process"] == q_process]

    if compare_df.empty:
        return {"text": f"No data available for {_format_date_label(date_a)} and {_format_date_label(date_b)}."}

    compare_df = compare_df.sort_values(["Process", "Date_Parsed"])
    table_df = compare_df[
        [
            "Date",
            "Process",
            "Planned",
            "Planned_Unit",
            "Actual",
            "Actual_Unit",
            "Efficiency",
            "Stoppages",
            "Stoppages_Unit",
        ]
    ].copy()
    table_df = table_df.round(2)

    # Chart view: aggregate by date for clean visualization.
    chart_df = (
        compare_df.groupby("Date", as_index=False)[["Actual", "Planned"]]
        .sum()
        .set_index("Date")
        .loc[[_format_date_label(date_a), _format_date_label(date_b)]]
    )

    summary_lines = []
    for process, group in compare_df.groupby("Process"):
        g = group.set_index("Date")
        label_a = _format_date_label(date_a)
        label_b = _format_date_label(date_b)
        if label_a in g.index and label_b in g.index:
            actual_diff = g.loc[label_b, "Actual"] - g.loc[label_a, "Actual"]
            planned_diff = g.loc[label_b, "Planned"] - g.loc[label_a, "Planned"]
            summary_lines.append(
                f"{process}: Actual {'up' if actual_diff >= 0 else 'down'} "
                f"{_format_value_with_unit(abs(actual_diff), _metric_unit(g.loc[label_b], 'Actual'))}, "
                f"Planned {'up' if planned_diff >= 0 else 'down'} "
                f"{_format_value_with_unit(abs(planned_diff), _metric_unit(g.loc[label_b], 'Planned'))} "
                f"from {label_a} to {label_b}."
            )

    title = f"Comparison: {_format_date_label(date_a)} vs {_format_date_label(date_b)}"
    summary_text = "\n".join(summary_lines) if summary_lines else "Comparison generated."
    response_text = f"**{title}**\n\n{summary_text}"

    return {
        "text": response_text,
        "table_records": table_df.to_dict(orient="records"),
        "chart_records": chart_df.reset_index().to_dict(orient="records"),
    }


def _build_context_for_query(df, query_text):
    q_date = _extract_query_date(query_text)
    q_process = _extract_query_process(query_text)

    context_df = df.copy()
    if q_process:
        context_df = context_df[context_df["Process"] == q_process]
    if q_date is not None:
        context_df = context_df[
            (context_df["Date_Parsed"].dt.month == q_date.month)
            | (context_df["Date_Parsed"].dt.month == q_date.month - 1)
            | (context_df["Date_Parsed"].dt.month == q_date.month + 1)
        ]
    if context_df.empty:
        context_df = df.tail(60)
    return context_df.drop(columns=["Date_Parsed"])


def _select_working_model(system_prompt):
    # Prefer newer models, but gracefully fallback based on account availability.
    preferred_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    try:
        available = set()
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                available.add(model.name.replace("models/", ""))
        for model_name in preferred_models:
            if model_name in available:
                return genai.GenerativeModel(model_name, system_instruction=system_prompt), model_name
    except Exception:
        pass

    # Final fallback if listing models is unavailable.
    fallback_name = preferred_models[0]
    return genai.GenerativeModel(fallback_name, system_instruction=system_prompt), fallback_name


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
        st.error("Gemini API key is not set. Add GEMINI_API_KEY (or GOOGLE_API_KEY) in Streamlit Secrets.")
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
        model, model_name = _select_working_model(system_prompt)
        st.caption(f"AI model: {model_name}")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("table_records"):
                    st.dataframe(pd.DataFrame(message["table_records"]), use_container_width=True)
                if message.get("chart_records"):
                    chart_df = pd.DataFrame(message["chart_records"]).set_index("Date")
                    st.bar_chart(chart_df[["Actual", "Planned"]], use_container_width=True)

        if prompt := st.chat_input("Ask about Corrugator efficiency, recent downtime, or yield..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_text = None
                response_payload = None

                # First: deterministic compare answer for 2-date comparison questions.
                response_payload = _build_compare_answer(df, prompt)
                if response_payload:
                    response_text = response_payload["text"]
                    if response_payload.get("table_records"):
                        st.dataframe(pd.DataFrame(response_payload["table_records"]), use_container_width=True)
                    if response_payload.get("chart_records"):
                        chart_df = pd.DataFrame(response_payload["chart_records"]).set_index("Date")
                        st.bar_chart(chart_df[["Actual", "Planned"]], use_container_width=True)

                # Second: deterministic single-date answer.
                if not response_text:
                    response_text = _build_reasoning_answer(df, prompt)

                # Third: common factory-ops questions (best/worst/summary).
                if not response_text:
                    response_text = _build_factory_ops_answer(df, prompt)

                # Fourth: deterministic single-date answer.
                if not response_text:
                    response_text = _direct_data_answer(df, prompt)

                # Fifth: model answer with relevant context for broader questions.
                if not response_text:
                    try:
                        context_df = _build_context_for_query(df, prompt)
                        context_data = context_df.to_markdown(index=False)

                        recent_messages = st.session_state.messages[-8:]
                        conversation = "\n".join(
                            [f"{m['role'].upper()}: {m['content']}" for m in recent_messages]
                        )
                        full_prompt = (
                            "Use the factory dataset context below and do not claim missing data if it is present.\n"
                            "If user asks for a specific date/process value, answer with exact numbers.\n\n"
                            f"Dataset context:\n{context_data}\n\n"
                            f"Conversation so far:\n{conversation}\n\n"
                            f"Latest user question:\n{prompt}"
                        )
                        response = model.generate_content(full_prompt)
                        response_text = response.text
                    except Exception as e:
                        st.error("AI assistant request failed. Please check your API key/project access and try again.")
                        with st.expander("Error details"):
                            st.code(str(e))

                if response_text:
                    st.markdown(response_text)
                    if response_payload:
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response_text,
                                "table_records": response_payload.get("table_records"),
                                "chart_records": response_payload.get("chart_records"),
                            }
                        )
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
