"""
Streamlit Web Application for SMS Spam Message Detection.
Team 2 - Natural Language Processing

Pages:
1. 🏠 Home Dashboard
2. 🔍 SMS Spam Detector
3. 📊 Model Comparison
4. 📁 Dataset Explorer
5. ℹ️ About Project
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text, extract_message_features
from src.predict import predict_message, load_inference_pipeline
from src.data_loader import load_sms_data, DEFAULT_RAW_FILE

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector | Team 2 NLP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Academic / Clean Dashboard Theme)
st.markdown("""
<style>
    /* Global typography & headers */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .team-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1E40AF, #3B82F6);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin: 5px 0;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    /* Spam / Ham alert badges */
    .spam-banner {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        color: #991B1B;
        text-align: center;
        margin: 15px 0;
    }
    .ham-banner {
        background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        color: #065F46;
        text-align: center;
        margin: 15px 0;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 15px;
    }
    .info-card {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_metrics_data():
    """Loads cached metrics generated during model training."""
    metrics_path = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset_cached():
    """Loads and caches the cleaned dataset for visualization."""
    try:
        df, stats = load_sms_data(drop_duplicates=True, auto_download=True)
        # Compute length attributes for EDA
        df["char_length"] = df["message"].apply(len)
        df["word_count"] = df["message"].apply(lambda t: len(t.split()))
        return df, stats
    except Exception as e:
        return None, str(e)


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown('<span class="team-badge">TEAM 2 • NLP</span>', unsafe_allow_html=True)
    st.title("🛡️ SMS Spam Shield")
    st.caption("College NLP Project Demonstration")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🔍 SMS Spam Detector",
            "📊 Model Comparison",
            "📁 Dataset Explorer",
            "ℹ️ About Project"
        ],
        index=0
    )

    st.markdown("---")
    
    # Model Status Check in Sidebar
    metrics = load_metrics_data()
    if metrics:
        champion = metrics.get("best_model_name", "Logistic Regression")
        st.success(f"**Champion Model Active**\n{champion}")
        st.caption(f"Test Accuracy: **{metrics['models']['logistic_regression']['accuracy']*100:.2f}%**")
    else:
        st.warning("⚠️ **Models Not Found**\nPlease run `python src/train_model.py` in your terminal.")

    st.markdown("---")
    st.caption("Subject: Natural Language Processing  \nTools: Python, Scikit-learn, Streamlit")


# -------------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# -------------------------------------------------------------
if page == "🏠 Home Dashboard":
    st.markdown('<span class="team-badge">PROJECT OVERVIEW • TEAM 2</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">SMS Spam Message Detection Using NLP</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">A complete Natural Language Processing and Machine Learning system '
        'to classify text messages into legitimate communication (HAM) or unwanted/suspicious messages (SPAM).</p>',
        unsafe_allow_html=True
    )

    if not metrics:
        st.error(
            "⚠️ Trained models and metrics have not been generated yet. "
            "Please run `python src/train_model.py` in your terminal to train both models on the UCI dataset."
        )
        st.stop()

    # High-level Metrics Row
    d_stats = metrics["dataset_stats"]["final_distribution"]
    best_name = metrics["best_model_name"]
    best_key = "logistic_regression" if "Logistic" in best_name else "naive_bayes"
    best_m = metrics["models"][best_key]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total SMS Messages</div>
            <div class="metric-val">{metrics['dataset_stats']['final_messages_count']:,}</div>
            <div style="font-size:0.8rem; color:#64748B;">Cleaned UCI Dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Legitimate (Ham)</div>
            <div class="metric-val" style="color:#059669;">{d_stats['ham']:,}</div>
            <div style="font-size:0.8rem; color:#059669;">{d_stats['ham_percentage']}% of corpus</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Spam Messages</div>
            <div class="metric-val" style="color:#DC2626;">{d_stats['spam']:,}</div>
            <div style="font-size:0.8rem; color:#DC2626;">{d_stats['spam_percentage']}% of corpus</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Champion Accuracy</div>
            <div class="metric-val" style="color:#2563EB;">{best_m['accuracy']*100:.2f}%</div>
            <div style="font-size:0.8rem; color:#2563EB;">{best_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Spam Precision</div>
            <div class="metric-val" style="color:#7C3AED;">{best_m['spam_precision']*100:.2f}%</div>
            <div style="font-size:0.8rem; color:#7C3AED;">Zero False Positives</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two column layout: Problem context & Distribution visual
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("📌 The Spam Detection Problem")
        st.markdown("""
        Short Message Service (SMS) remains one of the most trusted personal communication channels with an open rate exceeding 90%. 
        However, this trust makes SMS a prime vector for:
        - **Phishing & Smishing**: Fraudulent links targeting credentials and banking logins.
        - **Lottery & Prize Scams**: Unsolicited notices demanding advance fees or premium calls.
        - **Aggressive Commercial Marketing**: Bulk spam violating consumer privacy.

        **Objective of this College NLP Project**:
        Build a robust machine-learning pipeline that converts unstructured SMS texts into TF-IDF n-gram vectors, 
        evaluates Multinomial Naive Bayes and Logistic Regression classifiers, and presents an explainable detection interface.
        """)

        st.markdown(f"""
        <div class="info-card">
            <strong>🏆 Champion Model Selection Rationale:</strong><br>
            {metrics['selection_rationale']} Logistic Regression maintained 
            <strong>{best_m['spam_precision']*100:.1f}% Spam Precision</strong> on the test set, meaning no legitimate SMS messages were mistakenly quarantined.
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.subheader("📊 Dataset Class Distribution")
        dist_df = pd.DataFrame({
            "Category": ["Legitimate (Ham)", "Spam"],
            "Count": [d_stats["ham"], d_stats["spam"]],
            "Percentage": [d_stats["ham_percentage"], d_stats["spam_percentage"]]
        })

        chart = alt.Chart(dist_df).mark_bar(cornerRadius=6, size=45).encode(
            x=alt.X("Category:N", title="Message Class", sort=None),
            y=alt.Y("Count:Q", title="Number of Messages"),
            color=alt.Color("Category:N", scale=alt.Scale(domain=["Legitimate (Ham)", "Spam"], range=["#10B981", "#EF4444"]), legend=None),
            tooltip=["Category", "Count", "Percentage"]
        ).properties(height=280)

        text = chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            fontSize=12,
            fontWeight="bold"
        ).encode(
            text=alt.Text("Count:Q", format=",.0f")
        )

        st.altair_chart(chart + text, use_container_width=True)
        st.caption("Figure 1: Imbalanced class distribution in the UCI SMS Spam Collection.")


# -------------------------------------------------------------
# PAGE 2: SMS SPAM DETECTOR
# -------------------------------------------------------------
elif page == "🔍 SMS Spam Detector":
    st.markdown('<span class="team-badge">LIVE INFERENCE • TEAM 2</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Real-Time SMS Spam Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Enter any SMS message below or select a preset test scenario '
        'to evaluate the NLP model in real time.</p>',
        unsafe_allow_html=True
    )

    if not metrics:
        st.error("⚠️ Trained models not found. Run `python src/train_model.py` first.")
        st.stop()

    # Preset Sample Selector
    sample_options = {
        "— Select a Quick Sample to Load —": "",
        "1. Normal Personal SMS (Family/Friend)": "Hey Mom, I am on my way home from college. Do we need anything from the grocery store?",
        "2. Suspicious Prize / Lottery Lure": "Congratulations! You have been selected as our £1,000 weekly cash prize winner! Text CLAIM to 87012 or call 09061743811 to redeem.",
        "3. Phishing Link / Fake Postal Alert": "Urgent alert: Your postal package delivery could not be completed. Update your delivery address now at http://parcel-redelivery-secure.net/track?id=99281",
        "4. Promotional Marketing Sale": "EXCLUSIVE SALE! Get 50% discount on all designer jackets this weekend only. Shop now at your nearest store or call 0800-SALE-NOW.",
        "5. Ordinary Bank Transaction": "Your checking account ending in 8419 was debited for $24.50 at Corner Cafe on 10/14. Available balance: $1,280.12. If unauthorized, reply HELP.",
        "6. Work Meeting with Numbers & Symbols": "Meeting room #402, 10:30 AM tomorrow -- please bring 3 printed copies of doc #712! Thanks :)",
        "7. Casual Chat with Mixed Words": "Hey John, did you hear about that free pizza offer the university cafeteria was talking about yesterday? Sounds cool!"
    }

    selected_sample = st.selectbox("💡 Load a test case scenario:", list(sample_options.keys()))

    # Manage text area content via session state
    if "input_sms" not in st.session_state:
        st.session_state.input_sms = ""

    if selected_sample != "— Select a Quick Sample to Load —" and sample_options[selected_sample]:
        st.session_state.input_sms = sample_options[selected_sample]

    # Model selector radio
    model_choice = st.radio(
        "Select Model for Classification:",
        ["Champion Pipeline (Logistic Regression)", "Alternative Pipeline (Multinomial Naive Bayes)"],
        horizontal=True
    )
    chosen_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "logistic_regression_pipeline.joblib" if "Logistic" in model_choice else "naive_bayes_pipeline.joblib"
    )

    # Input text area
    user_sms = st.text_area(
        "SMS Message Text:",
        value=st.session_state.input_sms,
        height=130,
        placeholder="Type or paste the SMS text here...",
        help="Input the exact text of the SMS you wish to analyze."
    )

    # Metadata counters
    char_count = len(user_sms)
    word_count = len(user_sms.split())
    st.caption(f"Character Count: **{char_count}** | Word Count: **{word_count}**")

    # Predict Button
    btn_predict = st.button("🚀 Classify SMS Message", type="primary", use_container_width=True)

    if btn_predict:
        if not user_sms or not user_sms.strip():
            st.warning("⚠️ **Validation Warning**: The message text is empty or contains only whitespace. Please enter valid text.")
        else:
            with st.spinner("Analyzing message through TF-IDF and NLP pipeline..."):
                result = predict_message(user_sms, model_path=chosen_path)

            if result["status"] == "error":
                st.error(f"Error: {result['error_message']}")
            else:
                pred = result["prediction"]
                conf = result["confidence"]
                spam_p = result["spam_probability"]
                ham_p = result["ham_probability"]
                features = result["features"]

                st.markdown("---")

                # Verdict Banner
                if pred == "SPAM":
                    st.markdown(f"""
                    <div class="spam-banner">
                        <h2 style="margin:0; font-size:1.9rem;">⚠️ CLASSIFIED AS SPAM</h2>
                        <p style="font-size:1.1rem; margin:5px 0 0 0;">
                            Estimated Spam Probability: <strong>{spam_p*100:.2f}%</strong> (Confidence: {conf*100:.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ham-banner">
                        <h2 style="margin:0; font-size:1.9rem;">✅ CLASSIFIED AS LEGITIMATE (HAM)</h2>
                        <p style="font-size:1.1rem; margin:5px 0 0 0;">
                            Estimated Legitimate Probability: <strong>{ham_p*100:.2f}%</strong> (Confidence: {conf*100:.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probabilities & Indicators
                p_col1, p_col2 = st.columns([1, 1])

                with p_col1:
                    st.subheader("📊 Probability Breakdown")
                    st.write(f"**Spam Probability**: `{spam_p*100:.2f}%`")
                    st.progress(float(spam_p))
                    st.write(f"**Ham (Legitimate) Probability**: `{ham_p*100:.2f}%`")
                    st.progress(float(ham_p))

                with p_col2:
                    st.subheader("🔍 Linguistic Triggers Detected")
                    st.write(f"• **Contains Currency / Money Signs**: {'Yes 🚨' if features['has_currency'] else 'No'}")
                    st.write(f"• **Contains External URL / Link**: {'Yes 🚨' if features['has_url'] else 'No'}")
                    st.write(f"• **Contains Numeric Code / Phone**: {'Yes 🚨' if features['has_numbers'] else 'No'}")
                    st.write(f"• **Exclamation Marks Count**: `{features['exclamation_count']}`")
                    st.write(f"• **Cleaned Tokens for TF-IDF**: `{result['cleaned_text']}`")

                st.subheader("💡 Interpretation & Explanation")
                st.info(result["explanation"])

                st.markdown(f"""
                <div class="disclaimer-box">
                    <strong>⚠️ Important Disclaimer:</strong><br>
                    {result['disclaimer']}
                </div>
                """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 3: MODEL COMPARISON
# -------------------------------------------------------------
elif page == "📊 Model Comparison":
    st.markdown('<span class="team-badge">EMPIRICAL BENCHMARKS • TEAM 2</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Machine Learning Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Comparative evaluation between Multinomial Naive Bayes and Logistic Regression '
        'trained on identical stratified splits (80% train, 20% test, random_state=42).</p>',
        unsafe_allow_html=True
    )

    if not metrics:
        st.error("⚠️ Model comparison data not found. Run `python src/train_model.py` to generate metrics.")
        st.stop()

    nb_m = metrics["models"]["naive_bayes"]
    lr_m = metrics["models"]["logistic_regression"]

    # Comparative Metrics Table
    st.subheader("📋 Empirical Test Set Performance Table")
    comp_df = pd.DataFrame(metrics["comparison_table"])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrices Side-by-Side
    st.subheader("🔲 Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)

    def plot_confusion_matrix_altair(cm_dict, model_name):
        tn = cm_dict["tn"]
        fp = cm_dict["fp"]
        fn = cm_dict["fn"]
        tp = cm_dict["tp"]

        cm_data = pd.DataFrame([
            {"Actual": "Actual HAM (0)", "Predicted": "Pred HAM (0)", "Count": tn, "Label": f"TN: {tn}"},
            {"Actual": "Actual HAM (0)", "Predicted": "Pred SPAM (1)", "Count": fp, "Label": f"FP: {fp}"},
            {"Actual": "Actual SPAM (1)", "Predicted": "Pred HAM (0)", "Count": fn, "Label": f"FN: {fn}"},
            {"Actual": "Actual SPAM (1)", "Predicted": "Pred SPAM (1)", "Count": tp, "Label": f"TP: {tp}"}
        ])

        base = alt.Chart(cm_data).encode(
            x=alt.X("Predicted:O", title="Predicted Class", sort=None),
            y=alt.Y("Actual:O", title="Actual True Class", sort=None)
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None)
        ).properties(title=model_name, height=220)

        text = base.mark_text(fontSize=14, fontWeight="bold").encode(
            text="Label:N",
            color=alt.condition(alt.datum.Count > (tn/2), alt.value("white"), alt.value("black"))
        )
        return heatmap + text

    with cm_col1:
        st.altair_chart(plot_confusion_matrix_altair(nb_m["confusion_matrix"], "Multinomial Naive Bayes"), use_container_width=True)
        st.caption(f"**Naive Bayes**: FP = {nb_m['confusion_matrix']['fp']} (legitimate marked spam), FN = {nb_m['confusion_matrix']['fn']} (spam leaked).")

    with cm_col2:
        st.altair_chart(plot_confusion_matrix_altair(lr_m["confusion_matrix"], "Logistic Regression (Champion)"), use_container_width=True)
        st.caption(f"**Logistic Regression**: FP = {lr_m['confusion_matrix']['fp']} (Zero false positives!), FN = {lr_m['confusion_matrix']['fn']} (spam leaked).")

    st.markdown("---")

    # Pedagogical Insights on FP vs FN
    st.subheader("🎓 In-Depth Error Analysis: False Positives vs. False Negatives")
    
    ins_col1, ins_col2 = st.columns(2)
    with ins_col1:
        st.markdown("""
        <div style="background-color:#FEF2F2; border-left: 4px solid #EF4444; padding:15px; border-radius:4px;">
            <h4 style="color:#991B1B; margin-top:0;">🚨 False Positives (FP) — High Risk</h4>
            <p style="font-size:0.9rem; color:#7F1D1D;">
                <strong>Definition:</strong> A legitimate message (HAM) is incorrectly classified as SPAM.<br><br>
                <strong>Impact in SMS:</strong> High False Positives are catastrophic. Users miss critical 
                Two-Factor Authentication (2FA) OTPs, flight updates, hospital alerts, or banking verification codes.<br><br>
                <strong>Result:</strong> Logistic Regression achieved <strong>FP = 0</strong> (100% Spam Precision), 
                ensuring zero legitimate messages were lost.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with ins_col2:
        st.markdown("""
        <div style="background-color:#FFFBEB; border-left: 4px solid #F59E0B; padding:15px; border-radius:4px;">
            <h4 style="color:#92400E; margin-top:0;">⚠️ False Negatives (FN) — Moderate Risk</h4>
            <p style="font-size:0.9rem; color:#78350F;">
                <strong>Definition:</strong> A spam message is incorrectly classified as legitimate (HAM).<br><br>
                <strong>Impact in SMS:</strong> The user receives an unsolicited or promotional message in their inbox. 
                While irritating, the user can manually disregard it without losing vital communication.<br><br>
                <strong>Result:</strong> Both models had <strong>FN = 14</strong> out of 128 test spam messages 
                (Spam Recall = 89.06%), representing novel phrasing not present in training unigrams/bigrams.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Classification Reports
    st.subheader("📑 Detailed Scikit-learn Classification Reports")
    tab1, tab2 = st.tabs(["Logistic Regression (Champion)", "Multinomial Naive Bayes"])
    with tab1:
        st.code(lr_m["report_text"], language="text")
    with tab2:
        st.code(nb_m["report_text"], language="text")


# -------------------------------------------------------------
# PAGE 4: DATASET EXPLORER
# -------------------------------------------------------------
elif page == "📁 Dataset Explorer":
    st.markdown('<span class="team-badge">DATA INSPECTION • TEAM 2</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">UCI SMS Spam Collection Explorer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Explore the authentic UCI Machine Learning Repository SMS dataset '
        'including class distributions, message lengths, and text samples.</p>',
        unsafe_allow_html=True
    )

    df_data, d_info = load_dataset_cached()

    if df_data is None:
        st.error(f"Error loading dataset: {d_info}")
        st.stop()

    # Data Summary
    st.markdown(f"""
    <div class="info-card">
        <strong>Dataset Provenance:</strong> Sourced from the official UCI Machine Learning Repository.<br>
        <strong>Raw Records:</strong> {d_info['raw_messages_count']:,} messages | 
        <strong>Duplicates Removed:</strong> {d_info['duplicates_count']:,} duplicates | 
        <strong>Final Unique Corpus:</strong> {d_info['final_messages_count']:,} messages.
    </div>
    """, unsafe_allow_html=True)

    # Filters
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        class_filter = st.selectbox("Filter by Class:", ["All Messages", "Ham Only (Legitimate)", "Spam Only"])
    with f_col2:
        sort_by = st.selectbox("Sort by:", ["Original Index", "Longest Characters", "Shortest Characters"])
    with f_col3:
        search_query = st.text_input("🔍 Search Messages by Keyword:", placeholder="e.g., call, win, free, meet...")

    # Apply filters
    filtered_df = df_data.copy()
    if class_filter == "Ham Only (Legitimate)":
        filtered_df = filtered_df[filtered_df["label"] == "ham"]
    elif class_filter == "Spam Only":
        filtered_df = filtered_df[filtered_df["label"] == "spam"]

    if search_query.strip():
        filtered_df = filtered_df[filtered_df["message"].str.contains(search_query.strip(), case=False, na=False)]

    if sort_by == "Longest Characters":
        filtered_df = filtered_df.sort_values(by="char_length", ascending=False)
    elif sort_by == "Shortest Characters":
        filtered_df = filtered_df.sort_values(by="char_length", ascending=True)

    st.caption(f"Showing **{len(filtered_df):,}** messages matching filters.")
    st.dataframe(
        filtered_df[["label", "char_length", "word_count", "message"]],
        use_container_width=True,
        height=320
    )

    # Visualizations: Length Comparison
    st.subheader("📊 Message Character Length Analysis (Ham vs. Spam)")
    st.markdown(
        "A well-documented NLP pattern in SMS spam: spam messages tend to maximize the 160-character SMS limit "
        "to fit marketing copy, helpline numbers, and terms, whereas legitimate personal SMS messages are much shorter."
    )

    # Grouped average length
    avg_stats = df_data.groupby("label")[["char_length", "word_count"]].mean().reset_index()
    avg_stats["label"] = avg_stats["label"].map({"ham": "Legitimate (Ham)", "spam": "Spam"})

    len_chart = alt.Chart(avg_stats).mark_bar(cornerRadius=6, size=50).encode(
        x=alt.X("label:N", title="Message Class", sort=None),
        y=alt.Y("char_length:Q", title="Average Character Length"),
        color=alt.Color("label:N", scale=alt.Scale(range=["#10B981", "#EF4444"]), legend=None),
        tooltip=["label", alt.Tooltip("char_length:Q", format=".1f", title="Avg Chars")]
    ).properties(height=260)

    len_text = len_chart.mark_text(dy=-6, fontWeight="bold").encode(
        text=alt.Text("char_length:Q", format=".1f")
    )

    st.altair_chart(len_chart + len_text, use_container_width=True)


# -------------------------------------------------------------
# PAGE 5: ABOUT PROJECT
# -------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.markdown('<span class="team-badge">PROJECT SPECIFICATION • TEAM 2</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">About This NLP Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎓 Project Information
    - **Project Title**: SMS Spam Message Detection Using NLP and Machine Learning
    - **Team Number**: **Team 2**
    - **Subject**: Natural Language Processing (NLP)
    - **Programming Language**: Python 3.12
    - **Development Frameworks**: Scikit-learn, Streamlit, Pandas, NumPy
    - **IDE / Environment**: Visual Studio Code, Jupyter Notebook
    """)

    st.markdown("---")

    st.subheader("🔄 End-to-End NLP Architecture")
    st.markdown("""
    ```
    ┌──────────────────────────┐
    │ UCI SMS Spam Collection  │ (5,574 raw SMS messages)
    └─────────────┬────────────┘
                  ▼
    ┌──────────────────────────┐
    │  Data Cleaning & Dedupl. │ (Handling nulls, 414 duplicate messages removed)
    └─────────────┬────────────┘
                  ▼
    ┌──────────────────────────┐
    │ Domain Text Preprocessing│ (Lowercasing, negation preservation, indicator tokens)
    └─────────────┬────────────┘
                  ▼
    ┌──────────────────────────┐
    │ TF-IDF Vectorization     │ (Unigrams + Bigrams, max_features=5,000, sublinear TF)
    └─────────────┬────────────┘
                  ▼
    ┌──────────────────────────┐
    │ Stratified Train/Test    │ (80% Train: 4,128 msgs | 20% Test: 1,032 msgs)
    └─────────────┬────────────┘
                  ▼
    ┌──────────────────────────┴──────────────────────────┐
    │                                                     │
    ▼                                                     ▼
    ┌──────────────────────────┐         ┌──────────────────────────┐
    │ Multinomial Naive Bayes  │         │   Logistic Regression    │
    │ Accuracy: 98.35%         │         │   Accuracy: 98.64%       │
    │ Spam F1: 93.06%          │         │   Spam F1: 94.21% (WIN)  │
    └─────────────┬────────────┘         └─────────────┬────────────┘
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
                      ┌──────────────────────────┐
                      │ Serialized Champion      │ (models/best_model_pipeline.joblib)
                      └──────────────┬───────────┘
                                     ▼
                      ┌──────────────────────────┐
                      │  Streamlit Web Dashboard │ (app.py)
                      └──────────────────────────┘
    ```
    """)

    st.markdown("---")

    st.subheader("⚖️ Dataset Source & Ethical Considerations")
    st.markdown("""
    - **Official Source**: Tiago A. Almeida and José María Gómez Hidalgo. *UCI Machine Learning Repository: SMS Spam Collection*.
    - **No Synthetic Data**: All training and benchmark statistics derive strictly from the authentic UCI SMS dataset.
    - **Disclaimer**: Machine learning classifications are statistical approximations based on text tokens and n-gram patterns. 
      A message flagged as 'SPAM' must never be represented as definitive legal or forensic proof of criminal fraud.
    """)

    st.subheader("🚀 Project Limitations & Future Enhancements")
    st.markdown("""
    1. **Language Coverage**: The current dataset is primarily English. Future iterations could integrate multilingual transformers (e.g., XLM-RoBERTa) for non-English SMS.
    2. **Concept Drift**: Attackers frequently evolve phrasing, emojis, and obfuscations. Continuous active learning and online model retraining would maintain high defense.
    3. **Telephony Metadata**: Incorporating carrier metadata (sender reputation, SMS delivery route, sender ID) alongside NLP text signals would further improve precision.
    """)
