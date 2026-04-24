import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parcl Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)

# ── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    clients = pd.read_csv('clients_cleaned.csv')
    props   = pd.read_csv('properties_cleaned.csv')
    clients['satisfaction_score'] = pd.to_numeric(
        clients['satisfaction_score'], errors='coerce')
    clients['satisfaction_score'].fillna(
        clients['satisfaction_score'].median(), inplace=True)
    return clients, props

clients, props = load_data()

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🏠 Parcl Co. Limited — Buyer Segmentation Dashboard")
st.markdown("**ML-Based Buyer Segmentation & Investment Profiling | Aviral | BML Munjal University MBA 2025–27**")
st.markdown("---")

# ── Sidebar Filters ─────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("Use these to filter all charts")

all_countries = ['All'] + sorted(clients['country'].unique().tolist())
selected_country = st.sidebar.selectbox("Country", all_countries)

all_purposes = ['All', 'Home', 'Investment']
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", all_purposes)

all_types = ['All', 'Individual', 'Company']
selected_type = st.sidebar.selectbox("Client Type", all_types)

all_segments = ['All'] + sorted(clients['segment'].unique().tolist())
selected_segment = st.sidebar.selectbox("Segment", all_segments)

# ── Apply Filters ────────────────────────────────────────────────────────────
filtered = clients.copy()
if selected_country  != 'All': filtered = filtered[filtered['country']            == selected_country]
if selected_purpose  != 'All': filtered = filtered[filtered['acquisition_purpose'] == selected_purpose]
if selected_type     != 'All': filtered = filtered[filtered['client_type']         == selected_type]
if selected_segment  != 'All': filtered = filtered[filtered['segment']             == selected_segment]

# ── KPI Cards ────────────────────────────────────────────────────────────────
st.subheader("📊 Key Numbers at a Glance")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Clients",     f"{len(filtered):,}")
k2.metric("Investment Buyers", f"{(filtered['acquisition_purpose']=='Investment').sum():,}")
k3.metric("Home Buyers",       f"{(filtered['acquisition_purpose']=='Home').sum():,}")
k4.metric("Took a Loan",       f"{(filtered['loan_applied']=='Yes').sum():,}")
k5.metric("Avg Satisfaction",  f"{filtered['satisfaction_score'].mean():.2f} / 5")
st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Segmentation Overview",
    "💰 Investor Behavior",
    "🌍 Geographic Analysis",
    "🔍 Segment Insights"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEGMENTATION OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Buyer Segmentation Overview")
    st.write("Distribution of buyers across segments, purchase purpose, loan usage, and client type.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Distribution")
        seg = filtered['segment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(seg.values,
               labels=seg.index,
               autopct='%1.1f%%',
               colors=['#000000','#333333','#666666','#999999','#CCCCCC'],
               startangle=140,
               textprops={'fontsize': 9})
        ax.set_title("Buyer Segments", fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Home vs Investment Buyers")
        purpose = filtered['acquisition_purpose'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(purpose.index, purpose.values,
                      color=['#000000','#888888'], edgecolor='white', width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 8,
                    str(int(bar.get_height())),
                    ha='center', fontweight='bold', fontsize=11)
        ax.set_title("Acquisition Purpose", fontweight='bold')
        ax.set_ylabel("Number of Buyers")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Loan vs Cash Buyers")
        loan = filtered['loan_applied'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(loan.index, loan.values,
                      color=['#111111','#AAAAAA'], edgecolor='white', width=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 5,
                    str(int(bar.get_height())),
                    ha='center', fontweight='bold')
        ax.set_title("Loan Applied — Yes vs No", fontweight='bold')
        ax.set_ylabel("Number of Buyers")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Individual vs Company")
        ctype = filtered['client_type'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(ctype.index, ctype.values,
                      color=['#111111','#AAAAAA'], edgecolor='white', width=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 5,
                    str(int(bar.get_height())),
                    ha='center', fontweight='bold')
        ax.set_title("Client Type", fontweight='bold')
        ax.set_ylabel("Number of Buyers")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — INVESTOR BEHAVIOR
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Investor Behavior Dashboard")
    st.write("How different segments behave in terms of satisfaction, referral channels, and score distribution.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Avg Satisfaction by Segment")
        sat = filtered.groupby('segment')['satisfaction_score'].mean().round(2)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(sat.index, sat.values,
                      color='#000000', edgecolor='white', width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.03,
                    f"{bar.get_height():.2f}",
                    ha='center', fontsize=9)
        ax.set_title("Avg Satisfaction Score by Segment", fontweight='bold')
        ax.set_ylabel("Score (1–5)")
        ax.set_ylim(0, 5.8)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Referral Channel Breakdown")
        ref = filtered['referral_channel'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(ref.index, ref.values,
                       color=['#000000','#555555','#AAAAAA'], edgecolor='white')
        for bar in bars:
            ax.text(bar.get_width() + 5,
                    bar.get_y() + bar.get_height()/2,
                    str(int(bar.get_width())),
                    va='center', fontweight='bold')
        ax.set_title("How Buyers Found Parcl", fontweight='bold')
        ax.set_xlabel("Number of Buyers")
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

    st.subheader("Satisfaction Score Distribution (1 = Very Unhappy → 5 = Very Happy)")
    sat_dist = filtered['satisfaction_score'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(sat_dist.index.astype(str), sat_dist.values,
                  color='#222222', edgecolor='white', width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 3,
                str(int(bar.get_height())),
                ha='center', fontweight='bold')
    ax.set_title("Distribution of Satisfaction Scores", fontweight='bold')
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of Buyers")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig)
    plt.close()

    st.subheader("Loan Usage by Segment")
    loan_seg = filtered.groupby('segment').apply(
        lambda x: (x['loan_applied'] == 'Yes').mean() * 100).round(1)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(loan_seg.index, loan_seg.values,
                  color='#444444', edgecolor='white', width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha='center', fontsize=9)
    ax.set_title("Loan Usage % by Segment", fontweight='bold')
    ax.set_ylabel("Percentage (%)")
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Geographic Buyer Analysis")
    st.write("Where buyers are coming from — by country and region.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Countries")
        country = filtered['country'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.barh(country.index[::-1], country.values[::-1],
                       color='#000000', edgecolor='white')
        for bar in bars:
            ax.text(bar.get_width() + 5,
                    bar.get_y() + bar.get_height()/2,
                    str(int(bar.get_width())),
                    va='center', fontsize=9)
        ax.set_title("Buyers by Country", fontweight='bold')
        ax.set_xlabel("Number of Buyers")
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Top 10 Regions")
        region = filtered['region'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.barh(region.index[::-1], region.values[::-1],
                       color='#555555', edgecolor='white')
        for bar in bars:
            ax.text(bar.get_width() + 2,
                    bar.get_y() + bar.get_height()/2,
                    str(int(bar.get_width())),
                    va='center', fontsize=9)
        ax.set_title("Buyers by Region", fontweight='bold')
        ax.set_xlabel("Number of Buyers")
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

    st.subheader("Home vs Investment by Top 6 Countries")
    top6 = filtered['country'].value_counts().head(6).index
    df_t6 = filtered[filtered['country'].isin(top6)]
    grp = df_t6.groupby(['country','acquisition_purpose']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(grp.index))
    width = 0.35
    for i, col in enumerate(grp.columns):
        clr = '#000000' if i == 0 else '#999999'
        offset = (i - 0.5) * width
        bars = ax.bar([xi + offset for xi in x], grp[col],
                      width=width, label=col, color=clr, edgecolor='white')
    ax.set_xticks(list(x))
    ax.set_xticklabels(grp.index, rotation=15)
    ax.set_title("Purpose Split by Top 6 Countries", fontweight='bold')
    ax.set_ylabel("Number of Buyers")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig)
    plt.close()

    st.subheader("Gender Split by Country (Top 6)")
    if 'gender' in filtered.columns:
        grp_g = df_t6.groupby(['country','gender']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(11, 4))
        x = range(len(grp_g.index))
        for i, col in enumerate(grp_g.columns):
            clr = '#000000' if i == 0 else '#AAAAAA'
            offset = (i - 0.5) * 0.35
            ax.bar([xi + offset for xi in x], grp_g[col],
                   width=0.35, label=col, color=clr, edgecolor='white')
        ax.set_xticks(list(x))
        ax.set_xticklabels(grp_g.index, rotation=15)
        ax.set_title("Gender Distribution by Country", fontweight='bold')
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — SEGMENT INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Segment Insights Panel")
    st.write("Detailed statistics and profiles for each buyer segment.")

    summary = filtered.groupby('segment').agg(
        Total_Buyers     = ('client_id',           'count'),
        Avg_Age          = ('age',                 'mean'),
        Avg_Satisfaction = ('satisfaction_score',  'mean'),
        Loan_Pct         = ('loan_applied',         lambda x: round((x=='Yes').mean()*100,1)),
        Investment_Pct   = ('acquisition_purpose',  lambda x: round((x=='Investment').mean()*100,1)),
        Top_Country      = ('country',              lambda x: x.value_counts().index[0]),
        Top_Channel      = ('referral_channel',     lambda x: x.value_counts().index[0]),
    ).round(1).reset_index()

    st.subheader("Segment Summary Table")
    st.dataframe(summary, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Age by Segment")
        age_s = filtered.groupby('segment')['age'].mean().round(1)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(age_s.index, age_s.values,
                      color='#000000', edgecolor='white', width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}",
                    ha='center', fontsize=9)
        ax.set_title("Average Age by Segment", fontweight='bold')
        ax.set_ylabel("Age (years)")
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Investment Purpose % by Segment")
        inv_s = filtered.groupby('segment').apply(
            lambda x: (x['acquisition_purpose']=='Investment').mean()*100).round(1)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(inv_s.index, inv_s.values,
                      color='#555555', edgecolor='white', width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%",
                    ha='center', fontsize=9)
        ax.set_title("Investment Purpose % by Segment", fontweight='bold')
        ax.set_ylabel("Percentage (%)")
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        plt.close()

    st.subheader("View Raw Data")
    st.write(f"Showing {len(filtered):,} rows based on your current filters")
    st.dataframe(filtered[[
        'client_id','client_type','age','gender',
        'country','region','acquisition_purpose',
        'satisfaction_score','loan_applied',
        'referral_channel','segment'
    ]].reset_index(drop=True), use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*Aviral  |  MBA 2025–27  |  BML Munjal University  |  Parcl Co. Limited x Unified Mentor*")
