# app.py
import streamlit as st
from dotenv import load_dotenv
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load .env (API keys)
load_dotenv()

# Import helpers
from langchain_config import (
    get_summary_cached_module,
    get_news_articles,
    estimate_tokens,
    clear_module_cache,
)

# ------------------ Streamlit page config ------------------------
st.set_page_config(
    page_title="Equity Research AI Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ------------------ Custom CSS for professional styling ------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .summary-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .article-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Sidebar: Professional Controls ------------------
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/color/96/analytics.png", width=80)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("## 🔧 Analysis Controls")
    
    st.markdown("### 📊 Data Parameters")
    max_articles = st.slider("Maximum Articles", 5, 50, 15, 5, 
                           help="Number of articles to analyze")
    
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Start Date", 
                                value=datetime.now().date().replace(day=1))
    with col2:
        date_to = st.date_input("End Date")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Settings")
    
    cache_ttl_minutes = st.number_input("Cache Duration (minutes)", 
                                      min_value=1, max_value=1440, value=60,
                                      help="How long to cache results")
    
    if st.button("🔄 Clear System Cache", use_container_width=True):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        clear_module_cache()
        st.success("System cache cleared successfully")
    
    st.markdown("---")
    st.markdown("#### 💡 Usage Tips")
    st.markdown("""
    - Use specific queries for better results
    - Limit articles to control API costs
    - Check token estimates before analysis
    """)

# ------------------ Cached Wrapper ------------------
@st.cache_data(ttl=60 * 60 * 24)
def get_summary_cached_ui(query: str, max_articles: int) -> str:
    return get_summary_cached_module(query, max_articles)

# ------------------ Professional Header ------------------
st.markdown("<h1 class='main-header'>📈 Equity Research AI Analyst</h1>", unsafe_allow_html=True)

# Search section with improved layout
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "**Research Query**",
        placeholder="e.g., Tesla Q3 2024 earnings, Renewable energy sector trends, Federal Reserve policy impact...",
        help="Enter company names, sectors, economic events, or market trends"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_pressed = st.button("🚀 Analyze", use_container_width=True)

# ------------------ Quick Stats Row ------------------
if run_pressed and query:
    st.markdown("---")
    st.markdown("## 📊 Analysis Overview")
    
    # Placeholder for metrics - will be updated after analysis
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Articles Processed</h3>
            <h2>--</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Sources Analyzed</h3>
            <h2>--</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Time Period</h3>
            <h2>-- days</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Token Usage</h3>
            <h2>--</h2>
        </div>
        """, unsafe_allow_html=True)

# ------------------ Enhanced Tabs Layout ------------------
if run_pressed and query.strip():
    tabs = st.tabs(["📋 Executive Summary", "📰 News Articles", "📊 Market Analytics", "🕒 Research History"])

    articles, summary = [], None

    try:
        with st.spinner("🔍 Gathering market intelligence..."):
            articles = get_news_articles(query, max_articles=max_articles)
    except Exception as e:
        st.error(f"❌ Data collection failed: {str(e)}")
        logger.exception("NewsAPI fetch failed")
        articles = []

    if articles:
        # Update metrics
        concat_text = "\n\n".join(
            [(a.get("title") or "") + " — " + (a.get("description") or "") for a in articles]
        )
        tokens_est = estimate_tokens(concat_text) if concat_text else 0
        
        # Update metrics display
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Articles Processed</h3>
                <h2>{len(articles)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            sources = len(set(a.get("source", {}).get("name", "Unknown") for a in articles))
            st.markdown(f"""
            <div class="metric-card">
                <h3>Sources Analyzed</h3>
                <h2>{sources}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            dates = [a.get("publishedAt", "")[:10] for a in articles if a.get("publishedAt")]
            date_range = len(set(dates)) if dates else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Time Period</h3>
                <h2>{date_range} days</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Token Usage</h3>
                <h2>~{tokens_est}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Get summary
        try:
            with st.spinner("🤖 Generating executive insights..."):
                summary = get_summary_cached_ui(query, max_articles)
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.exception("Summarization error")

        # Save to history
        if summary:
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                "query": query, 
                "summary": summary,
                "articles_count": len(articles),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

    # ------------------ Tab 1: Executive Summary ------------------
    with tabs[0]:
        if summary:
            st.markdown("## 📋 Executive Summary")
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.download_button(
                    "📥 Download Report",
                    data=summary,
                    file_name=f"equity_research_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    use_container_width=True
                )
            with col2:
                if st.button("📧 Share Insights", use_container_width=True):
                    st.info("Share functionality would be implemented here")
            
            st.markdown("### 🔑 Key Insights")
            # Placeholder for key insights extraction
            st.info("Key insights extraction feature coming soon...")

    # ------------------ Tab 2: News Articles ------------------
    with tabs[1]:
        if articles:
            st.markdown(f"## 📰 Market News ({len(articles)} articles)")
            
            search_col, filter_col = st.columns([2, 1])
            with search_col:
                article_search = st.text_input("🔍 Filter articles...")
            with filter_col:
                source_filter = st.selectbox("Source", ["All"] + list(set(a.get("source", {}).get("name", "Unknown") for a in articles)))
            
            filtered_articles = articles
            if article_search:
                filtered_articles = [a for a in filtered_articles if article_search.lower() in (a.get("title") or "").lower()]
            if source_filter != "All":
                filtered_articles = [a for a in filtered_articles if a.get("source", {}).get("name") == source_filter]
            
            for i, article in enumerate(filtered_articles[:20]):
                with st.container():
                    title = article.get("title") or "Untitled"
                    source = article.get("source", {}).get("name") or "Unknown Source"
                    url = article.get("url", "#")
                    description = article.get("description", "No description available")
                    published = article.get("publishedAt", "")[:10]
                    
                    st.markdown(f"""
                    <div class="article-card">
                        <h4><a href="{url}" target="_blank">{title}</a></h4>
                        <p><strong>Source:</strong> {source} | <strong>Date:</strong> {published}</p>
                        <p>{description}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # ------------------ Tab 3: Market Analytics ------------------
    with tabs[2]:
        if articles:
            st.markdown("## 📊 Market Analytics")
            
            # Build dataframe
            df_data = []
            for a in articles:
                source = a.get("source", {}).get("name") or "Unknown"
                date = a.get("publishedAt", "")[:10] if a.get("publishedAt") else "Unknown"
                df_data.append({"source": source, "date": date, "title": a.get("title", "")})
            
            df = pd.DataFrame(df_data)
            
            # Analytics in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📌 Media Coverage Distribution")
                if not df.empty and "source" in df.columns:
                    source_counts = df["source"].value_counts()
                    fig = px.pie(values=source_counts.values, names=source_counts.index,
                               color_discrete_sequence=px.colors.sequential.Blues)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### ⏳ News Timeline")
                if not df.empty and "date" in df.columns and not df["date"].isnull().all():
                    timeline = df["date"].value_counts().sort_index()
                    fig = px.line(x=timeline.index, y=timeline.values,
                                markers=True, line_shape='spline')
                    fig.update_layout(xaxis_title="Date", yaxis_title="Articles Published")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Additional analytics
            st.markdown("#### 📈 Coverage Metrics")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                if not df.empty:
                    avg_articles = len(articles) / max(len(df['date'].unique()), 1)
                    st.metric("Average Articles per Day", f"{avg_articles:.1f}")
            
            with col4:
                if not df.empty and "source" in df.columns:
                    st.metric("Unique Sources", len(df['source'].unique()))
            
            with col5:
                st.metric("Analysis Period", f"{len(df['date'].unique())} days")

    # ------------------ Tab 4: Research History ------------------
    with tabs[3]:
        st.markdown("## 🕒 Research History")
        
        if "history" in st.session_state and st.session_state.history:
            for i, research in enumerate(st.session_state.history[:10]):
                with st.expander(f"🔍 {research['query']} - {research.get('timestamp', 'Recent')}"):
                    st.markdown(f"**Summary:**")
                    st.write(research["summary"])
                    st.markdown(f"**Articles:** {research.get('articles_count', 'N/A')} | **Saved:** {research.get('timestamp', 'Unknown')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🔄 Re-run", key=f"rerun_{i}"):
                            st.session_state.re_run_query = research['query']
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                            st.session_state.history.pop(i)
                            st.rerun()
        else:
            st.info("No research history available. Complete your first analysis to see history here.")

elif run_pressed and not query.strip():
    st.warning("⚠️ Please enter a research query to begin analysis.")

# ------------------ Professional Footer ------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚡ Powered by Advanced AI Analytics | Built for Professional Equity Research</p>
    <p>🔒 Secure • Fast • Intelligent Market Insights</p>
</div>
""", unsafe_allow_html=True)

# Handle re-run queries
if hasattr(st.session_state, 're_run_query'):
    query = st.session_state.re_run_query
    del st.session_state.re_run_query
    st.rerun()