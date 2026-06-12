import streamlit as st
import pandas as pd
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re, json
import time

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Clinical Semantic Search System",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Database connection
# -----------------------------
@st.cache_resource
def init_connection():
    return psycopg2.connect(
        dbname="clinical_db",
        user="postgres",
        password="",
        host="localhost",
        port=""
    )

conn = init_connection()
cur = conn.cursor()

# -----------------------------
# Load embedding model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# LLM (cached)
# -----------------------------
@st.cache_resource
def load_llm():
    return Ollama(
        model="gemma3:4b",
        temperature=0.3
    )

llm = load_llm()

# -----------------------------
# Processing function
# -----------------------------
def process_data_for_embedding(df):
    retrieval_texts = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                parts.append(f"{col}: {row[col]}")
        retrieval_texts.append(" | ".join(parts))
    return retrieval_texts

def embed_dataframe(df):
    with st.spinner("Generating embeddings..."):
        retrieval_texts = process_data_for_embedding(df)
        embeddings = model.encode(retrieval_texts)
        
        progress_bar = st.progress(0)
        for i, (text, emb) in enumerate(zip(retrieval_texts, embeddings)):
            vector = "[" + ",".join(map(str, emb)) + "]"
            cur.execute("""
            INSERT INTO codelist_4data_vectors      
            (retrieval_text, embedding)
            VALUES (%s, %s::vector)
            """, (text, vector))
            progress_bar.progress((i + 1) / len(retrieval_texts))
        
        conn.commit()
        st.success(f"✅ Successfully stored {len(retrieval_texts)} embeddings!")

def normalize(x):
    x = np.array(x)
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val == min_val:
        return np.ones_like(x)
    return (x - min_val) / (max_val - min_val)

# -----------------------------
# Sidebar for configuration
# -----------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload dataset", 
        type=["txt", "csv", "xlsx"],
        help="Upload your clinical dataset in TXT, CSV, or Excel format"
    )
    
    st.header("🔍 Search Settings")
    k = st.slider(
        "Number of initial candidates per query",
        min_value=1,
        max_value=50,
        value=20,
        help="Number of results to retrieve per expanded query"
    )
    
    st.header("🎯 Ranking Weights")
    col1, col2 = st.columns(2)
    with col1:
        semantic_weight = st.slider(
            "Semantic weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    with col2:
        tfidf_weight = st.slider(
            "TF-IDF weight",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05
        )
    
    bm25_weight = 1 - semantic_weight - tfidf_weight
    st.info(f"BM25 weight: {bm25_weight:.2f}")
    
    st.header("ℹ️ About")
    st.markdown("""
    This system uses:
    - **pgvector db** PostgreSQL to store index, and search high-dimensional vectors
    - **Semantic Search** (Sentence Transformers)
    - **TF-IDF** for keyword matching
    - **BM25** for text ranking
    - **LLM** for query expansion & explanation
    """)

# -----------------------------
# Main content
# -----------------------------
st.title("🏥 Code List Search System")
st.markdown("---")

# -----------------------------
# File upload and embedding
# -----------------------------
if uploaded_file:
    # Load data based on file type
    if uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        df = pd.DataFrame({"text": content.split("\n")})
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    
    # Display data preview
    with st.expander("📊 Dataset Preview", expanded=True):
        st.dataframe(df.head(10))
        st.caption(f"Total rows: {len(df)} | Columns: {', '.join(df.columns)}")
    
    # Embedding button
    if st.button("🚀 Generate Embeddings & Store", type="primary"):
        embed_dataframe(df)
    
    st.markdown("---")

# -----------------------------
# Search interface
# -----------------------------
st.header("🔍 Clinical Search")

# Query input with better layout
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Enter clinical search query",
        placeholder="e.g., 'diabetes type 2 treatment guidelines'",
        key="search_query"
    )
with col2:
    search_button = st.button("🔎 Search", type="primary", use_container_width=True)

# Search execution
if search_button and query:
    with st.spinner("Processing your query..."):
        # -----------------------------
        # Query Expansion
        # -----------------------------
        with st.status("🔍 Searching...", expanded=True) as status:
            st.write("📝 Expanding query...")
            expansion_prompt = f"""
            You are a clinical terminology expert.
            
            Generate alternative medical search queries with different characteristics:
            1. Exact synonyms
            2. Clinical abbreviations
            3. Related broader concepts
            4. Specific subtypes
            5. uncertainty: what a human analyst should check
            6. Common misspellings or variations
            7. Be explicit if the evidence is weak or ambiguous
    
            Original query: {query}
    
            Return as numbered list:
            """
            
            expanded = llm.invoke(expansion_prompt)
            
            expanded_queries = [query]
            for line in expanded.split("\n"):
                q = line.strip()
                if q and not q.startswith("1.") and not q.startswith("2."):
                    expanded_queries.append(q)
            
            # Display expanded queries
            with st.expander("📋 Expanded Queries", expanded=False):
                st.write("**Original:**", query)
                st.write("**Expanded:**")
                for i, q in enumerate(expanded_queries[1:], 1):
                    st.write(f"{i}. {q}")
            
            # -----------------------------
            # Vector Search
            # -----------------------------
            st.write("🔎 Performing vector search...")
            all_candidates = []
            
            progress_bar = st.progress(0)
            for idx, q in enumerate(expanded_queries):
                q_embedding = model.encode([q])[0]
                vector_str = "[" + ",".join(map(str, q_embedding)) + "]"
                
                cur.execute("""
                SELECT retrieval_text,
                embedding <=> %s::vector AS distance
                FROM codelist_4data_vectors   
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """, (vector_str, vector_str, k))
                
                rows = cur.fetchall()
                all_candidates.extend(rows)
                progress_bar.progress((idx + 1) / len(expanded_queries))
            
            # -----------------------------
            # Remove duplicates
            # -----------------------------
            unique_dict = {}

            for text, dist in all_candidates:
                if text not in unique_dict or dist < unique_dict[text]:
                    unique_dict[text] = dist
            unique_texts = list(unique_dict.keys())

            #for text, dist in all_candidates:
                #if text not in unique_dict:
                    #unique_dict[text] = dist
            
            #unique_texts = list(unique_dict.keys())
            distances = list(unique_dict.values())
            
            st.write(f"✅ Found {len(unique_texts)} unique results")
            
            # -----------------------------
            # TF-IDF
            # -----------------------------
            st.write("📊 Computing TF-IDF scores...")
            tfidf = TfidfVectorizer(max_features=5000)
            tfidf_matrix = tfidf.fit_transform(unique_texts)
            query_vec = tfidf.transform([query])
            scores_tfidf = cosine_similarity(query_vec, tfidf_matrix)[0]
            tfidf_nom = normalize(scores_tfidf)
            
            # -----------------------------
            # BM25
            # -----------------------------
            st.write("📈 Computing BM25 scores...")
            tokenized_corpus = [text.split() for text in unique_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(query.split())
            bm25_norm = normalize(bm25_scores)
            
            # -----------------------------
            # Hybrid Ranking
            # -----------------------------
            st.write("🎯 Computing hybrid ranking...")
            #semantic_scores = np.array([1 - distances[text] for text in unique_texts])
            #semantic_norm = normalize(semantic_scores)
            results = []
            for i, text in enumerate(unique_texts):
                semantic_score = 1 - distances[i]  # Convert distance to similarity
                
                final_score = (
                    semantic_weight * semantic_score +
                    tfidf_weight * scores_tfidf[i] +
                    bm25_weight * bm25_norm[i]
                )
                
                results.append({
                    "retrieval_text": text,
                    "semantic_score": f"{semantic_score:.3f}",
                    "tfidf_score": f"{scores_tfidf[i]:.3f}",
                    "bm25_score": f"{bm25_norm[i]:.3f}",
                    "final_score": final_score
                })
            
            results = sorted(results, key=lambda x: float(x["final_score"]), reverse=True)
            status.update(label="Search complete!", state="complete")
            
             

        # -----------------------------
        # Display Results
        # -----------------------------
        st.markdown("---")
        st.header("📊 Search Results")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Ranked Results", "📈 Scores Breakdown", "🤖 LLM Analysis"])
        
        with tab1:
            # Display results in a nice dataframe
            results_df = pd.DataFrame(results[:10])
            results_df.index = range(1, len(results_df) + 1)
            
            st.dataframe(
                results_df,
                column_config={
                    "retrieval_text": st.column_config.TextColumn("Clinical Information", width="large"),
                    "semantic_score": st.column_config.NumberColumn("Semantic", format="%.3f"),
                    "tfidf_score": st.column_config.NumberColumn("TF-IDF", format="%.3f"),
                    "bm25_score": st.column_config.NumberColumn("BM25", format="%.3f"),
                    "final_score": st.column_config.NumberColumn("Final Score", format="%.3f")
                },
                use_container_width=True
            )
        
        with tab2:
            # Visualization of scores
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            top_results = results[:10]
            indices = range(1, len(top_results) + 1)
            
            semantic_scores = [float(r["semantic_score"]) for r in top_results]
            tfidf_scores = [float(r["tfidf_score"]) for r in top_results]
            bm25_scores = [float(r["bm25_score"]) for r in top_results]
            
            ax.plot(indices, semantic_scores, 'o-', label='Semantic', linewidth=2, markersize=8)
            ax.plot(indices, tfidf_scores, 's-', label='TF-IDF', linewidth=2, markersize=8)
            ax.plot(indices, bm25_scores, '^-', label='BM25', linewidth=2, markersize=8)
            
            ax.set_xlabel('Rank', fontsize=12)
            ax.set_ylabel('Normalized Score', fontsize=12)
            ax.set_title('Score Distribution by Ranking Method', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with tab3:
            with st.spinner("Generating AI explanation..."):

                # -----------------------------
                # LLM Explanation
                # -----------------------------
                
                all_rows = []
                for r in results[:10]:

                # Top 10 results
                    prompt = f"""
                    You are assisting NICE analysts.

                    Retrieval text: {r['retrieval_text']}

                    Research question:
                    {query}

                    Return ONLY JSON.

                    {{
                      "code": "",
                      "term": "",
                      "source": "",
                      "type": "",
                      "Explain why this code is relevant and summarise":
                    }}

                    """
                    response = llm.invoke(prompt)
                    text = response if isinstance(response, str) else response.content

                    blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.S)

                    for block in blocks:
                        try:
                            data = json.loads(block)
                            if isinstance(data, list):
                                all_rows.extend(data)
                            elif isinstance(data, dict):
                                all_rows.append(data)

                        except Exception as e:
                            print("Parse error:", e)
                # convert into a pandas dataframe
                df = pd.DataFrame(all_rows)
                        
                st.dataframe(df)



                   
                 
                    

                    #explanation = llm.invoke(prompt)

                    #st.markdown("### Analysis")
                    #st.markdown("-----")
                    #st.markdown("Explanation:")
                    #st.markdown(explanation)
                    #st.markdown("-----\n")
                
                
                #st.markdown("### Analysis")
                #st.markdown(response)
                
                # Add feedback section
                st.markdown("---")
                st.markdown("### 📝 Feedback")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful"):
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 Not Helpful"):
                        st.info("We'll work on improving the results!")

elif search_button and not query:
    st.warning("⚠️ Please enter a search query before clicking Search.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Clinical Code List Semantic Search System | Powered by AI</div>",
    unsafe_allow_html=True
)