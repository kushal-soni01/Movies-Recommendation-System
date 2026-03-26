import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -------------------------
# PAGE CONFIG
# -------------------------


load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🍿 AI Movie Recommendation System")
st.markdown("Type a movie, genre, actor, or mood and get smart recommendations!")

# -------------------------
# LOAD SYSTEM (CACHED)
# -------------------------
@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="db",   # MUST match notebook
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a movie recommendation system.

Use the context below to recommend movies:

{context}

Return output in this format:
1. Movie Name - short reason
2. Movie Name - short reason
3. Movie Name - short reason
4. Movie Name - short reason
5. Movie Name - short reason
"""),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain

chain = load_system()

# -------------------------
# USER INPUT
# -------------------------
query = st.text_input("🔍 Enter your movie preference:")

# -------------------------
# BUTTON ACTION
# -------------------------
if st.button("Recommend 🎬"):
    if query.strip() != "":
        with st.spinner("Finding perfect movies for you... 🍿"):
            response = chain.invoke({"input": query})

            st.subheader("🎯 Recommended Movies")

            results = response['answer'].split("\n")

            # -------------------------
            # NETFLIX STYLE CARDS 🎬
            # -------------------------
            for movie in results:
                if movie.strip():
                    st.markdown(f"""
                    <div style="
                        padding:15px;
                        border-radius:12px;
                        margin:10px 0;
                        background-color:#1e1e1e;
                        color:white;
                        font-size:16px;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
                    ">
                        {movie}
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter something to get recommendations!")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("Built with ❤️ using LangChain + Groq + Streamlit")