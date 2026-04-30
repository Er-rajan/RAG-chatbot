import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

load_dotenv()

st.set_page_config(
    page_title="DocChat AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
.stApp {
    background: #ffffff;
    color: #0d0d0d;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f9f9f9 !important;
    border-right: 1px solid #ebebeb;
}
[data-testid="stSidebar"] * {
    color: #0d0d0d !important;
}
[data-testid="stSidebar"] section {
    padding: 0 0.8rem;
}

/* ── Sidebar buttons ── */
.stButton > button {
    background: transparent;
    color: #555 !important;
    border: none;
    border-radius: 8px;
    font-size: 13.5px;
    font-weight: 400;
    width: 100%;
    text-align: left;
    padding: 8px 10px;
    transition: background 0.15s;
    cursor: pointer;
}
.stButton > button:hover {
    background: #efefef !important;
    color: #0d0d0d !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #f7f7f8;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 4px;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "Click to upload a PDF";
    font-size: 12.5px;
    color: #aaa;
}
[data-testid="stFileUploader"] small {
    display: none !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
    max-width: 720px;
    margin: 0 auto;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #f7f7f8 !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
}

/* Avatar */
[data-testid="chatAvatarIcon-user"] {
    background: #ececec !important;
    color: #333 !important;
    border-radius: 6px !important;
    font-size: 12px !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: #10a37f !important;
    border-radius: 6px !important;
}

/* Message text */
[data-testid="stChatMessage"] p {
    font-size: 14.5px !important;
    line-height: 1.75 !important;
    color: #0d0d0d !important;
}

/* ── Chat input ── */
[data-testid="stBottom"] {
    background: #ffffff;
    padding-bottom: 1rem;
}
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    max-width: 720px;
    margin: 0 auto;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #c5c5c5 !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    color: #0d0d0d !important;
    font-size: 14px !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #aaa !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #10a37f !important;
}

/* ── Alert ── */
[data-testid="stAlert"] {
    background: #fff8e6;
    border: 1px solid #fde68a;
    border-radius: 8px;
    font-size: 13px;
    max-width: 720px;
    margin: 0 auto;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #ccc; }

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #ebebeb;
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ── Core functions ────────────────────────────────────────────
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.unlink(tmp_path)
    return vectorstore, len(chunks)


def get_answer(question, vectorstore, chat_history):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformulate the question as standalone given chat history. Do not answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based only on the context. Be concise and accurate.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_retriever, doc_chain)
    result = rag_chain.invoke({"input": question, "chat_history": chat_history})
    return result["answer"]


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)

    # Logo / title
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;padding:0 2px 1rem 2px">
        <div style="width:28px;height:28px;background:#10a37f;border-radius:6px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:13px;font-weight:700;color:#fff">D</div>
        <span style="font-size:15px;font-weight:600;color:#0d0d0d">DocChat AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    # New chat button
    if st.button("+ New conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<hr>', unsafe_allow_html=True)

    # Document section
    st.markdown("""
    <p style="font-size:11px;font-weight:500;color:#aaa;
              letter-spacing:.07em;text-transform:uppercase;
              margin:0 0 8px 2px">Document</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="hidden"
    )

    if uploaded_file:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Indexing document..."):
                vs, chunks = process_pdf(uploaded_file)
                st.session_state.vectorstore = vs
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chunk_count = chunks
                st.session_state.chat_history = []

        # Doc card
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e5e5e5;border-radius:10px;
                    padding:10px 12px;margin-top:8px">
            <div style="font-size:12.5px;font-weight:500;color:#0d0d0d;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                {uploaded_file.name}
            </div>
            <div style="font-size:11px;color:#999;margin-top:2px">
                {st.session_state.chunk_count} chunks indexed
            </div>
            <div style="font-size:11px;color:#10a37f;margin-top:5px;
                        display:flex;align-items:center;gap:5px">
                <span style="width:6px;height:6px;background:#10a37f;
                             border-radius:50%;display:inline-block"></span>
                Ready
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Remove document"):
            st.session_state.vectorstore = None
            st.session_state.doc_name = None
            st.session_state.chunk_count = 0
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:11px;color:#ccc;padding:0 2px">
        Gemini 1.5 Flash · LangChain · FAISS
    </p>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────
_, center, _ = st.columns([1, 6, 1])

with center:
    # Empty state — no document
    if not st.session_state.vectorstore and not st.session_state.chat_history:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center">
            <div style="width:56px;height:56px;background:#f0fdf8;
                        border:1px solid #d1fae5;border-radius:16px;
                        display:flex;align-items:center;justify-content:center;
                        margin:0 auto 20px auto;font-size:24px;font-weight:700;color:#10a37f">D</div>
            <div style="font-size:1.15rem;font-weight:600;color:#0d0d0d;margin-bottom:8px">
                How can I help you today?
            </div>
            <div style="font-size:13.5px;color:#999;line-height:1.7;max-width:360px;margin:0 auto">
                Upload a PDF from the sidebar and ask questions,<br>
                request summaries, or extract specific information.
            </div>
        </div>

        <br><br>

        <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
            <div style="background:#f7f7f8;border:1px solid #e5e5e5;border-radius:10px;
                        padding:14px 18px;font-size:13px;color:#555;max-width:200px;
                        text-align:left;cursor:default">
                <div style="font-weight:500;color:#0d0d0d;margin-bottom:4px">Summarize</div>
                <div style="color:#999">Get a quick summary of any document</div>
            </div>
            <div style="background:#f7f7f8;border:1px solid #e5e5e5;border-radius:10px;
                        padding:14px 18px;font-size:13px;color:#555;max-width:200px;
                        text-align:left;cursor:default">
                <div style="font-weight:500;color:#0d0d0d;margin-bottom:4px">Extract info</div>
                <div style="color:#999">Pull out specific data or facts</div>
            </div>
            <div style="background:#f7f7f8;border:1px solid #e5e5e5;border-radius:10px;
                        padding:14px 18px;font-size:13px;color:#555;max-width:200px;
                        text-align:left;cursor:default">
                <div style="font-weight:500;color:#0d0d0d;margin-bottom:4px">Ask questions</div>
                <div style="color:#999">Get answers from any PDF content</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Chat header
        st.markdown("""
        <div style="text-align:center;padding:1.2rem 0 1rem 0;
                    border-bottom:1px solid #f0f0f0;margin-bottom:0.8rem">
            <span style="font-size:14px;font-weight:600;color:#0d0d0d">DocChat AI</span>
        </div>
        """, unsafe_allow_html=True)

    # Messages
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant"):
                st.write(msg.content)

    # Input
    if prompt := st.chat_input("Message DocChat AI..."):
        if not st.session_state.vectorstore:
            st.warning("Please upload a PDF document first.")
        else:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    answer = get_answer(
                        prompt,
                        st.session_state.vectorstore,
                        st.session_state.chat_history
                    )
                st.write(answer)
            st.session_state.chat_history.extend([
                HumanMessage(content=prompt),
                AIMessage(content=answer)
            ])