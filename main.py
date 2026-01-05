from flask import Flask, Blueprint, render_template, request, session, jsonify, redirect, url_for, flash
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import mysql.connector
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from keybert import KeyBERT
import chromadb
from chromadb.config import Settings
from langchain.chains import ConversationalRetrievalChain
import pdfplumber, re, json
import pandas as pd
from werkzeug.utils import secure_filename

load_dotenv()
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Get API keys from .env
openai_api_key = os.getenv("OPENAI_API_KEY")
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

# MySQL connection
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="hello@123", # hello@123 for local and Working@2024
  database="Store"
)

ALLOWED_EMAILS = {
    "sg1807@iastate.edu",
    "aparnaj8@iastate.edu","anujs@iastate.edu","kevin.balke@gmail.com","s-sunkari@tti.tamu.edu","a-bibeka@tti.tamu.edu","s-poddar@tti.tamu.edu"
}



mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE IF NOT EXISTS Store")
mycursor.execute("CREATE TABLE IF NOT EXISTS data (`Sr. No.` INT, Email_ID VARCHAR(255), Question TEXT, SignalVerse_Answer TEXT, Rating INT, Raw_AI_Response TEXT, Rating2 INT)")

mydb_bdib = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="hello@123", # hello@123 for local and Working@2024
  database="Store_bdib"
)

mycursor_bdib = mydb_bdib.cursor()
mycursor_bdib.execute("CREATE DATABASE IF NOT EXISTS Store_bdib")
mycursor_bdib.execute("CREATE TABLE IF NOT EXISTS data (`Sr. No.` INT, Email_ID VARCHAR(255), Question TEXT, BDIB_Answer TEXT, Rating INT, Raw_AI_Response TEXT, Rating2 INT)")

mydb_nchrp = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="hello@123",  # hello@123 for local and Working@2024
    database="Store_nchrp"
)

mycursor_nchrp = mydb_nchrp.cursor()
mycursor_nchrp.execute("CREATE DATABASE IF NOT EXISTS Store_nchrp")
mycursor_nchrp.execute("CREATE TABLE IF NOT EXISTS data (`Sr. No.` INT, Email_ID VARCHAR(255), Question TEXT, nchrp_Answer TEXT, Rating INT, Raw_AI_Response TEXT, Rating2 INT)")

app = Flask(__name__)
app.secret_key = "abcd"
chat_history = []

# Load environment variables from .env file
load_dotenv()

# Datasets and their base directories
DATASETS = {
    "signalVerse": "dataSet",
    "bdib": "data2",
    "nchrp": "nchrp_data"
}

# Base embeddings directory
BASE_EMBEDDINGS_DIR = "embeddings"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")




def ensure_dir_exists(path):
    """
    Ensure that a directory exists; create it if not.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def find_all_files(base_dir):
    """
    Recursively find all PDF and DOCX files in the given directory and its subdirectories.
    """
    files_found = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pdf") or file.endswith(".docx"):
                files_found.append(os.path.join(root, file))
    return files_found
def load_or_create_embeddings(dataset_name, pdf_path):
    """
    Load existing embeddings from Chroma vector store or create new ones if not present.
    """
    embeddings_dir = os.path.join(BASE_EMBEDDINGS_DIR, dataset_name)
    ensure_dir_exists(embeddings_dir)

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_embeddings_dir = os.path.join(embeddings_dir, file_name)
    ensure_dir_exists(file_embeddings_dir)

    print(f"\n[INFO] Processing: {pdf_path}")
    try:
        # Connect to existing Chroma vector store
        chroma_client = chromadb.PersistentClient(path=file_embeddings_dir, settings=Settings())

        vectorstore = Chroma(
            client=chroma_client,
            collection_name="pdf_text_collection_large",
            embedding_function=embeddings
        )

        # Check if vectorstore already has documents
        try:
            collection = chroma_client.get_collection("pdf_text_collection_large")
            count = len(collection.get(include=["documents"])["documents"])
            if count > 0:
                print(f"[INFO] Found {count} existing documents in vectorstore for {file_name}, skipping embedding.")
                return vectorstore
        except Exception as e:
            print(f"[INFO] Collection not found or empty, proceeding to embed: {e}")

        # If not found or empty, generate embeddings
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        documents = splitter.split_documents(docs)
        if not documents:
            print(f"[ERROR] No documents loaded from {pdf_path}")
            return None

        print(f"[INFO] Loaded {len(documents)} documents from {file_name}")
        print(f"[INFO] Adding documents to vectorstore for {file_name}...")
        vectorstore.add_documents(documents)

        print(f"[SUCCESS] Embedded and saved documents for {file_name} at {file_embeddings_dir}")
        return vectorstore

    except Exception as e:
        print(f"[EXCEPTION] Error processing {file_name}: {e}")
        return None



def extract_keywords(text, top_k=5):
    
    keywords = kw_model.extract_keywords(text, top_n=top_k, stop_words='english')
    return [kw[0].lower() for kw in keywords]

import os
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

def rerank_docs(question, docs, top_n=4):
    """
    Ask the LLM to score relevance of each doc to the question.
    Returns the top_n most relevant docs.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    reranked = []
    for doc in docs:
        score_prompt = f"""
        Question: {question}
        Document: {doc.page_content}
        On a scale of 1-10, how relevant is this document to answering the question?
        Reply with just a number.
        """
        try:
            score = int(llm.invoke(score_prompt).content.strip())
        except:
            score = 5  # default if parsing fails
        reranked.append((score, doc))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in reranked[:top_n]]

def answer_question(question, vectorstore, chat_history=None, top_n=4):
    # Use MMR retriever for better relevance + diversity
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    if chat_history is None:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        response = qa_chain.invoke({"query": question})
        answer = response.get("result", "")
    else:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        response = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        answer = response.get("answer", "")

    # Get source docs & rerank them
    sources = response.get("source_documents", [])
    top_sources = rerank_docs(question, sources, top_n=top_n)

    # Build metadata string with page numbers
    metadata_info = []
    for doc in top_sources:
        src = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", None)
        if page is not None:
            metadata_info.append(f"{os.path.basename(src)} (page {page+1})")
        else:
            metadata_info.append(os.path.basename(src))
    metadata_text = "\nSources:\n" + "\n".join(set(metadata_info)) if metadata_info else ""

    return answer + "\n\n" + metadata_text, top_sources





import os
import hashlib
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import os
import hashlib
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

def file_hash(filepath):
    """Create a unique hash for a file (so we can detect changes)."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_file(file_path):
    """Load a file based on its type (PDF, DOCX)."""
    if file_path.lower().endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.lower().endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def process_dataset(dataset_name, base_dir):
    print(f"Processing dataset: {dataset_name}")
    
    # Find all PDF and DOCX files in the directory
    files = find_all_files(base_dir)
    print("[FILES LIST]", files)

    embeddings_dir = os.path.join(BASE_EMBEDDINGS_DIR, dataset_name)
    ensure_dir_exists(embeddings_dir)

    # Initialize persistent client
    chroma_client = chromadb.PersistentClient(path=embeddings_dir, settings=Settings())
    collection_name = f"{dataset_name}_collection"

    try:
        collection = chroma_client.get_collection(collection_name)
        print("[INFO] Found existing collection.")
    except Exception:
        print("[INFO] No existing collection. Creating new one.")
        chroma_client.create_collection(collection_name)
        collection = chroma_client.get_collection(collection_name)

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings
    )

    # --- STEP 1: Check what’s already embedded ---
    existing = collection.get(include=["metadatas"])
    embedded_hashes = set()
    if "metadatas" in existing:
        for m in existing["metadatas"]:
            if "file_hash" in m:
                embedded_hashes.add(m["file_hash"])

    # --- STEP 2: Loop over files and add only new ones ---
    for file in files:
        fhash = file_hash(file)

        if fhash in embedded_hashes:
            print(f"[SKIP] {file} already embedded (hash match).")
            continue

        # Load the file (PDF or DOCX)
        docs = load_file(file)
        print(f"[INFO] Loaded {len(docs)} pages from {file}")

        # Split documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        documents = splitter.split_documents(docs)

        for d in documents:
            d.metadata.update({"source": file, "file_hash": fhash})

        # --- Handling large batches by breaking into smaller chunks ---
        batch_size = 1000  # Adjust this number if necessary
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            vectorstore.add_documents(batch)
            print(f"[SUCCESS] Embedded {len(batch)} chunks from {file} (Batch {i//batch_size + 1})")


    return vectorstore

import pdfplumber, re, json

def parse_stage1_reports(folder_path="stage1_linebyline_reports"):
    """
    Reads all Stage 1 PDF reports from a folder and extracts key-value data.
    Returns a list of dicts.
    """
    fields = [
        "Test ID", "Vendor", "Technology", "Test Center", "Date of Testing",
        "Ground Truth Source", "Stage", "Detection Accuracy", "Speed RMSE",
        "Classification Accuracy", "Latency", "False Positive Rate",
        "False Negative Rate", "Uptime", "Weather", "Temperature",
        "Illumination", "Lane Configuration", "Sample Size",
        "Mean Speed Error", "Standard Deviation", "95% Confidence Interval"
    ]
    
    data_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"

            entry = {}
            for field in fields:
                match = re.search(rf"{field}:\s*(.*)", text)
                entry[field] = match.group(1).strip() if match else None
            data_list.append(entry)

    return data_list







vectorstores = {}
for dataset_name, base_dir in DATASETS.items():
    vectorstores[dataset_name] = process_dataset(dataset_name, base_dir)



global the_user_name

@app.route("/")
def index():
    chat_history.clear()
    return render_template('index.html')

@app.route('/index.html')
def first():
    chat_history.clear()
    return render_template('index.html')

@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history():
    global chat_history
    chat_history = []
    return jsonify({'message': 'Chat history cleared successfully'})

@app.route('/answer', methods=['POST'])
def answer():
    global the_user_name
    if request.method == 'POST':
        user_name = request.form['user_question']
        user_email = request.form['user_email']
        session['user_name'] = user_name
        session['user_email'] = user_email
        chat_history.clear()
        if user_name != "":
            the_user_name = user_name
            return render_template('answer.html', user_name=user_name)  
    return render_template("index.html")

@app.route('/submit_question', methods=['POST'])
def submit_question():
    if request.method == 'POST':
        ques_input = request.form['quesInput']
        if ques_input != "":
            user_name = session["user_name"]
            session["question"] = ques_input
            return redirect(url_for('display_result', user_name = user_name))
            


@app.route('/result/<user_name>')
def display_result(user_name):
            ques_input = session["question"]
            vectorstore = vectorstores["signalVerse"]
            answer, sources = answer_question(ques_input, vectorstore)
            # answer = "coming back soon"
            user_name = session["user_name"]
            user_email = session["user_email"]
            
            prompt = f"In context of traffic signals answer this: {ques_input}\n What is the answer and provide meta of the answer in the next line:"

            ChipAnswerText = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
            ChipAnswer = ChipAnswerText.choices[0].message.content.strip()
            # ChipAnswer = "I will be accessed later"

            
            session["question"] = ques_input
            session["answer"] = answer
            session["ChipAnswer"] = ChipAnswer

            chat_history.append({'question': ques_input, 'answer': answer, 'ChipAnswer': ChipAnswer})
            # print(chat_history)
            return render_template('answer.html', user_name=user_name, chat_history=chat_history)
        
@app.route("/rating_submission", methods=["POST"])
def rating_submission():
    if request.method == "POST":
        rating = request.form["rate"]
        rating2 = request.form["rate2"]
        question = session["question"]
        answer = session["answer"]
        user_name = session["user_name"]
        user_email = session["user_email"]
    
        ChipAnswer = session["ChipAnswer"]

        num_row = mycursor.execute("SELECT * FROM data")
        num_row = len(mycursor.fetchall())
        sqlFormula = "INSERT INTO data VALUES (%s,%s,%s,%s,%s,%s,%s)"
        toAppend = (num_row + 1, user_email,question,answer,rating,ChipAnswer,rating2)
        mycursor.execute(sqlFormula,toAppend)
        
        mydb.commit()
        # mycursor.close()
    
        return render_template('answer.html', user_name=user_name, question=question, answer=answer,
                               chat_history=chat_history)

# More routes for `app` go here...

# BDIB app as a Blueprint:
bdib_bp = Blueprint('bdib_bp', __name__)

@bdib_bp.route("/")
def index_bdib():
    chat_history.clear()
    return render_template('index_bdib.html')

@bdib_bp.route('/index_bdib.html')
def first_bdib():
    chat_history.clear()
    return render_template('index_bdib.html')

@bdib_bp.route('/clear_chat_history', methods=['POST'])
def clear_chat_history_bdib():
    global chat_history
    chat_history = []
    return jsonify({'message': 'Chat history cleared successfully'})

@bdib_bp.route('/answer_bdib', methods=['POST'])
def answer_bdib():
    global the_user_name_bdib
    if request.method == 'POST':
        user_name = request.form['user_question']
        user_email = request.form['user_email']
        session['user_name'] = user_name
        session['user_email'] = user_email
        chat_history.clear()
        if user_name != "":
            the_user_name_bdib = user_name
            return render_template('answer_bdib.html', user_name=user_name)

    return render_template("index_bdib.html")

@bdib_bp.route('/submit_question_bdib', methods=['POST'])
def submit_question_bdib():
    if request.method == 'POST':
        ques_input = request.form['quesInput']
        if ques_input != "":
            user_name = session["user_name"]
            session["question"] = ques_input
            return redirect(url_for('bdib_bp.display_result_bdib', user_name=user_name))

@bdib_bp.route('/result/<user_name>')
def display_result_bdib(user_name):
    ques_input = session["question"]
    vectorstore = vectorstores["bdib"]
    answer, sources = answer_question(ques_input, vectorstore)
    # result = qa_chain({"query": ques_input})
    # answer = result["result"]
    
    user_name = session["user_name"]
    user_email = session["user_email"]
    
    prompt = f"In context of Organic Farming answer this: {ques_input}\n What is the answer and provide meta of the answer in the next line:"

    ChipAnswerText = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    # print(ChipAnswerText)
    ChipAnswer = ChipAnswerText.choices[0].message.content.strip()
    # ChipAnswer = "I will be accessed later"
    session["question"] = ques_input
    session["answer"] = answer
    session["ChipAnswer"] = ChipAnswer

    chat_history.append({'question': ques_input, 'answer': answer, 'ChipAnswer': ChipAnswer})
    return render_template('answer_bdib.html', user_name=user_name, chat_history=chat_history)

@bdib_bp.route("/rating_submission", methods=["POST"])
def rating_submission_bdib():
    if request.method == "POST":
        rating = request.form["rate"]
        rating2 = request.form["rate2"]
        question = session["question"]
        answer = session["answer"]
        user_name = session["user_name"]
        user_email = session["user_email"]
        
        ChipAnswer = session["ChipAnswer"]
        num_row = mycursor_bdib.execute("SELECT * FROM data")
        num_row = len(mycursor_bdib.fetchall())
        sqlFormula = "INSERT INTO data VALUES (%s,%s,%s,%s,%s,%s,%s)"
        toAppend = (num_row + 1, user_email, question, answer, rating, ChipAnswer, rating2)
        mycursor_bdib.execute(sqlFormula, toAppend)
        
        mydb_bdib.commit()
      
        return render_template('answer_bdib.html', user_name=user_name, question=question, answer=answer,
                               chat_history=chat_history)

@bdib_bp.route('/show_table')
def show_table_bdib():
    mycursor_bdib.execute("SELECT * FROM data")
    table_data = mycursor_bdib.fetchall()
    column_headers = ["Sr. No.", "Email ID", "Question", "BDIB Answer", "Rating", "Raw AI Response", "Rating2"]
    return render_template('show_table.html', table_data=table_data, column_headers=column_headers)

# for nchrp
nchrp_bp = Blueprint('nchrp_bp', __name__)

SUMMARY_COLS = [
    "Test ID",
    "Vendor Name",
    "Sensor model name",
    "Sensor Technology",
    "Stage & Level",
    "Test Center",
    "Test Location (State)",
    "Date of Testing",
    "Ground Truth Source",
]

# Metrics "categories"
METRIC_BASE_COLS = [
    "Test ID",
    "Sensor Function",
    "Performance Measure",
    "Measured value (%)",
    "Sample size",
    "Weather (F)",
    "Lighting",
]

OPTIONAL_METRIC_COLS = [
    "Testing Notes (optional)",
]

# ----------------------------
# Data loader (reads ALL files in DATA_DIR)
# Supports: .xlsx (all sheets), .csv
# Returns:
#  tests: list[dict] (unique Test IDs)
#  metrics_by_test: dict[TestID -> list[dict metric rows]]
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "sampleData")      # source-of-truth files live here
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads") # optional staging
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
import os
import pandas as pd
from flask import current_app

def load_nchrp_from_files():
    DATA_DIR = os.path.join(current_app.root_path, "sampleData")
    frames = []

    if not os.path.isdir(DATA_DIR):
        return [], {}

    for fn in os.listdir(DATA_DIR):
        # Skip temp/hidden files (Excel lock files, macOS metadata, etc.)
        if fn.startswith("~$") or fn.startswith("."):
            continue

        path = os.path.join(DATA_DIR, fn)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(fn)[1].lower()

        try:
            if ext == ".csv":
                df = pd.read_csv(path, dtype=object)
                df["__source__"] = fn
                frames.append(df)

            elif ext == ".xlsx":
                # Force engine for xlsx
                xls = pd.ExcelFile(path, engine="openpyxl")
                for sheet in xls.sheet_names:
                    s = pd.read_excel(xls, sheet_name=sheet, dtype=object)
                    s["__source__"] = fn
                    s["__sheet__"] = sheet
                    frames.append(s)

            else:
                # Ignore anything else (pdf, txt, xls, etc.)
                continue

        except Exception as e:
            # Don't crash the whole page because of one bad file
            print(f"[WARN] Skipping file {fn}: {e}")
            continue

    if not frames:
        return [], {}

    df = pd.concat(frames, ignore_index=True)

    # Clean columns
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.where(pd.notnull(df), None)

    # Strip string cells
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Validate required columns (summary + base metrics)
    required = set(SUMMARY_COLS + METRIC_BASE_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # Forward-fill Test ID so block rows inherit the same Test ID
    # Forward-fill Test ID so block rows inherit the same Test ID
    df["Test ID"] = df["Test ID"].replace("", None).ffill()

    # Forward-fill Sensor Function within each Test ID block (CRITICAL for Performance Measure dropdown)
    df["Sensor Function"] = df["Sensor Function"].replace("", None)
    df["Sensor Function"] = df.groupby("Test ID")["Sensor Function"].ffill()

# Now drop rows that STILL don't have a Test ID
    df = df[df["Test ID"].notna() & (df["Test ID"].astype(str).str.strip() != "")]

    if df.empty:
        return [], {}
    def clean_val(x):
    # converts NaN/NaT to None so JSON is valid
        return None if pd.isna(x) else x


    # Tests (unique by Test ID)
    tests_df = df[SUMMARY_COLS].drop_duplicates(subset=["Test ID"]).copy()
    tests = tests_df.fillna("").to_dict(orient="records")

    # Metrics per test
    metric_cols = [c for c in (METRIC_BASE_COLS + OPTIONAL_METRIC_COLS) if c in df.columns]

    for extra in ["__source__", "__sheet__"]:
        if extra in df.columns and extra not in metric_cols:
            metric_cols.append(extra)

    metrics_by_test = {}
    for _, row in df[metric_cols].iterrows():
        tid = str(row["Test ID"]).strip()
        metric = {k: clean_val(row[k]) for k in metric_cols if k != "Test ID"}
        metrics_by_test.setdefault(tid, []).append(metric)


    return tests, metrics_by_test



@nchrp_bp.route("/")
def index_nchrp():
    chat_history.clear()
    return render_template('index_nchrp.html')

@nchrp_bp.route('/index_nchrp.html')
def first_nchrp():
    chat_history.clear()
    return render_template('index_nchrp.html')

@nchrp_bp.route('/clear_chat_history', methods=['POST'])
def clear_chat_history_nchrp():
    global chat_history
    chat_history = []
    return jsonify({'message': 'Chat history cleared successfully'})


@nchrp_bp.route('/answer_nchrp', methods=['POST'])
def answer_nchrp():
    global the_user_name_nchrp
    user_name = request.form['user_question']
    user_email = request.form['user_email'].strip().lower()
    user_role = request.form.get('user_role', '').strip().lower()

    # Approval check
    if user_email not in ALLOWED_EMAILS:
        flash("Your access is not approved yet.")
        return render_template("index_nchrp.html")

    # ⭐ ALWAYS save the role the user selected
    # (do NOT check if it's already in session)
    session['user_role'] = user_role

    session['user_name'] = user_name
    session['user_email'] = user_email
    chat_history.clear()

    return render_template(
        'nchrp_choice.html',
        user_name=user_name,
        user_email=user_email,
        user_role=user_role
    )




@nchrp_bp.route('/go_to_clearinghouse', methods=['GET'])
def go_to_clearinghouse():
    """
    Redirect user from the choice page to the clearinghouse question page.
    """
    user_name = session.get("user_name", "")
    user_email = session.get("user_email", "")
    return render_template('answer_nchrp.html', user_name=user_name)


@nchrp_bp.route('/submit_question_nchrp', methods=['POST'])
def submit_question_nchrp():
    if request.method == 'POST':
        ques_input = request.form['quesInput']
        if ques_input != "":
            user_name = session["user_name"]
            session["question"] = ques_input
            return redirect(url_for('nchrp_bp.display_result_nchrp', user_name=user_name))

@nchrp_bp.route('/result/<user_name>')
def display_result_nchrp(user_name):
    ques_input = session["question"]
    vectorstore = vectorstores["nchrp"]
    answer, sources = answer_question(ques_input, vectorstore)  # now includes metadata
    user_name = session["user_name"]
    
    prompt = f"In context of transportation answer this: {ques_input}\n What is the answer and provide meta of the answer in the next line:"

    ChipAnswerText = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    ChipAnswer = ChipAnswerText.choices[0].message.content.strip()

    session["question"] = ques_input
    session["answer"] = answer
    session["ChipAnswer"] = ChipAnswer

    chat_history.append({
        'question': ques_input,
        'answer': answer,
        'ChipAnswer': ChipAnswer
    })

    return render_template('answer_nchrp.html', user_name=user_name, chat_history=chat_history)


@nchrp_bp.route("/rating_submission", methods=["POST"])
def rating_submission_nchrp():
    if request.method == "POST":
        rating = request.form["rate"]
        rating2 = request.form["rate2"]
        question = session["question"]
        answer = session["answer"]
        user_name = session["user_name"]
        user_email = session["user_email"]
        
        ChipAnswer = session["ChipAnswer"]
        num_row = mycursor_nchrp.execute("SELECT * FROM data")
        num_row = len(mycursor_nchrp.fetchall())
        sqlFormula = "INSERT INTO data VALUES (%s,%s,%s,%s,%s,%s,%s)"
        toAppend = (num_row + 1, user_email, question, answer, rating, ChipAnswer, rating2)
        mycursor_nchrp.execute(sqlFormula, toAppend)
        
        mydb_nchrp.commit()
      
        return render_template('answer_nchrp.html', user_name=user_name, question=question, answer=answer,
                               chat_history=chat_history)

@nchrp_bp.route('/show_table')
def show_table_nchrp():
    mycursor_nchrp.execute("SELECT * FROM data")
    table_data = mycursor_nchrp.fetchall()
    column_headers = ["Sr. No.", "Email ID", "Question", "nchrp Answer", "Rating", "Raw AI Response", "Rating2"]
    return render_template('show_table.html', table_data=table_data, column_headers=column_headers)


import os
import pandas as pd
from flask import current_app


from flask import current_app, render_template, session
import os

@nchrp_bp.route("/report")
def testSampleReport():
    # OPTIONAL: if you want /report to always read from sampleData
    sample_dir = os.path.join(current_app.root_path, "sampleData")
    if not os.path.isdir(sample_dir):
        return f"sampleData folder not found at: {sample_dir}"

    # Use the unified loader (reads ALL .xlsx/.csv files from the directory it uses)
    tests, metrics = load_nchrp_from_files()

    # Template CSV headers button uses these
    column_headers = SUMMARY_COLS

    return render_template(
        "testSample.html",
        column_headers=column_headers,
        tests=tests,
        metrics=metrics,
        user_role=session.get("user_role", "public"),
        user_name=session.get("user_name", "")
    )






@nchrp_bp.route("/upload_report", methods=["POST"])
def upload_report():
    if session.get("user_role") == "public":
        flash("You do not have permission to upload reports.")
        return redirect(url_for("nchrp_bp.testSampleReport"))

    uploaded_file = request.files.get("pdf_file")  # keep field name to avoid changing HTML form right now
    if not uploaded_file:
        flash("No file uploaded.")
        return redirect(url_for("nchrp_bp.testSampleReport"))

    filename = secure_filename(uploaded_file.filename or "")
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".xlsx", ".csv"]:
        flash("Upload failed: Please upload an Excel (.xlsx) or CSV (.csv) file.")
        return redirect(url_for("nchrp_bp.testSampleReport"))

    # Save into DATA_DIR (source of truth)
    save_path = os.path.join(DATA_DIR, filename)
    uploaded_file.save(save_path)

    # Validate quickly by attempting to load (so bad files are rejected immediately)
    try:
        load_nchrp_from_files()
    except Exception as e:
        # rollback the bad file
        try:
            os.remove(save_path)
        except Exception:
            pass
        flash(f"Upload failed: {e}")
        return redirect(url_for("nchrp_bp.testSampleReport"))

    flash("Report file uploaded successfully!")
    return redirect(url_for("nchrp_bp.testSampleReport"))

# app/nchrp_ai.py
# Fully working multi-Excel + multi-sheet Ask AI endpoint
# - Reads ALL .xlsx files from "sample data/" folder
# - Each sheet can represent a stage/level (Stage 1 Level 1, Stage 2 Level 2, etc.)
# - Treats each Excel ROW as one record (dict)
# - Adds Source File + Source Sheet + ensures Stage & Level exists
# - Refuses gibberish/unrelated questions by requiring overlap with DB vocabulary
# - Retrieves relevant rows using embeddings, then answers ONLY from those rows

import os
import re
import json
import time
import hashlib
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from flask import request, jsonify
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
DATA_FOLDER = os.path.join(os.getcwd(), "sampleData")  # <-- your folder
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# retrieval strictness (tune)
VOCAB_OVERLAP_MIN = 1          # require >=1 DB term overlap to avoid gibberish
TOP_K = 30                     # rows returned as evidence
SIM_THRESHOLD = 0.22           # higher = stricter, fewer matches

# embedding cache controls
CACHE_TTL_SECONDS = 10 * 60    # rebuild embeddings every 10 minutes
MAX_ROWS_FOR_EMBED = 25000     # safety cap

# -----------------------------
# GLOBAL CACHES (in-memory)
# -----------------------------
_CACHE = {
    "built_at": 0.0,
    "fingerprint": None,
    "columns": [],
    "rows": [],
    "docs": [],
    "doc_embs": None,          # np.ndarray shape (N, D)
    "allowed_terms": set(),
    "row_count": 0,
}

# -----------------------------
# Helpers: normalization
# -----------------------------
def clean_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()

def norm_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()

def tokenize(text: str) -> List[str]:
    # keeps tokens like CAL-016
    return re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)?", (text or "").lower())

def infer_stage_level_from_sheet(sheet_name: str) -> str:
    s = (sheet_name or "").strip()

    m = re.search(r"stage\D*(\d+)\D*level\D*(\d+)", s, flags=re.I)
    if m:
        return f"Stage {m.group(1)} Level {m.group(2)}"

    m = re.search(r"\bs(\d+)\s*l(\d+)\b", s, flags=re.I)
    if m:
        return f"Stage {m.group(1)} Level {m.group(2)}"

    return s  # fallback: keep sheet name

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# -----------------------------
# Excel loading (folder -> workbooks -> sheets -> rows)
# -----------------------------
def read_workbook_all_sheets(filepath: str) -> List[Dict[str, Any]]:
    xls = pd.ExcelFile(filepath)
    all_rows: List[Dict[str, Any]] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet)

        # drop template artifact columns
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

        # normalize headers
        df.columns = [clean_col(c) for c in df.columns]

        # drop fully empty rows
        df = df.dropna(how="all")
        if df.empty:
            continue

        # normalize values to strings
        for c in df.columns:
            df[c] = df[c].apply(norm_str)

        stage_level = infer_stage_level_from_sheet(sheet)

        rows = df.to_dict(orient="records")
        for r in rows:
            r["Source File"] = os.path.basename(filepath)
            r["Source Sheet"] = sheet
            if not norm_str(r.get("Stage & Level")):
                r["Stage & Level"] = stage_level
        all_rows.extend(rows)

    return all_rows

def load_all_excels(folder_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"DATA_FOLDER not found: {folder_path}")

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".xlsx"):
            continue
        if fname.startswith("~$"):  # skip Excel temp files
            continue
        fp = os.path.join(folder_path, fname)
        all_rows.extend(read_workbook_all_sheets(fp))

    # union of columns
    cols = set()
    for r in all_rows:
        cols.update(r.keys())

    preferred = [
        "Test ID", "Stage & Level", "Test Center", "Test Location (State)",
        "Sensor Technology", "Vendor Name", "Sensor model name",
        "Sensor Function", "Performance Measure", "Measured value (%)",
        "Sample size", "Weather (F)", "Lighting", "Testing Notes (optional)",
        "Source File", "Source Sheet"
    ]
    columns = [c for c in preferred if c in cols] + sorted([c for c in cols if c not in preferred])

    return columns, all_rows

# -----------------------------
# Build searchable “docs” per row + allowed vocabulary
# -----------------------------
def row_to_doc(row: Dict[str, Any], columns: List[str]) -> str:
    # Keep the doc compact but descriptive.
    keep_cols = [
        "Test ID", "Stage & Level", "Test Center", "Test Location (State)",
        "Sensor Technology", "Vendor Name", "Sensor model name",
        "Ground Truth Source",
        "Sensor Function", "Performance Measure",
        "Measured value (%)", "Sample size", "Weather (F)", "Lighting",
        "Testing Notes (optional)",
        "Source File", "Source Sheet"
    ]
    keep = [c for c in keep_cols if c in columns]
    parts = []
    for c in keep:
        v = norm_str(row.get(c))
        if v:
            parts.append(f"{c}: {v}")
    return " | ".join(parts)

def build_allowed_terms(rows: List[Dict[str, Any]], columns: List[str]) -> set:
    """
    DB-driven gibberish filter.
    If a question has zero overlap with these terms, we refuse.
    """
    important_cols = [
        "Test ID", "Test Center", "Test Location (State)",
        "Sensor Technology", "Vendor Name", "Sensor model name",
        "Stage & Level", "Sensor Function", "Performance Measure"
    ]
    existing = [c for c in important_cols if c in columns]

    terms = set()

    for r in rows:
        for c in existing:
            v = norm_str(r.get(c))
            if not v:
                continue

            # full phrase
            terms.add(v.lower())

            # token pieces
            for t in tokenize(v):
                if len(t) >= 3:
                    terms.add(t)

    # include column name tokens as well
    for c in existing:
        for t in tokenize(c):
            if len(t) >= 3:
                terms.add(t)

    return terms

# -----------------------------
# Fingerprint folder (to rebuild cache when files change)
# -----------------------------
def folder_fingerprint(folder_path: str) -> str:
    """
    Hash of filenames + mtime + size. Cheap and good enough.
    """
    h = hashlib.sha256()
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".xlsx") or fname.startswith("~$"):
            continue
        fp = os.path.join(folder_path, fname)
        st = os.stat(fp)
        h.update(fname.encode("utf-8"))
        h.update(str(st.st_mtime_ns).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
    return h.hexdigest()

# -----------------------------
# Build / refresh embedding cache
# -----------------------------
def ensure_cache_fresh():
    now = time.time()

    # Refresh if TTL expired or fingerprint changed
    fp = folder_fingerprint(DATA_FOLDER)
    ttl_expired = (now - _CACHE["built_at"]) > CACHE_TTL_SECONDS
    fp_changed = (_CACHE["fingerprint"] != fp)

    if _CACHE["doc_embs"] is not None and not ttl_expired and not fp_changed:
        return

    columns, rows = load_all_excels(DATA_FOLDER)
    if not rows:
        # keep empty cache
        _CACHE.update({
            "built_at": now,
            "fingerprint": fp,
            "columns": columns,
            "rows": [],
            "docs": [],
            "doc_embs": None,
            "allowed_terms": set(),
            "row_count": 0
        })
        return

    # safety cap
    if len(rows) > MAX_ROWS_FOR_EMBED:
        rows = rows[:MAX_ROWS_FOR_EMBED]

    docs = [row_to_doc(r, columns) for r in rows]
    allowed_terms = build_allowed_terms(rows, columns)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Embed in chunks to avoid huge requests
    all_embs = []
    BATCH = 512
    for i in range(0, len(docs), BATCH):
        batch_docs = docs[i:i+BATCH]
        emb = client.embeddings.create(model=EMBED_MODEL, input=batch_docs)
        batch_embs = [np.array(e.embedding, dtype=np.float32) for e in emb.data]
        all_embs.extend(batch_embs)

    doc_embs = np.vstack(all_embs)  # shape (N, D)

    _CACHE.update({
        "built_at": now,
        "fingerprint": fp,
        "columns": columns,
        "rows": rows,
        "docs": docs,
        "doc_embs": doc_embs,
        "allowed_terms": allowed_terms,
        "row_count": len(rows),
    })

# -----------------------------
# Your Flask route
# -----------------------------
# Make sure you have:
# from flask import Blueprint
# nchrp_bp = Blueprint("nchrp", __name__)
#
# and register blueprint in your app.

@nchrp_bp.route("/ask_ai", methods=["POST"])
def ask_ai():
    payload = request.json or {}
    question = norm_str(payload.get("question", ""))
    debug = bool(payload.get("debug", False))

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 1) Ensure cache exists (loads all excel files + builds embeddings)
    try:
        ensure_cache_fresh()
    except Exception as e:
        return jsonify({"error": f"Data loading error: {e}"}), 500

    columns = _CACHE["columns"]
    rows = _CACHE["rows"]
    allowed_terms = _CACHE["allowed_terms"]
    doc_embs = _CACHE["doc_embs"]

    if not rows or doc_embs is None:
        return jsonify({"answer": "I don't know.", "columns": columns, "matched_rows": []})

    # 2) Gibberish / unrelated guard: must overlap DB vocabulary
    q_tokens = tokenize(question)
    overlap = [t for t in q_tokens if t in allowed_terms]

    if len(overlap) < VOCAB_OVERLAP_MIN:
        out = {"answer": "I don't know.", "columns": columns, "matched_rows": []}
        if debug:
            out["debug"] = {
                "reason": "no_vocab_overlap",
                "question_tokens": q_tokens[:30],
                "overlap": overlap[:30],
                "vocab_overlap_min": VOCAB_OVERLAP_MIN
            }
        return jsonify(out)

    # 3) Embed the question and retrieve top rows
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    q_emb_resp = client.embeddings.create(model=EMBED_MODEL, input=[question])
    q_emb = np.array(q_emb_resp.data[0].embedding, dtype=np.float32)

    # cosine similarity against all row embeddings
    # sim = (A·q) / (||A|| ||q||) for each row A
    q_norm = float(np.linalg.norm(q_emb)) or 1.0
    A = doc_embs
    A_norms = np.linalg.norm(A, axis=1)
    A_norms[A_norms == 0] = 1.0
    sims = (A @ q_emb) / (A_norms * q_norm)

    # top indices
    top_idx = np.argsort(-sims)[:TOP_K]
    top_pairs = [(float(sims[i]), int(i)) for i in top_idx]

    # threshold filter
    kept = [(s, i) for s, i in top_pairs if s >= SIM_THRESHOLD]
    if not kept:
        out = {"answer": "I don't know.", "columns": columns, "matched_rows": []}
        if debug:
            out["debug"] = {
                "reason": "semantic_no_hits",
                "best_similarity": top_pairs[0][0] if top_pairs else None,
                "threshold": SIM_THRESHOLD,
                "top_10": top_pairs[:10],
            }
        return jsonify(out)

    matched_rows = [rows[i] for _, i in kept]

    # 4) Ask GPT: MUST answer only from matched rows
    prompt = f"""
You are answering questions about an Excel database of NCHRP sensor testing.

User question:
{question}

Matched Rows (ONLY source of truth):
{json.dumps(matched_rows[:50], indent=2)}

Rules:
- Answer ONLY using facts present in Matched Rows.
- If the answer cannot be determined from Matched Rows, reply exactly: I don't know.
- Do not invent, assume, or generalize beyond the data shown.
- If helpful, group by Stage & Level or Test Center / Sensor Technology.
- Be concise.
""".strip()

    reply = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    answer = norm_str(reply.choices[0].message.content) or "I don't know."

    out = {
        "answer": answer,
        "columns": columns,
        "matched_rows": matched_rows
    }

    if debug:
        out["debug"] = {
            "reason": "ok",
            "row_count_total": _CACHE["row_count"],
            "cache_built_at": _CACHE["built_at"],
            "vocab_overlap": overlap[:25],
            "threshold": SIM_THRESHOLD,
            "top_similarities": [s for s, _ in kept[:10]],
            "matched_count": len(matched_rows),
        }

    return jsonify(out)



# Register the Blueprint with `app`:
app.register_blueprint(bdib_bp, url_prefix='/bdib_bp')
app.register_blueprint(nchrp_bp, url_prefix='/nchrp_bp')

if __name__ == '__main__':

    app.run(debug=True)
