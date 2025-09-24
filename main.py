from flask import Flask, Blueprint, render_template, request, session, jsonify, redirect, url_for
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
  passwd="Working@2024", # working@2024 for local and Working@2024
  database="Store"
)

mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE IF NOT EXISTS Store")
mycursor.execute("CREATE TABLE IF NOT EXISTS data (`Sr. No.` INT, Email_ID VARCHAR(255), Question TEXT, SignalVerse_Answer TEXT, Rating INT, Raw_AI_Response TEXT, Rating2 INT)")

mydb_bdib = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Working@2024", # working@2024 for local and Working@2024
  database="Store_bdib"
)

mycursor_bdib = mydb_bdib.cursor()
mycursor_bdib.execute("CREATE DATABASE IF NOT EXISTS Store_bdib")
mycursor_bdib.execute("CREATE TABLE IF NOT EXISTS data (`Sr. No.` INT, Email_ID VARCHAR(255), Question TEXT, BDIB_Answer TEXT, Rating INT, Raw_AI_Response TEXT, Rating2 INT)")

mydb_nchrp = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Working@2024",  # working@2024 for local and Working@2024
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
    "bdib": "Data",
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

def answer_question(question, vectorstore, chat_history=None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
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

    sources = response.get("source_documents", [])

    # Build metadata string with page numbers
    metadata_info = []
    for doc in sources:
        src = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", None)
        if page is not None:
            metadata_info.append(f"{os.path.basename(src)} (page {page+1})")  
            # page is usually zero-indexed, so +1
        else:
            metadata_info.append(os.path.basename(src))

    metadata_text = "\nSources:\n" + "\n".join(set(metadata_info)) if metadata_info else ""

    return answer + "\n\n" + metadata_text, sources




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
    if request.method == 'POST':
        user_name = request.form['user_question']
        user_email = request.form['user_email']
        session['user_name'] = user_name
        session['user_email'] = user_email
        chat_history.clear()
        if user_name != "":
            the_user_name_nchrp = user_name
            return render_template('answer_nchrp.html', user_name=user_name)
    return render_template("index_nchrp.html")

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

# Register the Blueprint with `app`:
app.register_blueprint(bdib_bp, url_prefix='/bdib_bp')
app.register_blueprint(nchrp_bp, url_prefix='/nchrp_bp')

if __name__ == '__main__':

    app.run(debug=True)
