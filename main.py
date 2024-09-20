from flask import Flask, render_template, request, session,jsonify, redirect, url_for
import pandas as pd
from openpyxl import load_workbook
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

#FOR 2nd model
import os
from openai import OpenAI

#FOR mySQL
import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="working@2024", #Working@2024 for deployed
  database = "Store"
)

mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE IF NOT EXISTS Store")

mycursor.execute("CREATE TABLE IF NOT EXISTS data (`Sr. No.` INT,Email_ID VARCHAR(255), Question TEXT, SignalVerse_Answer TEXT, Rating INT, Raw_AI_Response TEXT, Rating2 INT)")




loader = PyPDFLoader("signal_timing_manual_fhwa.pdf")
pages = loader.load()
print("document 1 successfully loaded")

loader2 = PyPDFLoader("22097.pdf")
pages2 = loader2.load()
print("document 2 successfully loaded")

loader3 = PyPDFLoader("mutcd11thedition.pdf")
pages3 = loader3.load()
print("document 3 successfully loaded")

loader4 = PyPDFLoader("ATSPM_Methods_and_Assumptions_4.3.pdf")
pages4 = loader4.load()
print("document 4 successfully loaded")

loader5 = PyPDFLoader("ATSPM_User Case Examples_Manual_20200128.pdf")
pages5 = loader5.load()
print("document 5 successfully loaded")

loader6 = PyPDFLoader("Performance Measures for Traffic Signal Systems_ An Outcome-Orien.pdf")
pages6 = loader6.load()
print("document 6 successfully loaded")


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len
)

splits = text_splitter.split_documents(pages + pages2 + pages3 + pages4 + pages5 + pages6)

len(pages + pages2 + pages3 + pages4 + pages5 + pages6)

len(splits)

print("documents splitted")


embedding = OpenAIEmbeddings(openai_api_key="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")

persist_directory = 'chroma/stm_brandNew2/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print("chroma created")

print(vectordb._collection.count())

vectordb.persist()

persist_directory = persist_directory
OpenAIEmbeddings(openai_api_key="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")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print(vectordb._collection.count())

# llm2 = OpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
llm = ChatOpenAI(model_name='gpt-4o',  openai_api_key="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",temperature=0)
llm2 = ChatOpenAI(model_name='gpt-4o-2024-08-06', openai_api_key="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",temperature=0)
llm3 = ChatOpenAI(model_name='gpt-4o-2024-05-13', openai_api_key="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",temperature=0)
openai_api_key = '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'
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI(# defaults to os.environ.get("OPENAI_API_KEY")
)

# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever()
# )

def query_with_model(query, use_llm2=False, use_llm3=False):
    if use_llm2:
        model = llm2
    elif use_llm3:
        model = llm3
    else:
        model = llm
    return RetrievalQA.from_chain_type(
        llm=model,
        retriever=vectordb.as_retriever()
    )
print("beginning of frontend code")

#####   BEGINNING OF FRONT END CODE #####


app = Flask(__name__)
app.secret_key = "abcd"
chat_history = []
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
            qa_chain = query_with_model("your query", use_llm2=True)
            result = qa_chain({"query": ques_input})
            answer = result["result"]
            # answer = "coming back soon"
            user_name = session["user_name"]
            user_email = session["user_email"]
            
            prompt = f"In context of traffic signals answer this: {ques_input}\n What is the answer and provide meta of the answer in the next line:"

            ChipAnswerText = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
            # print(ChipAnswerText)
            ChipAnswer = ChipAnswerText.choices[0].message.content
            # ChipAnswer = "I will be accessed later"

            
            session["question"] = ques_input
            session["answer"] = answer
            session["ChipAnswer"] = ChipAnswer

            chat_history.append({'question': ques_input, 'answer': answer, 'ChipAnswer': ChipAnswer})
            print(chat_history)
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
        # print(answer)
        # print(ChipAnswer)
        # chat_histories[user_email].append({'question': question, 'answer': answer, 'message': message})
        
        num_row = mycursor.execute("SELECT * FROM data")
        num_row = len(mycursor.fetchall())
        sqlFormula = "INSERT INTO data VALUES (%s,%s,%s,%s,%s,%s,%s)"
        toAppend = (num_row + 1, user_email,question,answer,rating,ChipAnswer,rating2)
        mycursor.execute(sqlFormula,toAppend)

        mydb.commit()
        # mycursor.close()
      
        return render_template('answer.html', user_name=user_name, question=question, answer=answer,
                               chat_history=chat_history)

  
if __name__ == '__main__':
    app.run(debug=True)