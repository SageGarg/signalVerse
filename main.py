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
  passwd="working@2024",
  database = "Inventory"
)

mycursor = mydb.cursor()
# mycursor.execute("CREATE DATABASE Inventory")
# mycursor.execute("CREATE TABLE store (`Sr. No.` INT,Email_ID VARCHAR(255), Question TEXT, SignalVerse_Answer TEXT, Rating INT, Raw_AI_Response TEXT, Rating2 INT)")




loader = PyPDFLoader("signal_timing_manual_fhwa.pdf")
pages = loader.load()
print("document 1 successfully loaded")

loader2 = PyPDFLoader("22097.pdf")
pages2 = loader2.load()
print("document 2 successfully loaded")

loader3 = PyPDFLoader("mutcd11thedition.pdf")
pages3 = loader3.load()
print("document 3 successfully loaded")


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len
)

splits = text_splitter.split_documents(pages+pages2 + pages3)

len(pages+pages2 + pages3)

len(splits)

print("documents splitted")

# pdf_paths = [r"22097.pdf"]

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
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613', openai_api_key="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",temperature=0)

openai_api_key = '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'
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI(# defaults to os.environ.get("OPENAI_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
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
            result = qa_chain({"query": ques_input})
            answer = result["result"]
            # answer = "coming back soon"
            user_name = session["user_name"]
            user_email = session["user_email"]
            
            prompt = f"In context of traffic signals answer this: {ques_input}\n What is the answer and provide meta data of the answer in the next line:"

            ChipAnswerText = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
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
        
        num_row = mycursor.execute("SELECT * FROM store")
        num_row = len(mycursor.fetchall())
        sqlFormula = "INSERT INTO store VALUES (%s,%s,%s,%s,%s,%s,%s)"
        toAppend = (num_row + 1, user_email,question,answer,rating,ChipAnswer,rating2)
        mycursor.execute(sqlFormula,toAppend)

        mydb.commit()
        # mycursor.close()
      
        return render_template('answer.html', user_name=user_name, question=question, answer=answer,
                               chat_history=chat_history)

  
if __name__ == '__main__':
    app.run(debug=True)