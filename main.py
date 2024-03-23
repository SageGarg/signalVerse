from flask import Flask, render_template, request, session,jsonify
import pandas as pd
from openpyxl import load_workbook
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

loader = PyPDFLoader("signal_timing_manual_fhwa.pdf")


pages = loader.load()



loader2= PyPDFLoader("22097.pdf")



pages2 = loader2.load()


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len
)

splits = text_splitter.split_documents(pages+pages2)

len(pages+pages2)

len(splits)



embedding = OpenAIEmbeddings(openai_api_key="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")

persist_directory = 'chroma/stm_brandNew/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

vectordb.persist()

persist_directory = persist_directory
OpenAIEmbeddings(openai_api_key="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")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print(vectordb._collection.count())


llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613', openai_api_key="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",temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)




#####   BEGINNING OF FRONT END CODE #####


app = Flask(__name__)
app.secret_key = "abcd"
chat_history = []


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

# Other routes and application code...


@app.route('/answer', methods=['POST'])
def answer():
    if request.method == 'POST':
        user_name = request.form['user_question']
        session['user_name'] = user_name
        
        chat_history.clear()
        if user_name != "":
            return render_template('answer.html', user_name=user_name)
        
    return render_template("index.html")

@app.route('/submit_question', methods=['POST'])
def submit_question():
    if request.method == 'POST':
        ques_input = request.form['quesInput']
        
        if ques_input != "":
            result = qa_chain({"query": ques_input})
            answer = result["result"]
            # answer = "I don't have an API key yet; therefore, I won't be able to answer your question."
            user_name = session["user_name"]
            print("I am inside the if conditional")
            session["question"] = ques_input
            session["answer"] = answer
            chat_history.append({'question': ques_input, 'answer': answer})
            return render_template('answer.html', user_name=user_name, chat_history=chat_history)
         
            
@app.route("/rating_submission" , methods = ["POST", "GET"])
def rating_submission():
    if request.method == "POST":
        
        rating = request.form["rate"]
        question = session["question"]
        answer = session["answer"]
        print("in the rating conditional")
        user_name = session["user_name"]
        message = "You rated the above answer: " + rating
        # chat_history.append({'question': question, 'answer': answer, 'message':message})
        sheet_name = user_name.upper()
        excel_file_path = "Book1.xlsx"
        try:
            append_data_to_existing_sheet(excel_file_path, sheet_name, question, answer, rating)
        except FileNotFoundError:
            create_new_sheet(excel_file_path, sheet_name, question, answer, rating)
        
        return render_template('answer.html', user_name=user_name, question=question, answer=answer, chat_history=chat_history, message = message)
        



def append_data_to_existing_sheet(excel_file_path, sheet_name, question, answer, rating):
    workbook = load_workbook(excel_file_path)
    

    if sheet_name in workbook.sheetnames:
        existing_df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        append_df = pd.DataFrame({'Question': [question], 'Answers': [answer], 'Rating': [rating]})
        updated_df = pd.concat([existing_df, append_df], ignore_index=True)

        with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists="overlay") as writer:
            updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print("Appended data to the existing sheet")
    else:
        print("Sheet does not exist. Creating a new sheet.")
        create_new_sheet(excel_file_path, sheet_name, question, answer, rating)



def create_new_sheet(excel_file_path, sheet_name, question, answer, rating):
    new_data = {'Question': [question], 'Answers': [answer], 'Rating': [rating]}
    updated_df = pd.DataFrame(new_data)

    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
        updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("Created a new sheet")

if __name__ == '__main__':
    app.run(debug=True)