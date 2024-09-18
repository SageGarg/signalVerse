# SignalVerse Chatbot

SignalVerse is a chatbot application built using Flask and Tailwind CSS. It utilizes language processing techniques for answering user queries based on a predefined knowledge base.

## Setup

### Prerequisites

- Python 3.x
- Node.js and npm

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/SageGarg/signalVerse.git
    ```

2. Install Python dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. Install Tailwind CSS:

follow this:
https://tailwindcss.com/docs/installation

              OR
              THE GIVEN COMMANDS

    ```bash
    npm install -D tailwindcss
    npx tailwindcss init
    ```

    //node's version has to be greater than 14.0.0

    the above commands create a tailwind.config.js file, in that file add the following command:
    content: ["./templates/*"],; this ensures all the html files in the templates folder can use tailwind.

    then create an input.css file in the static/src directory, and add the following commands in the file:
    @tailwind base;
@tailwind components;
@tailwind utilities;

now create a css folder in the static folder.
4. Build Tailwind CSS:

    ```bash
    npx tailwindcss -i ./static/src/input.css -o ./static/css/output.css --watch
    ```

    IF YOU DON'T WANT TO RUN THIS COMMAND REPEATEDLY, ADD THIS IN package.json:
    "scripts": {
    "tailwind":"npx tailwindcss -i ./static/src/input.css -o ./static/css/output.css --watch"
  }
  and then run the following command in the terminal:
  npm run tailwind

5. Install mySQL: 
    Make the database named Inventory and a sheet named store.
    ```bash
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Working@2024",
    database = "Store"
    )

    mycursor = mydb.cursor()
    mycursor.execute("CREATE DATABASE Store")
    mycursor.execute("CREATE TABLE data (`Sr. No.` INT,Email_ID VARCHAR(255), Question TEXT, SignalVerse_Answer TEXT, Rating INT, Raw_AI_Response TEXT, Rating2 INT)")
    ```

### Usage

1. Run the Flask application:

    ```bash
    python3 main.py
    ```

2. Access the application through a web browser at `trafficsignalverse.com.

## Components

### Backend

- **Flask**: Python web framework used for handling HTTP requests and responses.
- **Pandas**: Library for data manipulation and analysis.
- **openpyxl**: Library for reading and writing Excel files.
- **langchain**: Library for language processing tasks such as text splitting and embeddings.

### Frontend

- **HTML/CSS**: Markup and styling for the user interface.
- **Tailwind CSS**: Utility-first CSS framework for designing responsive web applications.

### Features

- Users can ask questions and receive answers from the chatbot.
- Chat history is displayed and can be cleared.
- Users can rate the answers provided by the chatbot, which get stored in the Inventory database.

## File Structure

- `main.py`: Flask application and backend logic.
- `requirements.txt`: Python dependencies.
- `templates/`: HTML templates for rendering pages.
- `static/`: Static files including CSS and images.

## Deploying SQL on ec2 instance

### Step 1: Update the system

sudo apt update

### Step 2: Install MySql

sudo apt install mysql-server

### Step 3: Check the Status of MySql (Active or Inactive)

sudo systemctl status mysql

### Step 4: Login to MySql as a root

sudo mysql

### Step 5: Update the password for the MySql Server

ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'place-your-password-here';

FLUSH PRIVILEGES;

### Step 6: Test the MySql server if it is working by running sample sql queries

CREATE DATABASE mysql_test;

USE mysql_test;

CREATE TABLE table1 (id INT, name VARCHAR(45));

INSERT INTO table1 VALUES(1, 'Virat'), (2, 'Sachin'), (3, 'Dhoni'), (4, 'ABD');

SELECT * FROM table1;


