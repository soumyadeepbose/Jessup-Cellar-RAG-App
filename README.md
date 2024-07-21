# Jessup's Helper 🤖

Jessup's Helper is a Retrieval Based Chatbot that uses a pre-trained Llama3-8b model to generate responses to user input. Groq has been used for a near instant inference because of the high token speed, and Google Gen AI Studio API has been used to get the embeddings. The model uses a PDF file for information retrieval and generation of responses based on the context. The PDF file can be found in the 'pdfs' folder.

## Live Demo

The project is hosted on Streamlit and can be accessed [here](https://jessup-cellar.streamlit.app/).

## Getting Started

These instructions will guide you through getting a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites



First of, you will need to clone the repository to your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/soumyadeepbose/Jessup-Cellar-RAG-App.git
```

Next, you will need to install the required packages. You can do this by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

### Setting up the Environment Variables

You will need to set up the environment variables. For this, you need to create a `.env` file in the root directory of the project. The `.env` file should contain the following environment variables:

- `GOOGLE_API_KEY`: This is the API key for the Google Gen AI Studio API. You can get this by following the instructions [here](https://ai.google.dev/aistudio).

- `GROQ_API_KEY`: This is the API key for the Groq API. You can get this by following the instructions [here](https://groq.com/).

### Running the App

Now to run the app, you will need to run the following command in your terminal:

```bash
streamlit run app.py
```

## Author

- [Soumyadeep Bose 😊](https://www.linkedin.com/in/soumyadeepbose)