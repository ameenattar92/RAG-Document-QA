# RAG Document Q&A with LangChain, Groq, FAISS, and Streamlit

A Retrieval-Augmented Generation (RAG) application that lets you ask questions about research papers stored as PDF files. The app loads documents from a local folder, splits them into chunks, stores embeddings in FAISS, retrieves the most relevant chunks for a query, and uses a Groq-hosted LLM to generate grounded answers.

## Features

- Ask natural-language questions about PDF research papers
- Load multiple PDFs from a local directory
- Split large documents into retrievable chunks
- Store embeddings in a FAISS vector index for fast similarity search
- Generate answers using Groq models through LangChain
- Display retrieved document chunks alongside the final answer
- Simple interactive UI with Streamlit

## Tech Stack

- **Python**
- **Streamlit** for the web interface
- **LangChain** for retrieval and orchestration
- **FAISS** as the vector database
- **OpenAI Embeddings** for document embeddings
- **Groq** for fast LLM inference
- **PyPDFDirectoryLoader** for loading PDF files

## Project Structure

```bash
.
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env
└── research_papers/
    ├── paper1.pdf
    ├── paper2.pdf
    └── ...
```

## How It Works

1. PDFs are loaded from the `research_papers/` folder.
2. Documents are split into smaller chunks using `RecursiveCharacterTextSplitter`.
3. Embeddings are generated for the chunks.
4. Chunks are stored in FAISS for semantic retrieval.
5. When a user asks a question, relevant chunks are retrieved.
6. The retrieved context is passed to the Groq LLM.
7. The app returns an answer grounded in the uploaded documents.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Create a `.env` file in the project root and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

### 5. Add your PDF files

Create a folder named `research_papers` in the project root and place your PDF files inside it:

```bash
mkdir research_papers
```

Then add your `.pdf` files into that folder.

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Streamlit will start a local server and open the app in your browser.

## Usage

1. Launch the app.
2. Click **Document Embedding** to process the research papers.
3. Wait until the vector database is created.
4. Enter a question in the input box.
5. View the generated answer and the relevant retrieved document chunks.

## Example Questions

- What is the main contribution of this paper?
- Summarize the methodology used in these documents.
- What datasets are mentioned in the research papers?
- Compare the approaches discussed in the uploaded PDFs.
- What are the limitations described in the papers?

## Notes

- Make sure the `research_papers/` folder contains valid PDF files before creating embeddings.
- Do not commit your `.env` file or API keys to GitHub.
- If you use a Groq model name that has been deprecated, update it in `app.py` to a currently supported model.
- If you face import issues, ensure your virtual environment is active and all packages from `requirements.txt` are installed.

## License

This project is for learning and experimentation purposes.

## Acknowledgements

Built with:
- Streamlit 
- LangChain 
- Groq 
- FAISS