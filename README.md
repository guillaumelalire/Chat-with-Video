# Chat with Video🎬

## Project Overview

This project enhances a Large Language Model (LLM) by integrating Retrieval-Augmented Generation (RAG) to answer questions about a given video using its transcript.

https://github.com/user-attachments/assets/2dc20ccd-c4f7-4699-8f93-d0bb0ec03b7c

The process involves the following steps:

1. **Video Input 🎥**: The user uploads a video file or provides a YouTube link.

2. **Transcription 📝**: The audio is extracted and transcribed using Whisper. If necessary, the transcript is translated into English.

3. **Vector Database Creation 🗃️**: The transcription is split into chunks, which are used to build a vector database for RAG.

4. **Model Deployment 🤖**: The Mistral-7B-Instruct model is used for generating answers. This compact model was selected for its effectiveness in delivering accurate responses while being small enough to run locally.

5. **Contextual Retrieval 🔍**: Langchain's RAG-chain connects context from RAG to the model via a prompt. The prompt format is designed to ensure clear responses based only on the provided context.

6. **User Interface 🖥️**: The demo was built using the Streamlit library, known for its simplicity in building interactive, data-driven demos.

## Getting Started

To run the demo, follow these simple steps:

### 1. Generate a Hugging Face Access Token

To load the Mistral-7B-Instruct model, you need a Hugging Face access token:

1. Go to the Hugging Face website and log in.
2. Navigate to the Access Tokens page and generate a new token.
3. Accept the use of the Mistral-7B-Instruct model.
4. Add the following line to the .env file at the root of the project:
```
HUGGINGFACE_TOKEN=your_huggingface_token
```

### 2. **Install the Requirements**

First, ensure you have all the necessary dependencies installed. You can do this by running the following command in the main folder:
```
pip install -r requirements.txt
```

### 3. **Start the Project**

Once the dependencies are installed, you can start the project by launching the Streamlit demo. Run the following command:
```
streamlit run src/app.py
```

### 4. **Access the Application**

After the project has started, open your web browser and navigate to the following URL to access the application: [http://localhost:8501](http://localhost:8501)
