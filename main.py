import base64
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

os.environ["OPENAI_API_KEY"] = "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"

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    set_background('cool-background.png')
    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        #message = st.chat_message("assistant")
        #message.write("It's a smart PDF assistant. You can ask me a question about your PDF here!")
        # Store LLM generated responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": """It's a smart PDF assistant.
                                                                          You can ask me a question about your PDF here!
                                                                          If you want to stop the conversation, press the button
                                                                          below the chat!"""}]
        #message1 = st.chat_message("user")
        # show user input

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("Ask here")
        #user_question = st.text_input("Ask a question about your PDF:")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            #message.write(prompt)
            docs = knowledge_base.similarity_search(prompt)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=prompt)
                print(cb)

            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        docs = knowledge_base.similarity_search(prompt)
                        llm = OpenAI()
                        chain = load_qa_chain(llm, chain_type="stuff")
                        with get_openai_callback() as cb:
                            response = chain.run(input_documents=docs, question=prompt)
                            print(cb)
                        st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

            #message.write(response)
        stop_button = st.button("Press here to end the conversation")
        if stop_button:
            st.experimental_rerun()

if __name__ == '__main__':
    main()