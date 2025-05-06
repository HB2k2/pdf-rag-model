from bot import ChatBot
import streamlit as st

# Initialize the bot
bot = ChatBot()

# Streamlit page setup
st.set_page_config(page_title="Random Book Summarizer Bot")
with st.sidebar:
    st.title('Random Book Summarizer Bot')

# Function for generating LLM response
def generate_response(input_text):
    # Use the new answer_question method to get the response
    response = bot.answer_question(input_text)
    print(response)  # Optional: For debugging
    return response

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's go through the book pages together!"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input_text := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.write(input_text)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer from the book..."):
                response = generate_response(input_text)
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
