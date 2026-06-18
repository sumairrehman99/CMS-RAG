import requests
import gradio as gr

URL = 'http://localhost:8000/ask'

def chat(question, history):
    # convert input in JSON
    json = {
        'question':question,
        'history':history
    }

    response = requests.post(URL, json=json)    # Send json to URL
    response.raise_for_status()     # Raise error if occurs

    reply = response.json()

    return reply['answer']


if __name__ == '__main__':
    gr.ChatInterface(chat, title='CMS RAG Assistant',
                     type='tuples').launch()      # Launch Gradio Interface