import streamlit as st
from streamlit_chat import message
import requests
from app import init_chain, get_docstore, format_result_to_markdown
from langchain import OpenAI, PromptTemplate


from typing import Dict, List, Tuple
for key in ['history', 'generated', "past"]:
    if key not in st.session_state:
        st.session_state[key] = []

def query(payload):
	# response = requests.post(API_URL, headers=headers, json=payload)
	# return response.json()
    return {"generated_text": "I am fine, thank you!"}

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 

def format_result_to_markdown(result: dict):
    result_markdown = ""
    result_markdown += f"""
{result["output_text"].strip()}
    """
    for i, intermediate_step in enumerate(result["intermediate_steps"]):
        link = result["input_documents"][i].metadata["source"]
        result_markdown += f"""
\nLink: {link}  
\nExtracted Information: {intermediate_step}
        """

    return result_markdown


def _get_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = ""
    for human_s, ai_s in chat_history:
        human = f"質問者: " + human_s
        ai = f"アシスタント: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer

_template = """あなたはアクセンチュアのAIアシスタントとして、人と会話をしています。
人が次の質問をしてます。会話履歴を参考しながら、その質問に言い換えなさい。
質問はアクセンチュアに関連するものであると仮定してください。
会話履歴:
{chat_history}
次の質問: {question}
独立した質問:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


if __name__ == "__main__":
    query = st.text_input("User Input")
    vector_store = get_docstore()
    chain = init_chain()
    # llm = OpenAI(temperature=0, model_name="text-curie-001")
    llm = OpenAI(temperature=0)
    with st.spinner("考え中..."):
        if query:
            print("history: ", st.session_state.history)
            if st.session_state.history:
                chat_history = _get_chat_history(st.session_state.history)
                formated_query_prompt = CONDENSE_QUESTION_PROMPT.format(
                    chat_history=chat_history, question=query, 
                )
                rephrased_query = llm(formated_query_prompt)
            else:
                rephrased_query = query
                

            docs = vector_store.similarity_search(rephrased_query, k=10)
            docs = [doc for doc in docs if len(doc.page_content) > 200][:3]
            result = chain(
                {"input_documents": docs, "question": rephrased_query},
            )
            result["formatted_query"] = query
            st.session_state.history.append((query, result["output_text"]))

            st.session_state.past.append(query)
            st.session_state.generated.append(result["output_text"].strip())

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')