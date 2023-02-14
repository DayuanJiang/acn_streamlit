import streamlit as st
from streamlit_chat import message
import requests
from app import init_chain, get_docstore, format_result_to_markdown
from langchain import OpenAI, PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain


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

def init_chain():
    combine_prompt_template = """
あなたは、アクセンチュアのAIアシスタントです。
あなたには、以下のような長いドキュメントの抜粋部分とアクセンチュアに関する質問が与えられています。
提供されたテキストを参考して答えてください。
また、提供されたテキストに根拠がない場合、もしくは全部なしの場合は、わからないと答えなさい。答えを作り上げないでください。
なお、質問には完全な文でお答えください。
=========
テキスト：{summaries}
=========
質問: {question}
=========
答案:"""
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )
    chain = load_qa_with_sources_chain(OpenAI(temperature=0), 
                                chain_type="stuff", 
                                prompt=COMBINE_PROMPT,
                                verbose=True,
                                )
    return chain

if __name__ == "__main__":
    st.title("ACN AIアシスタント(超爆速版※)")
    st.subheader("ACN　HPをベースとしているため、HPにある内容のみ回答します。")
    st.text("※超爆速とは、開発スピードを指しています。Botの反応速度ではありません。")
    st.markdown(
        """
## 質問の例：
- アクセンチュアのAIグループのリーダーは誰ですか？
- アクセンチュアのCEOは誰ですか？
- アクセンチュアの社長は誰ですか？
- アクセンチュアの事業内容は？
- アクセンチュアの事業所在地は？
- アクセンチュア社員の有給は何日ありますか？
        """
    )
    query = st.text_input("User Input")
    vector_store = get_docstore()
    chain = init_chain()
    # llm = OpenAI(temperature=0, model_name="text-curie-001")
    llm = OpenAI(temperature=0)

    template = """
下記の質問をアクセンチェアに関する質問に書き換えてください。
質問: {question}
書き換えた質問:
"""

    query_temp = PromptTemplate(
        template=template, input_variables=["question"]
    )
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
                formated_query_prompt = query_temp.format(question= query)
                rephrased_query = llm(formated_query_prompt)
                

            docs = vector_store.similarity_search(rephrased_query, k=2)
            print(len(docs))
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