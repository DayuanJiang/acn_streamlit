from typing import Dict

import streamlit as st
import pickle
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import os

from langchain import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


# with open("data/key.txt") as f:
#     openai_key = f.read().strip()
# os.environ["OPENAI_API_KEY"] = openai_key

@st.cache
def get_docstore():
    with open("data/acn_homepage_faiss_store_1000_tokens.pickle", "rb") as f:
        store = pickle.load(f)
    return store


def init_chain():
    question_prompt_template = """長い文書の次の部分を使って、質問に答えるために関連するテキストがあるかどうかを確認します。
    関連するテキストがあれば、そのテキストを返す。なかったら、「関連情報なし」と返さないてください。

    {context}

    質問: {question}
    関連するテキスト:"""
    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )

    combine_prompt_template = """
    あなたは、アクセンチュアのAIアシスタントです。
    あなたには、以下のような長いドキュメントの抜粋部分と質問が与えられています。
    提供されたテキストを参考して答えてください。
    提供されたテキストに根拠がない場合は、わからないと答えなさい。答えを作り上げないでください。
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
                                    chain_type="map_reduce", 
                                    return_intermediate_steps=True, 
                                    question_prompt=QUESTION_PROMPT, 
                                    combine_prompt=COMBINE_PROMPT,
                                    verbose=True
                                    )
    return chain

# format the result to markdown format
def format_result_to_markdown(result: dict):
    result_markdown = "# Source \n"
    result_markdown += f"""
# Answer
{result["output_text"].strip()}
    """
    for i, intermediate_step in enumerate(result["intermediate_steps"]):
        link = result["input_documents"][i].metadata["source"]
        result_markdown += f"""
**Link**: {link}  
**Extracted Information**: {intermediate_step}
        """

    return result_markdown

if __name__ == "__main__":
    st.title("ACN AIアシスタント(超爆速版※)")
    st.subheader("ACN　HPをベースとしているため、HPにある内容のみ回答します。")
    st.text("※超爆速とは、開発スピードを指しています。Botの反応速度ではありません。")
    st.markdown(
        """
## 質問の例：
- アクセンチュアのAI領域のリーダーは誰ですか？
- アクセンチュアのCEOは誰ですか？
- アクセンチュアの社長は誰ですか？
- アクセンチュアの事業内容は？
- アクセンチュアの事業所在地は？
- アクセンチュア社員の有給は何日ありますか？
        """
    )
    docsearch = get_docstore()
    chain = init_chain()
    question = st.text_input("質問を入力してください", "")

    answer_slot = st.empty()
    with st.spinner("答案を検索中..."):
        if st.button("Submit"):
            try:
                query = question.strip()
                # st.write("len(docsearch):", len(docsearch))
                # st.write("help(docsearch.index):", help(docsearch.index))
                # st.write("help(docsearch.index.search):", help(docsearch.index.search))
                st.write(len(docsearch.index_to_docstore_id))
                docs = docsearch.similarity_search(query, k=3)
                result = chain(
                    {"input_documents": docs, "question": query},
                )

                answer_slot.markdown(format_result_to_markdown(result))
            except Exception as e:
                st.write(e)
