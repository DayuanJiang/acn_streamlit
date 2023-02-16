from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain import FAISS

from typing import Dict, List, Tuple

import os


# with open("../data/key.txt") as f:
#     key = f.read()
# os.environ["OPENAI_API_KEY"] = key

VERBOSE = True

llm = OpenAI(temperature=0)

"""
会話の履歴を参考して質問を言い換えるためのチェイン
"""

next_quesiton_template = """あなたはアクセンチュアのAIアシスタントとして、人と会話をしています。
人が次の質問をしてます。会話履歴を参考しながら、その質問を言い換えなさい。
質問はアクセンチュアに関連するものであると仮定してください。
また、次の質問が日本語ではない場合は、言い換えた質問を日本語に翻訳してください。
会話履歴:
{chat_history}
次の質問: {question}
言い換えた質問:"""

next_question_prompt_template = PromptTemplate(
    template=next_quesiton_template.strip(),
    input_variables=["chat_history", "question"],
)

next_question_chain = LLMChain(
    llm=llm, prompt=next_question_prompt_template, verbose=VERBOSE
)


history_cell_format = """
質問者: {rephrased_question}({orig_question})
アシスタント：{translated_answer}( {orig_answer})
"""

def form_chat_history(history: List[Dict]):
    histroy_list = [history_cell_format.format(**h) for h in history]
    history_text = "" .join(histroy_list)
    return history_text


"""
コアとなる質問応答チェイン
"""
qc_template = """
あなたは、アクセンチュアのAIアシスタントです。
あなたには、以下のような長いドキュメントの抜粋部分とアクセンチュアに関する質問が与えられています。
提供されたテキストを参考して答えてください。
また、提供されたテキストに根拠がない場合は、わからないと答えなさい。答えを作り上げないでください。
なお、質問には完全な文で答えください。
=========
テキスト：{summaries}
=========
質問: {question}
=========
答案:
"""
qc_prompt_template = PromptTemplate(
    template=qc_template.strip(), input_variables=["summaries", "question"]
)
qc_chain = load_qa_with_sources_chain(
    llm=llm, prompt=qc_prompt_template, verbose=VERBOSE
)

"""
回答を翻訳するためのチェイン
"""
answer_translate_template = """
    あなたは、アクセンチュアの多言語を対応できるAIアシスタントです。現在、人と会話をしています。
    人が次の質問をしています。その回答を{language}に翻訳してください。
    質問: {query}
    ===
    日本語の回答: {text}
    翻訳した回答:
    """

answer_translate_prompt_template = PromptTemplate(
    template=answer_translate_template, input_variables=["text", "language", "query"]
)
answer_translate_chain = LLMChain(
    llm=llm, prompt=answer_translate_prompt_template, verbose=VERBOSE
)


class ChatBot:

    """
    一連のチェインをまとめて質問を回答するのチャットボット
    """

    def __init__(self, vector_score_path, language="日本語"):
        self.vector_score = FAISS.load_local(vector_score_path, OpenAIEmbeddings())
        self.language = language
        self.history = []

    def query(self, query_text):
        """
        処理の流れ：
        1. 質問を言い換える
        2. 言い換えた質問に対して、質問応答チェインを適用
        3. 質問応答チェインの出力を翻訳チェインに入力
        4. 翻訳チェインの出力を返す
        """
        rephrased_query_result = next_question_chain(
            {"chat_history": form_chat_history(self.history), "question": query_text}
        )
        rephrased_query = rephrased_query_result["text"].strip()

        docs = self.vector_score.similarity_search(rephrased_query, k=2)

        result = qc_chain(
            {
                "input_documents": docs,
                "question": rephrased_query,
            },
        )

        answer = result["output_text"].strip()

        if self.language == "日本語":
            translated_answer = answer
        else:
            translated_answer = answer_translate_chain(
                {
                    "text": answer,
                    "language": self.language,
                    "query": query_text,
                }
            )["text"].strip()
    
        self.history.append(
            {
                "orig_question": query_text,
                "rephrased_question": rephrased_query,
                "orig_answer": answer,
                "translated_answer": translated_answer,
            }   
        )

        return translated_answer



# chatbot = ChatBot(
#     "../model/data/acn_homepage_faiss_store_1000_tokens_orig", language="英語"
# )
# chatbot.query("who is the CEO of Accenture?")
# chatbot.query("What did he do after joining the company?")
# chatbot.query("when did he joined Accenture?")

# chatbot.query("Who is the leader of AI group?")