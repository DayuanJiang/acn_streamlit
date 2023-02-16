import streamlit as st
from streamlit_chat import message
from src import ChatBot


faiss_store_path = "data/acn_homepage_faiss_store_1000_tokens_orig"


language_dic = {
    "日本語": "日本語",
    "English": "英語",
    "中文": "中国語",
    "한국어": "韓国語",
    "Français": "フランス語",
    "Deutsch": "ドイツ語",
    "Español": "スペイン語",
    "Italiano": "イタリア語",
    "Português": "ポルトガル語",
    "Русский": "ロシア語",
    "Türkçe": "トルコ語",
    "Nederlands": "オランダ語",
    "Polski": "ポーランド語",
    "العربية": "アラビア語",
}

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

language = st.selectbox("Please select the language", list(language_dic.keys()))

if "bot" not in st.session_state:
    st.session_state.bot = ChatBot(faiss_store_path, language=language_dic[language])
for key in ["generated", "past"]:
    if key not in st.session_state:
        st.session_state[key] = []

query_text = st.text_input("User Input")

with st.spinner("考え中..."):
    if query_text:
        st.session_state.bot.language = language_dic[language]
        result = st.session_state.bot.query(query_text)
        st.session_state.past.append(query_text)
        st.session_state.generated.append(result)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
