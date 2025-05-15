import streamlit as st
import chromadb
import ollama
import os


def read_text_files(folder_path):
    dialogues = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # 只讀取 .txt 檔案
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if ":" in line and len(line.split(":")) >= 2:
                        _, content = line.split(":", 1)
                        dialogues.append(content.strip())
                    else:
                        dialogues.append(line)
    
    return dialogues

def setup_database(folder_path):
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=True)
    )
    collection = client.get_or_create_collection(name="dialogues")

    dialogues = []
    current_question = ""
    current_answer = ""

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            print(f"🔍 正在讀取: {file_path}")
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("問題:") or line.startswith("問題：") or line.startswith("問:") or line.startswith("問："):
                        current_question = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                    elif line.startswith("回答:") or line.startswith("回答：") or line.startswith("答:") or line.startswith("答："):
                        current_answer = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                        if current_question and current_answer:
                            dialogues.append((current_question, current_answer))
                            current_question = ""
                            current_answer = ""


    # 清空舊資料
    existing_data = collection.get()
    existing_ids = existing_data.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    # 加入向量資料庫
    for idx, (question, answer) in enumerate(dialogues):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=question)
        collection.add(
            ids=[str(idx)],
            embeddings=[response["embedding"]],
            documents=[answer],
            metadatas=[{"question": question}]
        )

    print(f"✅ 已加入 {len(dialogues)} 筆 QA 到向量資料庫")
    st.session_state.collection = collection
    st.session_state.already_executed = True




def initialize():
    if "already_executed" not in st.session_state:
        st.session_state.already_executed = False
    
    if not st.session_state.already_executed:
        folder_path = "d:/Lab/TsengLab/RAG/KiW2RAG_test/db"  # 使用當前資料夾
        setup_database(folder_path)

def main():
    initialize()
    st.title("我的第一個LLM+RAG本地知識問答")
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    user_input = st.text_area("您想問什麼？", st.session_state.user_input)
    
    if st.button("送出"):
        if user_input:
            handle_user_input(user_input, st.session_state.collection)
            st.session_state.user_input = ""  # 清空輸入框
        else:
            st.warning("請輸入問題！")


def handle_user_input(user_input, collection):
    response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")
    print(f"Embedding shape: {len(response['embedding'])}")  # 檢查嵌入向量的維度
    try:
        results = collection.query(query_embeddings=[response["embedding"]], n_results=3)
        if results["documents"]:
            answers = [doc for doc in results["documents"][0]]
            prompt = f"根據以下資訊回答問題：\n\n" + "\n\n".join(answers) + f"\n\n問題：{user_input}，請用中文回答。"

        else:
            model_name = "hf.co/chtseng/TAIDE-Medicine-QA-TW-Q6"
            prompt = f"此問題與對話並無明確相關，改為採用 {model_name} 來回答本問題：{user_input}"
    except RuntimeError as e:
        st.error(f"查詢時發生錯誤：{e}")
        return
    
    output = ollama.generate(model="hf.co/chtseng/TAIDE-Medicine-QA-TW-Q6", prompt=prompt)
    st.text("回答：")
    st.write(output["response"])

    st.markdown("#### 🔍 檢索到的內容")
    for i, doc in enumerate(results["documents"][0]):
        st.markdown(f"{i+1}. {doc}")

if __name__ == "__main__":
    main()