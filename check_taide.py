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
    # 使用純 Python 實現代替 Rust 綁定
    client = chromadb.PersistentClient(path="./chroma_db", settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=True))
    collection = client.get_or_create_collection(name="dialogues")
    
    dialogues = read_text_files(folder_path)
    
    existing_data = collection.get()
    existing_ids = existing_data.get("ids", [])
    
    if existing_ids:
        collection.delete(ids=existing_ids)
    
    for idx, content in enumerate(dialogues):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=content)
        collection.add(ids=[str(idx)], embeddings=[response["embedding"]], documents=[content])
    
    st.session_state.collection = collection
    st.session_state.already_executed = True

def initialize():
    if "already_executed" not in st.session_state:
        st.session_state.already_executed = False
    
    if not st.session_state.already_executed:
        folder_path = "d:/Lab/TsengLab/RAG/KiW2RAG_test"  # 使用當前資料夾
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
            data = results["documents"][0]
            prompt = f"根據以下資訊回答問題：\n\n{data}\n\n問題：{user_input}\n請用中文回答。"
        else:
            model_name = "hf.co/chtseng/TAIDE-Medicine-QA-TW-Q6"
            prompt = f"此問題與對話並無明確相關，改為採用 {model_name} 來回答本問題：{user_input}"
    except RuntimeError as e:
        st.error(f"查詢時發生錯誤：{e}")
        return
    
    output = ollama.generate(model="hf.co/chtseng/TAIDE-Medicine-QA-TW-Q6", prompt=prompt)
    st.text("回答：")
    st.write(output["response"])

if __name__ == "__main__":
    main()