import streamlit as st
import io
import sys
import os
from dotenv import load_dotenv
from H_typhoon_app import TyphoonAgent
import matplotlib.pyplot as plt

def main():
    load_dotenv()

    # ตั้งค่า session_state
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = []
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    #if "username" not in st.session_state:
    #    st.session_state.username = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}

    def start_new_session():
        st.session_state.current_session = None

    # ฟังก์ชันเพิ่มข้อความในเซสชันปัจจุบัน
    def add_to_current_session(role, content):
        #timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if st.session_state.current_session is not None:
            st.session_state.chat_sessions[st.session_state.current_session]["history"].append(
            {"role": role, "content": content, } #"timestamp": timestamp
            )

    # Check if there are valid files
    #if not file_paths:
    #    st.error("Please upload file first.")
    #    return
    
    # Check if there are api_key provided
    #if not api_key:
    #    st.error("Please enter API Key first.")
    #    return
    

    ###main app 
    st.title("Streamlit Chat App with Pandas Agent")
    st.write("Chat with the DataFrame to analyze or interact with it.")

    # sidebar - model settings
    with st.sidebar.expander("⚙️ Model Settings", expanded=True):
        model_options = {
            "typhoon": "typhoon-v1.5x-70b-instruct",
            "gpt-4o-mini": "GPT-4o Mini",
            "llama-3.1-405b": "Llama 3.1 405B",
            "llama-3.2-3b": "Llama 3.2 3B",
            "Gemini Pro 1.5": "Gemini Pro 1.5",
        }
        model = st.selectbox("Choose your AI Model:", model_options.values())
        temp = st.slider('Set temperature',min_value=0.0, max_value=1.0,value=0.1)
    
        api_key = st.text_input("API Key", type="password")
        st.session_state["api_key"] = api_key

    #side bar - file upload
    with st.sidebar.expander("📂 File Upload", expanded=True):
        # ส่วนสำหรับอัปโหลดไฟล์ใน Sidebar
        #st.sidebar.markdown("### 📂 File Upload")
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

        uploaded_file_paths = []  # เปลี่ยนจาก dict เป็น list
        if uploaded_files:
            #st.sidebar.markdown("### 📂 File Upload")
            for file in uploaded_files:
                #st.sidebar.write(f"- {file.name}")
                # สร้างพาธชั่วคราวสำหรับไฟล์ที่อัปโหลด
                file_path = f"./{file.name}"  # เก็บ path ไฟล์ในตัวแปร
                #with open(file_path, "wb") as f:
                #    f.write(file.getbuffer())
                uploaded_file_paths.append(file_path)  # เพิ่ม path ไฟล์ใน list

        # File paths (adjust based on your setup)
        filepaths = uploaded_file_paths #['./McDonald_s_Reviews.csv', './Financials.csv']

        # Helper function to map file names to paths
        def get_filepath(filepaths: list) -> dict:
            valid_paths = {}
            for filepath in filepaths:
                if os.path.exists(filepath):
                    valid_paths[filepath.split('/')[-1]] = filepath
                else:
                    st.warning(f"File not found: {filepath}")
            return valid_paths

        file_paths = get_filepath(filepaths)

        # Dataset selection
        dataset_key = st.sidebar.selectbox("Select a dataset", file_paths.keys())

    def response_generator():
        
        agent = TyphoonAgent(
            temperature=temp,
            base_url="https://api.opentyphoon.ai/v1", #base_url must be change by model name
            model_name=model,
            api_key=api_key,
            dataset_paths=file_paths,
            dataset_key=dataset_key
        )

        response = agent.agent_executor.invoke({"input": user_input}) #.run(user_input=user_input)#.agent_executor.invoke({"input": user_input})
        return response['output']
    
    #sidebar - chat history
    st.sidebar.title("Chat History")

    # ปุ่มเริ่มต้นเซสชันใหม่
    if st.sidebar.button("Start New Chat"):
        if st.session_state.current_session is not None:
            # บันทึกเซสชันเก่าด้วย title จากข้อความแรก
            if len(st.session_state.chat_sessions[st.session_state.current_session]["history"]) > 0:
                first_message = st.session_state.chat_sessions[st.session_state.current_session]["history"][0]["content"]
                st.session_state.chat_sessions[st.session_state.current_session]["title"] = first_message
        start_new_session()

    # แสดงรายการเซสชันใน Sidebar โดยแชทใหม่อยู่ข้างบน
    if st.session_state.chat_sessions:
        for idx, session in reversed(list(enumerate(st.session_state.chat_sessions))):
            title = session.get("title", f"Session {idx + 1}")
            if st.sidebar.button(title, key=f"session_{idx}"):
                st.session_state.current_session = idx

    ###homepage
    col1, col2 = st.columns([175, 100])

    #chat
    with col1:
        chat_container = st.container()
        user_input = st.chat_input("Type your message here...")

        if user_input:
            # หากไม่มีเซสชัน เริ่มเซสชันใหม่
            if st.session_state.current_session is None:
                st.session_state.chat_sessions.append({"title": "", "history": []})
                st.session_state.current_session = len(st.session_state.chat_sessions) - 1

            # เพิ่มข้อความใหม่ในเซสชันปัจจุบัน
            add_to_current_session("user", user_input)

            # Redirect stdout to capture agent's output
            #old_stdout = sys.stdout
            #sys.stdout = new_stdout = io.StringIO()

            try:
                output = response_generator()
            except Exception as e:
                output = f"An error occurred: {e}"
            
            add_to_current_session("assistant", output)
    
            # Check for plots in the output
            if "plt" in output or "figure" in output:
                st.session_state.chat_sessions.append({"role": "assistant", "content": "Generated a plot."})

    with chat_container:
        if st.session_state.current_session is not None:
            session = st.session_state.chat_sessions[st.session_state.current_session]
            for chat in session["history"]:
                message_alignment = "flex-end" if chat["role"] == "user" else "flex-start"
                message_background = "#e1f5fe" if chat["role"] == "user" else "#f0f0f0"
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: {message_alignment}; margin-bottom: 10px;">
                        <div style="background-color: {message_background}; padding: 10px; border-radius: 8px; max-width: 80%; word-wrap: break-word;">
                            {chat['content']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
    #chat log
    with col2:
        with st.expander("Console", expanded=False):
           st.write("No chat logs available now.")

    # Display chat chat_sessions
    #for message in st.session_state.chat_sessions:
    #    if message["role"] == "user":
    #        with st.chat_message("user"):
    #            st.write(message["content"])
    #    elif message["role"] == "assistant":
    #        with st.chat_message("assistant"):
    #            if "--- Explanation ---" in message["content"]:
    #                parts = message["content"].split("--- Python Code ---")
    #                explanation, code_snippet = parts[0], parts[1] if len(parts) > 1 else ""

                    # Show explanation
    #                st.subheader("Explanation")
    #                st.write(explanation.replace("--- Explanation ---", "").strip())

    #                try:
    #                    if code_snippet:
    #                        st.subheader("Generated Python Code")
    #                        st.code(code_snippet.strip())
    #                except:
    #                    pass
                
    #            else:
    #                parts = message["content"]
    #                st.write(parts.replace("--- Agent respond ---", "").strip())



    #            if "Generated a plot." in message["content"]:
    #                try:
                        # Display the current Matplotlib figure
    #                    fig = plt.gcf()
    #                    st.pyplot(fig)
    #                except Exception as e:
    #                    st.write(f"Error rendering plot: {e}")


if __name__ == "__main__":
    main()
