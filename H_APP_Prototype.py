import streamlit as st
import io
import sys
import os
from dotenv import load_dotenv
from H_typhoon_app import TyphoonAgent
import matplotlib.pyplot as plt

def main():
    load_dotenv()

    st.title("Streamlit Chat App with Pandas Agent")
    st.write("Chat with the DataFrame to analyze or interact with it.")

    # File paths (adjust based on your setup)
    filepaths = ['./McDonald_s_Reviews.csv', './Financials.csv']

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

    # Check if there are valid files
    if not file_paths:
        st.error("No valid files found. Please check the file paths.")
        return

    st.sidebar.header('Agent option')
    # Temperature selection
    temp = st.sidebar.select_slider(
        'Set temperature',
        options=[round(i * 0.1, 1) for i in range(0, 11)],
        value=0.1
    )

    # Dataset selection
    dataset_key = st.sidebar.selectbox("Select a dataset", file_paths.keys())

    # Initialize the TyphoonAgent
    agent = TyphoonAgent(
        temperature=temp,
        base_url="https://api.opentyphoon.ai/v1",
        model_name="typhoon-v1.5x-70b-instruct",
        dataset_paths=file_paths,
        dataset_key=dataset_key
    )

    # Chat history container
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Add user query
    user_query = st.chat_input("Enter your query")

    if user_query:
        if user_query.strip():
            st.session_state.messages.append({"role": "user", "content": user_query})

            # Redirect stdout to capture agent's output
            old_stdout = sys.stdout
            sys.stdout = new_stdout = io.StringIO()

            try:
                agent.run(user_input=user_query)
                output = new_stdout.getvalue()
            except Exception as e:
                output = f"An error occurred: {e}"
            finally:
                sys.stdout = old_stdout

            st.session_state.messages.append({"role": "assistant", "content": output})
            # Check for plots in the output
            if "plt" in output or "figure" in output:
                st.session_state.messages.append({"role": "assistant", "content": "Generated a plot."})

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                if "--- Explanation ---" in message["content"]:
                    parts = message["content"].split("--- Python Code ---")
                    explanation, code_snippet = parts[0], parts[1] if len(parts) > 1 else ""

                    # Show explanation
                    st.subheader("Explanation")
                    st.write(explanation.replace("--- Explanation ---", "").strip())

                    try:
                        if code_snippet:
                            st.subheader("Generated Python Code")
                            st.code(code_snippet.strip())
                    except:
                        pass
                
                else:
                    parts = message["content"]
                    st.write(parts.replace("--- Agent respond ---", "").strip())



                if "Generated a plot." in message["content"]:
                    try:
                        # Display the current Matplotlib figure
                        fig = plt.gcf()
                        st.pyplot(fig)
                    except Exception as e:
                        st.write(f"Error rendering plot: {e}")


if __name__ == "__main__":
    main()
