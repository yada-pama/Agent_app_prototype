import os
import pandas as pd
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from H_datahandle_app import DataHandler
import re

class PandasAgent:
    def __init__(self, temperature: float, base_url: str, model_name: str, dataset_paths: dict):
        self.handler = DataHandler(dataset_paths=dataset_paths)
        self.handler.load_data()
        self.handler.preprocess_data()

        self.temperature = temperature
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = os.getenv("PANDAS_API_KEY")
        self.llm = self.initialize_llm()

    def initialize_llm(self) -> ChatOpenAI:
        """Initialize the language model."""
        if not self.api_key:
            raise ValueError("API key is missing. Ensure 'PANDAS_API_KEY' is set in your environment.")
        return ChatOpenAI(
            base_url=self.base_url,
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
        )

    def create_agent(self, df_key: str):
        """Create an agent for the specified dataset."""
        if df_key not in self.handler._data:
            raise ValueError(f"Dataset '{df_key}' not found.")
        
        df = self.handler.get_data(df_key)
        # suffix = (
        #     f"You are working with a DataFrame. Columns are: {', '.join(df.columns)}. "
        #     "Use proper syntax to access and manipulate data."
        # )
        suffix = (
            f"""
    You are an expert Python programmer specializing in data processing and analysis. Your main responsibilities include:
    You are working with a DataFrame. Columns are: {', '.join(df.columns)}.
    Use proper syntax to access and manipulate data.
    1. Writing clean, efficient Python code for data manipulation, cleaning, and transformation.
    2. Implementing statistical methods and machine learning algorithms as needed.
    3. Debugging and optimizing existing code for performance improvements.
    4. Adhering to PEP 8 standards and ensuring code readability with meaningful variable and function names.

    Constraints:
    - Focus solely on data processing tasks; do not write non-Python code.
    - Provide only valid, executable Python code, including necessary comments for complex logic.
    - Avoid unnecessary complexity; prioritize readability and efficiency.
    """
        )

        return create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            suffix=suffix,
            verbose=False,
            allow_dangerous_code=True,  
        )

    def extract_code_snippet(self, response: str) -> str:
        """Extract Python code from agent response."""
        match = re.search(r'```(?:python|code)?\n(.*?)\n```', response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    def execute_code(self, code: str, context: dict):
        """Safely execute Python code."""
        try:
            exec(code, context)
        # except Exception as e:
        #     logging.error(f"Error executing code: {e}")
        except:
            pass
    def run(self, query: str, dataset_key: str):
        """Handle user interactions."""
        logging.info("Available datasets: %s", ", ".join(self.handler._data.keys()))
        
        try:
            agent = self.create_agent(dataset_key)
            response = agent.invoke({"input": query})
            
            # Separate code snippet and explanation
            code_snippet = self.extract_code_snippet(response["output"])
            explanation = response["output"].replace(f"```python\n{code_snippet}\n```", "").strip()
            
            print("\n--- Explanation ---")
            print(explanation)
            print("\n--- Python Code ---")
            print(code_snippet)
            print("\n--- Executing Code ---")
            context = {"pd": pd, "df": self.handler.get_data(dataset_key)}
            self.execute_code(code_snippet, context)
        except Exception as e:
            logging.error(f"An error occurred: {e}")

    # def run(self, query: str, datasetkey: str):
    #     """Handle user interactions."""
    #     logging.info("Available datasets: %s", ", ".join(self.handler._data.keys()))

    #     try:
    #         dataset_key = datasetkey
    #         agent = self.create_agent(dataset_key)
    #         response = agent.invoke({"input": query})

    #         code_snippet = self.extract_code_snippet(response["output"])
    #         logging.info(f"Executing Code:\n{code_snippet}")
                
    #         context = {"pd": pd, "df": self.handler.get_data(dataset_key)}
    #         self.execute_code(code_snippet, context)

    #     # except Exception as e:
    #     #     logging.error(f"An error occurred: {e}")
    #     except:
    #         pass

    # def run(self):
    #     """Handle user interactions."""
    #     logging.info("Available datasets: %s", ", ".join(self.handler._data.keys()))
    #     while True:
    #         query = input("Enter your query ('stop' to quit): ").strip()
    #         if query.lower() == "stop":
    #             logging.info("Goodbye!")
    #             break

    #         try:
    #             dataset_key = input("Specify the dataset key (e.g., 'df1'): ").strip()
    #             agent = self.create_agent(dataset_key)
    #             response = agent.invoke({"input": query})

    #             code_snippet = self.extract_code_snippet(response["output"])
    #             logging.info(f"Executing Code:\n{code_snippet}")
                
    #             context = {"pd": pd, "df": self.handler.get_data(dataset_key)}
    #             self.execute_code(code_snippet, context)
    #         except Exception as e:
    #             logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    load_dotenv()
    file_paths = {
        "Financials": "./Financials.csv",
        "McDonald_s_Reviews": "./McDonald_s_Reviews.csv"
    }

    agent = PandasAgent(
        temperature=0.1,
        base_url="https://api.opentyphoon.ai/v1",
        model_name="typhoon-v1.5x-70b-instruct",
        dataset_paths=file_paths
    )
    agent.run()
