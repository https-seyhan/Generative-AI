import streamlit as st
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 🧠 MODEL INITIALISATION
# -----------------------------
def get_local_llm(model_name="llama3", temperature=0.2):
    return ChatOllama(model=model_name, temperature=temperature)

# -----------------------------
# 🏃 MAIN APP FUNCTION
# -----------------------------
def main():
    st.set_page_config(page_title="Copilot AI (Local)", page_icon="💻", layout="wide")
    st.title("🤖 Copilot AI (Local Edition)")
    st.write("Generate code locally using Ollama models — no API key required.")

    tab1, tab2, tab3 = st.tabs(["💻 Single Code Agent", "👥 Multi-Agent Crew", "🧠 Stateful Memory"])

    # -----------------------------
    # TAB 1: SINGLE CODE AGENT
    # -----------------------------
    with tab1:
        st.subheader("💻 Single Code Agent")
        code_query = st.text_area(
            "Describe the code you want to generate:", 
            "Write a Python function to sort a list of dictionaries by key 'age'."
        )

        if st.button("Generate Code", key="single_agent"):
            llm = get_local_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert coding assistant."),
                ("human", "{query}\nProvide a fully working code snippet.")
            ])
            chain = prompt | llm | StrOutputParser()
            code_output = chain.invoke({"query": code_query})
            
            st.success("Generated Code:")
            st.code(code_output, language="python")

    # -----------------------------
    # TAB 2: MULTI-AGENT CREW
    # -----------------------------
    with tab2:
        st.subheader("👥 Multi-Agent Crew Simulation")
        task1_desc = st.text_input("Task 1 (Coder):", "Write Python code for a Fibonacci generator.")
        task2_desc = st.text_input("Task 2 (Reviewer):", "Review and optimise the code for efficiency.")

        if st.button("Run Coding Crew", key="multi_agent"):
            llm_coder = get_local_llm(temperature=0.2)
            llm_reviewer = get_local_llm(temperature=0.2)

            # Coder agent
            prompt1 = ChatPromptTemplate.from_messages([
                ("system", "You are a skilled programmer."),
                ("human", "{task}")
            ])
            chain1 = prompt1 | llm_coder | StrOutputParser()
            code1 = chain1.invoke({"task": task1_desc})

            # Reviewer agent
            prompt2 = ChatPromptTemplate.from_messages([
                ("system", "You are an expert code reviewer."),
                ("human", f"Review this code and optimise it:\n{code1}")
            ])
            chain2 = prompt2 | llm_reviewer | StrOutputParser()
            code2 = chain2.invoke({"query": code1})

            st.success("Crew Output:")
            st.write("📝 Coder Output:")
            st.code(code1, language="python")

            st.write("🧠 Reviewer Output:")
            st.code(code2, language="python")

    # -----------------------------
    # TAB 3: STATEFUL MEMORY AGENT
    # -----------------------------
    with tab3:
        st.subheader("🧠 Stateful Code Memory")
        if "code_memory" not in st.session_state:
            st.session_state.code_memory = []

        user_input = st.text_input("Your coding question:", "Write a function to fetch JSON data from an API.")

        if st.button("Send Code Request", key="memory_agent"):
            llm = get_local_llm()
            history_text = "\n".join(st.session_state.code_memory)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert coding assistant."),
                ("human", f"History:\n{history_text}\nUser request: {user_input}\nProvide working code:")
            ])
            chain = prompt | llm | StrOutputParser()
            code_response = chain.invoke({"query": user_input})

            st.session_state.code_memory.append(f"User: {user_input}")
            st.session_state.code_memory.append(f"Code: {code_response}")

            st.write("🗣️ Code Assistant Response:")
            st.code(code_response, language="python")
            st.caption(f"Memory turns: {len(st.session_state.code_memory)//2}")

    st.markdown("---")
    st.caption("Built locally using Ollama and LangChain 0.3.x — Copilot-style code generation.")

# -----------------------------
# 🔹 ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
