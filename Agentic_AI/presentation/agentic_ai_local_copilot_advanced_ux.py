import streamlit as st

# -----------------------------
# LangChain 0.3 IMPORTS
# -----------------------------
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# MODEL INITIALISATION
# -----------------------------
def get_local_llm(model_name="llama3", temperature=0.2):
    return ChatOllama(model=model_name, temperature=temperature)


# -----------------------------
# MAIN APP FUNCTION
# -----------------------------
def main():
    st.set_page_config(page_title="Copilot AI (Local)", page_icon="💻", layout="wide")
    st.title("Copilot AI (Local Edition)")
    st.write("Generate code locally using Ollama models — no API key required.")

    # Sidebar
    st.sidebar.title("Model Configuration")
    model_name = st.sidebar.selectbox(
        "Choose Ollama model:",
        ["llama3", "mistral", "gemma2"],
        index=0
    )
    temperature = st.sidebar.slider("Temperature (creativity):", 0.0, 1.0, 0.2, 0.05)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "💻 Single Code Agent",
        "👥 Multi-Agent Crew",
        "🧠 Stateful Memory"
    ])

    # ---------------------------------------------------------
    # TAB 1: SINGLE CODE AGENT
    # ---------------------------------------------------------
    with tab1:
        st.subheader("💻 Single Code Agent")

        code_query = st.text_area(
            "Describe the code you want to generate:",
            "Write a Python function to sort a list of dictionaries by key 'age'."
        )

        if st.button("Generate Code", key="single_agent"):
            llm = get_local_llm(model_name, temperature)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert coding assistant."),
                ("human", "{query}\nProvide a fully working code snippet.")
            ])

            chain = prompt | llm | StrOutputParser()

            code_output = chain.invoke({"query": code_query})

            st.success("Generated Code:")
            st.code(code_output, language="python")

    # ---------------------------------------------------------
    # TAB 2: MULTI-AGENT CREW
    # ---------------------------------------------------------
    with tab2:
        st.subheader("👥 Multi-Agent Crew Simulation")

        task1_desc = st.text_input("Task 1 (Coder):", "Write Python code for a Fibonacci generator.")
        task2_desc = st.text_input("Task 2 (Reviewer):", "Review and optimise the code for efficiency.")

        if st.button("Run Coding Crew", key="multi_agent"):
            llm_coder = get_local_llm(model_name, temperature)
            llm_reviewer = get_local_llm(model_name, temperature)

            # Coder agent (Chain 1)
            coder_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a skilled programmer."),
                ("human", "{task}")
            ])
            coder_chain = coder_prompt | llm_coder | StrOutputParser()

            coder_output = coder_chain.invoke({"task": task1_desc})

            # Reviewer agent (Chain 2)
            reviewer_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert code reviewer."),
                ("human", "Review this code and optimise it:\n\n{code}")
            ])
            reviewer_chain = reviewer_prompt | llm_reviewer | StrOutputParser()

            reviewer_output = reviewer_chain.invoke({"code": coder_output})

            st.success("Crew Output:")
            st.write("📝 Coder Output:")
            st.code(coder_output, language="python")

            st.write("🧠 Reviewer Output:")
            st.code(reviewer_output, language="python")

    # ---------------------------------------------------------
    # TAB 3: STATEFUL MEMORY AGENT
    # ---------------------------------------------------------
    with tab3:
        st.subheader("🧠 Stateful Code Memory")

        if "code_memory" not in st.session_state:
            st.session_state.code_memory = []

        user_input = st.text_input(
            "Your coding question:",
            "Write a function to fetch JSON data from an API."
        )

        if st.button("Send Code Request", key="memory_agent"):
            llm = get_local_llm(model_name, temperature)

            history = "\n".join(st.session_state.code_memory)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert coding assistant."),
                ("human",
                 "Conversation history:\n"
                 "{history}\n\n"
                 "User request:\n"
                 "{query}\n\n"
                 "Provide working code:")
            ])

            chain = prompt | llm | StrOutputParser()

            code_response = chain.invoke({
                "history": history,
                "query": user_input
            })

            # Save conversation memory
            st.session_state.code_memory.append(f"User: {user_input}")
            st.session_state.code_memory.append(f"Code: {code_response}")

            st.write("🗣️ Code Assistant Response:")
            st.code(code_response, language="python")
            st.caption(f"Memory turns: {len(st.session_state.code_memory) // 2}")

        # Download full memory
        if st.session_state.code_memory:
            combined_code = "\n\n".join(
                entry.replace("Code: ", "")
                for entry in st.session_state.code_memory
                if entry.startswith("Code:")
            )

            st.download_button(
                "📋 Download All Memory Code",
                combined_code,
                file_name="all_memory_code.py"
            )

    # Footer
    st.markdown("---")
    st.caption(f"Built with Ollama ({model_name}) + LangChain 0.3.x — Local Copilot AI")


# Entry point
if __name__ == "__main__":
    main()
