import os
from io import StringIO
from dotenv import load_dotenv

import streamlit as st
# from langchain_cohere import CohereEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage 

from KnowledgeBase import KnowledgeBase
from HistoryAdding import HistoryAdder
from Retrievial import Retriever
from Generation import Generator

load_dotenv()

# llm = ChatCohere()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="AIzaSyBYNcmgcf3vOTBpsU7IOvGFofzK5hDkV4A")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
ha = HistoryAdder(llm)
kb = KnowledgeBase(embeddings, "D:\\EngRAG\\LiteratureIsEasy\\knowledge_base")
db = kb.modify_vector_store()
r = Retriever(db, "similarity", {"k": 2})

question_generator_prompt = """
    You are an AI-powered TOEIC question generator designed to create high-quality TOEIC-style questions focus on Writing task based on a given knowledge base. Your goal is to generate questions that test students' English proficiency in alignment with TOEIC standards.

    Instructions:
        Input Knowledge Base: You will receive a knowledge base that contains grammar rules, vocabulary lists, question examples, format of the exam. Use this information as the foundation for generating questions.
        Question Types: Generate questions that reflect real TOEIC exam formats style based on given knowledge base.
        Question Format: You MUST point out which part of the exam the question belongs to and direction of how to answer the questions.
        Difficulty Level: Adjust the difficulty of the questions based on the knowledge base provided. Label each question as Beginner, Intermediate, or Advanced.
    
    Here is the knowledge base: 
        {kb}
"""

question_agent = Generator(llm, question_generator_prompt)
prompt = "Generate a TOEIC writing task part 1 (or part 6 totally) question"
# query_aware_history = ha.get_hist_context(st.session_state.hist_for_context, prompt)
rel_docs = r.get_context(prompt)

res = question_agent.generate_answer(prompt, rel_docs)

grader_prompt = """
    You are a TOEIC teacher who specializes in grading students' answers to writing questions. Your objective is to evaluate a student's written response using a set of clearly defined criteria. For every essay you grade, please follow these steps:
        1. Task Achievement / Task Response:
            _ Check if the response fully addresses the prompt and covers all required parts.
            _ Assess whether the content is relevant, well-developed, and supported with details or examples.

        2. Coherence and Cohesion:
            _ Evaluate the overall organization and structure of the response.
            _ Look for clear paragraphing, logical sequencing of ideas, and the effective use of linking devices (e.g., transition words, referencing).

        3. Lexical Resource:
            _ Assess the range and accuracy of vocabulary.
            _ Check for appropriate word choice, collocations, and correct spelling.

        4. Grammatical Range and Accuracy:
            _ Evaluate the variety and complexity of sentence structures (including simple, compound, and complex sentences).
            _ Ensure grammar, punctuation, and syntax are used correctly with minimal errors.

        5. Task-Specific Considerations:
            _ Verify that the response follows the required format, tone, and style (e.g., essay, report, letter) as specified by the question.

        For each criterion, provide a qualitative score (or descriptive feedback, out of 10 score) and specific comments on strengths and areas for improvement. Then, calculate an overall score by averaging the scores from the four main criteria (Task Achievement/Response, Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy).
        You MUST not afraid to give a low grade.
        Just give your evaluation and comment, dont add anythings else.
        In your final output, include:
            _ A brief overall summary of the response's performance.
            _ Detailed feedback for each criterion.
            _ A final overall score along with recommendations for improvement.
            _ Make sure your evaluation is objective, evidence-based, and strictly focused on the content and language of the essay.
"""

grade_agent = Generator(llm, grader_prompt)

with open("question.txt", "w") as f:
    f.write(res)

answer = input("Put your answer here: ")

with open("grade.txt", "w") as f:
    f.write(grade_agent.generate_answer(query=f"Here is the question: {res}.\nHere is student's answer: {answer}", mode="normal"))
# st.title("Let's learn literature together !!!")

# # BUG:
# # Token limit exceed
# # Prompt in is None

# if "hist" not in st.session_state:
#     st.session_state.hist = []

# if "disable_input" not in st.session_state:
#     st.session_state.disable_input = False

# if "processed_file" not in st.session_state:
#     st.session_state.processed_file = []

# if "db" not in st.session_state:
#     st.session_state.db = None

# if "hist_for_context" not in st.session_state:
#     st.session_state.hist_for_context = []

# def main():
#     with st.sidebar:
#         uploaded_file = st.file_uploader("Choose a file", type=["txt"])

#         # if uploaded_file[i].name in st.session_state.processed_file:
#         #     continue
#         if uploaded_file:
#             with st.spinner('Processing document...'):
#                 file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

#                 dm = DataManager(file_content, embeddings, uploaded_file.name)
#                 st.session_state.db = dm.modify_vector_store("create")

#                 st.session_state.processed_file.append(uploaded_file.name)

#     if st.session_state.db is None:
#         st.info("Upload at least 1 file please")
#     else:
#         r = Retriever(st.session_state.db, "similarity", {"k": 2})

#         for chat in st.session_state.hist:
#                 with st.chat_message(chat["role"]):
#                     st.markdown(chat["content"])  

#         prompt = st.chat_input("Nhập câu hỏi của bạn tại đây")

#         if prompt is not None and prompt.strip != "":
#             st.session_state.hist.append(
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             )

#             with st.chat_message("user"):
#                 st.markdown(prompt)

#             with st.chat_message("assistant"):    
#                 st.session_state.disable_input = True
#                 holder = st.empty()
#                 holder.markdown("Let me think...")
                
#                 query_aware_history = ha.get_hist_context(st.session_state.hist_for_context, prompt)
#                 rel_docs = r.get_context(query_aware_history)

#                 res = g.generate_answer(rel_docs, query_aware_history)

#                 holder.markdown(res)
#                 st.session_state.disable_input = False

#             st.session_state.hist.append(
#                 {
#                     "role": "assistant",
#                     "content": res
#                 }
#             )

#             st.session_state.hist_for_context.extend(
#             [
#                 HumanMessage(content=prompt),
#                 AIMessage(content=res),
#             ]
#             )


# if __name__ == "__main__":
#     main()

