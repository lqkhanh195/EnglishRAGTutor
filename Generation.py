from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

class Generator:
    def __init__(self, llm, system_prompt):
        self.llm = llm
        self.system_prompt = SystemMessagePromptTemplate.from_template(system_prompt)

    def parse_output(self, output):
        return output.content if hasattr(output, "content") else output

    def generate_answer(self, query, rel_docs=None, mode="rag"):
        # Merge context from retrieved documents
        if mode == "rag":
            print(mode)
            context = "\n\n".join(doc.page_content for doc in rel_docs)

        query = HumanMessagePromptTemplate.from_template(query)

        chat_prompt = ChatPromptTemplate.from_messages([self.system_prompt, query])

        # Define prompt template
        if mode == "rag":
            messages = chat_prompt.format_messages(kb=context)
        else:
            messages = chat_prompt.format_messages()

        response = self.llm(messages)

        # Run the chain
        return self.parse_output(response)
