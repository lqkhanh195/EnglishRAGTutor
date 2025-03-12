class Retriever:
    def __init__(self, vector_store, search_type, search_kwargs):
        self.retriever = vector_store.as_retriever(
                                                search_type=search_type,
                                                search_kwargs=search_kwargs
                                                )
        
    def get_context(self, query):
        return self.retriever.invoke(query)