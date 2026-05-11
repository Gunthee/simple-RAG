import requests

class Rag:
    def __init__(self, retriever, llm):
        
        self.llm = ""
        self.vector_db = ""
        
    def change_llm_model(self, model_name):
        pass

    def llm_generate(self, prompt):
        pass

    def query(self, query):
        pass    

    def answer(self, query):
        query = self.query(query)
        prompt = f"""You are a helpful assistant that answers questions based on the following context:
        {query}
        """ 
        return self.llm_generate(prompt)