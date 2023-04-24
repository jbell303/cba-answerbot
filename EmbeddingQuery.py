# answer.py
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
import streamlit as st  # for building the app

class EmbeddingQuery:
    def __init__(self, query, embeddings_path, embeddings_model="text-embedding-ada-002", gpt_model="gpt-3.5-turbo"):
        self.query = query
        self.embeddings_model = embeddings_model
        self.gpt_model = gpt_model

        # download pre-chunked text and pre-computed embeddings from S3
        self.df = pd.read_csv(
            f"s3://{embeddings_path}",
            storage_options={
                "key": st.secrets["AWS_ACCESS_KEY_ID"],
                "secret": st.secrets["AWS_SECRET_ACCESS_KEY"],
            },
        )

        # convert embeddings from CSV str type back to list type
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)

    # search function
    def strings_ranked_by_relatedness(
        self,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = openai.Embedding.create(
            model=self.embeddings_model,
            input=self.query,
        )
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in self.df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]
    
    # ask function
    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.gpt_model)
        return len(encoding.encode(text))


    def query_message(
        self,
        token_budget: int = 4096 - 500,
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        strings, relatednesses = self.strings_ranked_by_relatedness()
        introduction = 'Use the below definitions from the fedex pilot bargaining agreement to answer the subsequent question. If the answer cannot be found in the contract, write "I could not find an answer." If additional information is needed from the prompter, say what information is needed.'
        question = f"\n\nQuestion: {self.query}"
        message = introduction
        for string in strings:
            next_article = f'\n\nContract article section:\n"""\n{string}\n"""'
            if (
                self.num_tokens(message + next_article + question)
                > token_budget
            ):
                break
            else:
                message += next_article
        return message + question