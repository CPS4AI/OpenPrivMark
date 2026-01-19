import argparse
import json
import jax
import jax.numpy as jnp
import numpy as np
import random
import pickle
import time 

from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, FlaxAutoModel
from transformers.utils import logging

class LLM:
    """
    Handles text generation using a JAX-based transformer model.
    """
    def __init__(self, llm_model, llm_tokenizer):
        """
        Initializes the model and tokenizer.
        llm_model: A string name of the pretrained model or a FlaxGPT2LMHeadModel instance.
        llm_tokenizer: A string name of the pretrained tokenizer or an AutoTokenizer instance.
        """
        if isinstance(llm_model, str):
            self.llm_model = FlaxGPT2LMHeadModel.from_pretrained(llm_model)
        else:
            self.llm_model = llm_model

        if isinstance(llm_tokenizer, str):
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer)
        else:
            self.llm_tokenizer = llm_tokenizer

    def text_generation(self, input_ids, params, token_num=50):
        """
        Generates text by sampling from the model.
        """
        for _ in range(token_num):
            outputs = self.llm_model(input_ids=input_ids, params=params)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = jnp.argmax(next_token_logits)
            input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
        return input_ids

    def get_input_ids(self, prompt):
        """
        Tokenizes the prompt into JAX input_ids.
        """
        return self.llm_tokenizer.encode(prompt, return_tensors='jax')

    def get_params(self):
        """
        Returns model parameters.
        """
        return self.llm_model.params

class Embedder:
    """
    Handles text tokenization and embedding using a JAX-based transformer model.
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = FlaxAutoModel.from_pretrained(model_name)
        self.params = self.model.params

    def tokenize(self, text: str | list[str]):
        """
        Tokenizes the input text or list of texts.
        """
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="jax"
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def mean_pool(self, hidden: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        Performs mean pooling on hidden states using the attention mask.
        """
        mask = mask[..., None]
        summed = (hidden * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        return summed / counts

    def l2_norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        L2 normalizes the input tensor.
        """
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-9)
        return x / norm

    def embed(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray, params=None) -> jnp.ndarray:
        """
        Embeds input_ids using the model.
        """
        if params is None:
            params = self.params
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, params=params, train=False)
        hidden = outputs.last_hidden_state
        pooled = self.mean_pool(hidden, attention_mask)
        return self.l2_norm(pooled)

class Watermarking:
    """
    Performs watermarking operations locally using JAX.
    """
    def __init__(self, r: float):
        self.r = r

    def count_words(self, input_text: str) -> int:
        """
        Counts the number of words in the input text locally.
        """
        return len([w.strip(".,!?;/\"#@&'([)`]{\\n") for w in input_text.split() if w])

    def get_number_of_watermark_words(self, n: int) -> int:
        """
        Calculates the number of watermark words to insert.
        """
        return int(n * self.r)

    def get_k_bis(self, k: int) -> int:
        """
        Calculates the number of watermark words to select from the vocabulary.
        """
        return 3 * k

    def create_embedding_table(self, vocab_path: str, doc_emb_path: str):
        """
        Creates an embedding table from vocabulary and document embeddings.
        """
        with open(vocab_path, "rb") as f:
            words = pickle.load(f)
        with open(doc_emb_path, "rb") as f:
            redpj_embs = pickle.load(f)

        assert len(redpj_embs) >= len(words)

        indices = random.sample(range(len(redpj_embs)), len(words))
        emb_list = [redpj_embs[i] for i in indices]
        random.shuffle(emb_list)
        embedding_table_np = np.stack(emb_list).astype(np.float32)

        return jnp.array(embedding_table_np), words

    @staticmethod
    def cosine_sim(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Computes cosine similarity between a vector and a matrix of vectors.
        """
        a_norm = a / jnp.linalg.norm(a)
        b_norm = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
        return jnp.dot(b_norm, a_norm)

    @staticmethod
    def top_k(sims: jnp.ndarray, k: int):
        """
        Computes top-k values and indices locally using JAX.
        """
        return jax.lax.top_k(sims, k)

    def save_watermark_output(self, inserted_text: str, filtered_words: list[str], output_path="watermark_output.json"):
        """
        Saves the inserted text and filtered words to a JSON file.
        """
        payload = {
            "inserted_text": inserted_text,
            "filtered_words": filtered_words
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)



if __name__ == '__main__':
    logging.set_verbosity_error()  # Suppress transformers warnings
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rate', type=float, default=0.12, help="Watermarking rate")
    args = parser.parse_args()

    print("--- Running Local Watermarking Test with JAX ---")
    original_text = 'We are not! Like men in the story of the Good Samaritan, they pass by on the other side...'
    print(f"\nOriginal text: \"{original_text}\"")

    r = args.rate
    print(f"Watermarking rate:\t\t r = {r}")

    watermarker = Watermarking(r)

    n = watermarker.count_words(original_text)
    print(f"Word count:\t\t\t n = {n}")

    k = watermarker.get_number_of_watermark_words(n)
    print(f"Watermark words to insert:\t k = {k}")

    k_bis = watermarker.get_k_bis(k)
    print(f"Watermark words to select:\t k_bis = {k_bis}")

    path = "postmark_utils/" 
    sectable, idx2word = watermarker.create_embedding_table(
        f"{path}valid_wtmk_words_in_wiki_base-only-f1000.pkl",
        f"{path}filtered_data_100k_unique_250w_sentbound_e5_base_wikitext_embs.pkl"
    )

    embedder = Embedder("intfloat/e5-base")
    text_ids, text_mask = embedder.tokenize(original_text)
    text_emb = embedder.embed(text_ids, text_mask)[0]


    # start_time_1 = time.time()
    sims = watermarker.cosine_sim(text_emb, sectable)
    sims.block_until_ready() 
    # end_time_1 = time.time()
    # time_1 = end_time_1 - start_time_1

    vals, idxs = watermarker.top_k(sims, k_bis)
    words = [idx2word[i] for i in np.array(idxs)]
    print(f"\nTop {k_bis} potential watermark words: {words}")

    words_ids, words_mask = embedder.tokenize(words)
    words_emb = embedder.embed(words_ids, words_mask)

    # start_time_2 = time.time()
    sims_filtered = watermarker.cosine_sim(text_emb, words_emb)
    sims_filtered.block_until_ready() 
    # end_time_2 = time.time()
    # time_2 = end_time_2 - start_time_2

    

    # average_time = (time_1 + time_2) / 2
    # print(f"Average cosine similarity computation time: {average_time:.6f} seconds")

    filtered_vals, filtered_idxs = watermarker.top_k(sims_filtered, k)

    filtered_words = [words[i] for i in np.array(filtered_idxs)]
    print(f"\nFiltered top {k} watermark words: {filtered_words}")

    inserter = LLM(llm_model="gpt2", llm_tokenizer="gpt2")
    prompt = f"Insert the following words: [{', '.join(filtered_words)}] into the text: \"{original_text}\""
    
    input_ids = inserter.get_input_ids(prompt)
    params = inserter.get_params()
    output_ids = inserter.text_generation(input_ids, params, token_num=50)
    
    inserted_text = inserter.llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nText with inserted watermark:\n \"{inserted_text}\"")

    watermarker.save_watermark_output(
        inserted_text=inserted_text,
        filtered_words=filtered_words,
        output_path="watermark_output.json"
    )
    print("\nSaved output to watermark_output.json")