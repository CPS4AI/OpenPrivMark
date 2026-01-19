# bazelisk run -c opt //utils:nodectl -- --config `conf/3pc.json up
# bazelisk run -c opt //privmark -- --config conf/3pc.json


import argparse
import json
import jax
import jax.numpy as jnp
import numpy as np
import random
import pickle
import secrets

from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, FlaxAutoModel, GPT2Config
from transformers.utils import logging

import spu.utils.distributed as ppd
import spu.intrinsic as intrinsic
import spu.libspu as libspu

copts = libspu.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = libspu.XLAPrettyPrintKind(2)
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True


class LLM:
    def __init__(self, llm_model, llm_tokenizer):
        """
        llm_model: either a string name of the pretrained model or a FlaxGPT2LMHeadModel instance
        llm_tokenizer: either a string name of the pretrained tokenizer or an AutoTokenizer instance
        """
        # Load or assign model
        if isinstance(llm_model, str):
            self.llm_model = FlaxGPT2LMHeadModel.from_pretrained(llm_model)
        else:
            self.llm_model = llm_model
        # Load or assign tokenizer
        if isinstance(llm_tokenizer, str):
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer)
        else:
            self.llm_tokenizer = llm_tokenizer
    
    def text_generation(self, input_ids, params, token_num=50)->jnp.ndarray:
        """
        Generate text by sampling from the model.
        input_ids: JAX tensor of shape [batch_size, seq_length]
        params: model parameters for FlaxGPT2LMHeadModel
        token_num: number of tokens to generate
        Returns: JAX tensor of shape [batch_size, seq_length + token_num]
        """
        model = self.llm_model

        for _ in range(token_num):
            outputs = model(input_ids=input_ids, params=params)
            next_token_logits = outputs[0][0, -1, :]
            next_token = jnp.argmax(next_token_logits)
            input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
        return input_ids

    def get_input_ids(self, prompt)->jnp.ndarray:
        """
        Tokenize the prompt into JAX input_ids.
        """
        return self.llm_tokenizer.encode(
            prompt,
            return_tensors='jax'
        )

    def get_params(self)->dict:
        """
        Return model parameters for SPU placement.
        """
        return self.llm_model.params
    
    
class Embedder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = FlaxAutoModel.from_pretrained(model_name)
        self.params = self.model.params

    def tokenize(self, text: str | list[str]):
        """
        Tokenize the input text or list of texts.
        Returns input_ids and attention_mask as JAX tensors.
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
        Mean pool the hidden states using the attention mask.
        """
        mask = mask[..., None]
        summed = (hidden * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1)
        return summed / counts

    def l2_norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        L2 normalize the input tensor along the last dimension.
        """
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-9)
        return x / norm

    def embed(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray, params=None) -> jnp.ndarray:
        """
        Embed input_ids with attention_mask using the model parameters.
        If params is None, use the instance's parameters.
        """
        if params is None:
            params = self.params
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, params=params, train=False)
        hidden = outputs.last_hidden_state
        pooled = self.mean_pool(hidden, attention_mask)
        return self.l2_norm(pooled)


class SPU_Watermarking:
    def __init__(self, conf, r):
        self.conf = conf
        self.r = r
        
        ppd.init(conf['nodes'], conf['devices'])
    
    def __init__(self, conf, r: float, embeddder: Embedder, inserter: LLM):
        self.conf = conf
        self.r = r
        self.embedder = embeddder
        self.inserter = inserter
        
        ppd.init(conf['nodes'], conf['devices'])
    

    
    
    def create_sectable_on_spu(self, vocab_path: str, doc_emb_path: str)->tuple[jnp.ndarray, dict, list]:
        """
        Creates a secure embedding table on the SPU.
        The numerical table is created on the SPU, while the Python-specific
        word-to-index mappings are created on the host.
        """
        def _load_pickle(path: str):
            with open(path, "rb") as f:
                return pickle.load(f)

        words_ref = _load_pickle(vocab_path)
        word2idx = {w: i for i, w in enumerate(words_ref)}
        idx2word = list(words_ref)

        embs_ref = ppd.device("P2")(_load_pickle)(doc_emb_path)

        def _create_embedding_table_on_spu(words_len, embs):
            emb = jnp.array(embs, dtype=jnp.float32)
            N = emb.shape[0]
            M = words_len

            seed = secrets.randbits(32) 
            key = jax.random.PRNGKey(seed)
            perm = jax.random.permutation(key, N)
            chosen_indices = perm[:M]
            embedding_table = emb[chosen_indices]

            return embedding_table
        
        spu_embedding_table = ppd.device("SPU")(
            _create_embedding_table_on_spu, copts=copts, static_argnums=(0,)
        )(len(words_ref), embs_ref)
        
        return spu_embedding_table, word2idx, idx2word
    

    @staticmethod
    def cosine_sim(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Compute cosine similarity between a (vector) and each row of b (matrix).
        Returns a vector of similarities of shape [vocab_size].
        """
        a_norm = a / jnp.linalg.norm(a)
        b_norm = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
        return jnp.dot(b_norm, a_norm)



    
    def save_watermark_output(self, inserted_text: str, filtered_words: list[str], output_path="watermark_output.json"):
        """
        Save the inserted text and filtered words to a JSON file.
        """
        payload = {
            "inserted_text": inserted_text,
            "filtered_words": filtered_words
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    

    
    
    def privmark_insertion(self, text: str, sectable: jnp.ndarray, word2idx: dict, idx2word: list)->str:
        """
        Inserts a watermark into the text using secure computation with SPU.
        Word selection is performed entirely within the SPU to preserve privacy.
        """
        spu_table = ppd.device("SPU")(lambda x: x)(sectable)
        vocab_size = sectable.shape[0]

        n_plain = len([w.strip(".,!?;/\"#@&'([)]{\}\n") for w in text.split() if w])
        k_plain = int(np.floor(n_plain * self.r))
        k_bis_plain = int(np.floor(3 * k_plain))

        texts = [text]
        input_ids, attention_mask = self.embedder.tokenize(texts)

        spu_ids = ppd.device("P2")(lambda x: x)(input_ids)
        spu_mask = ppd.device("P2")(lambda x: x)(attention_mask)
        spu_params = ppd.device("P2")(lambda x: x)(self.embedder.params)

        def _forward(ids, mask, params):
            return self.embedder.embed(ids, mask, params)

        text_emb = ppd.device("SPU")(_forward, copts=copts)(spu_ids, spu_mask, spu_params)
        
        sims = ppd.device("SPU")(lambda emb, table: self.cosine_sim(emb.squeeze(), table), copts=copts)(text_emb, spu_table)

        def _top_kbis_fn(x):
            return jax.lax.top_k(x, k=k_bis_plain)

        _, spu_idxs_kbis = ppd.device("SPU")(_top_kbis_fn, copts=copts)(sims)

        def _select_embeddings_fn(indices, full_embedding_table):
            one_hot_matrix = jax.nn.one_hot(indices, num_classes=vocab_size)
            selected_embeddings = jnp.matmul(one_hot_matrix, full_embedding_table)
            return selected_embeddings

        spu_words_emb = ppd.device("SPU")(_select_embeddings_fn, copts=copts)(spu_idxs_kbis, spu_table)

        sims_filtered = ppd.device("SPU")(lambda emb, table: self.cosine_sim(emb.squeeze(), table), copts=copts)(text_emb, spu_words_emb)
        
        def _top_k_fn(x):
            return jax.lax.top_k(x, k=k_plain)

        _, spu_filtered_relative_idxs = ppd.device("SPU")(_top_k_fn, copts=copts)(sims_filtered)

        def _gather_indices_fn(original_indices, relative_indices):
            final_absolute_indices = original_indices.take(relative_indices)
            return final_absolute_indices

        spu_final_idxs = ppd.device("SPU")(_gather_indices_fn, copts=copts)(spu_idxs_kbis, spu_filtered_relative_idxs)

        final_idxs = ppd.get(spu_final_idxs)
        
        filtered_words = [idx2word[i] for i in final_idxs]
        print(f"Filtered top {k_plain} watermark words (securely selected):", filtered_words)
        
        prompt = f"Insert the following words: [{', '.join(filtered_words)}] into the text: \"{original_text}\""
        
        spu_input = ppd.device('P1')(lambda x: x)(inserter.get_input_ids(prompt))
        spu_params = ppd.device('P1')(lambda x: x)(inserter.get_params())
        spu_out = ppd.device("SPU")(inserter.text_generation, copts=copts)(spu_input, spu_params)
        
        output_ids = ppd.get(spu_out)
        
        inserted_text = inserter.llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Inserted text on SPU:", inserted_text)
        
        return inserted_text
    
    def privmark_detection(self,
                candidate_words: list[str],
                watermark_words: list[str],
                embedder: Embedder,
                theta_sim: float = 0.85,  
                theta_det: float = 0.45
                ) -> bool:
        """
        Detect watermark presence in candidate words using SPU.
        :param candidate_words: List of words from the candidate text.
        :param watermark_words: List of words from the watermark.
        :param embedder: Embedder instance for generating embeddings.
        :param theta_sim: Similarity threshold for individual watermark word detection.
        :param theta_det: Detection threshold (fraction of watermark words present).
        :return: (detected: bool, present_count: int, fraction: float)
        """
        cand_embs = self.embedder_on_spu(embedder, candidate_words)
        wm_embs   = self.embedder_on_spu(embedder, watermark_words)
        theta_sim_spu = ppd.device("SPU")(lambda x: x)(theta_sim)

        def _check_presence(similarities, threshold):
            return (similarities >= threshold).any()

        present_count = ppd.device("SPU")(lambda x: 0)(0) 
        for word, wm_emb in zip(watermark_words, wm_embs):
            sims = ppd.device("SPU")(lambda emb, table: self.cosine_sim(emb.squeeze(), table), copts=copts)(wm_emb, cand_embs)
            
            is_present_spu = ppd.device("SPU")(_check_presence, copts=copts)(sims, theta_sim_spu)
            
            if is_present_spu:
                present_count = ppd.device("SPU")(lambda x: x + 1)(present_count)
        
        spu_present = ppd.device("SPU")(lambda x: x)(present_count)
        spu_total   = ppd.device("SPU")(lambda x: x)(len(watermark_words))

        def _compute_fraction(present, total):
            present = jnp.asarray(present, dtype=jnp.float32)
            total   = jnp.asarray(total,   dtype=jnp.float32)
            return jnp.where(total > 0, present / total, 0.0)

        p = ppd.device("SPU")(_compute_fraction, copts=copts)(spu_present, spu_total)
        
        theta_det_spu = ppd.device("SPU")(lambda x: x)(theta_det)
        
        detected_spu = ppd.device("SPU")(lambda frac, thres: frac > thres)(p, theta_det_spu)

        return ppd.get(detected_spu)

    


if __name__ == '__main__':
    logging.set_verbosity_error()  # Suppress warnings from transformers
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='conf3pc.json')
    parser.add_argument('-r', '--rate', type=float, default=0.12)
    parser.add_argument('-voc', '--vocab_path', type=str, default='utils/valid_wtmk_words_in_wiki_base-only-f1000.pkl')
    parser.add_argument('-emb', '--emb_path', type=str, default='utils/filtered_data_100k_unique_250w_sentbound_e5_base_wikitext_embs.pkl')
    parser.add_argument("--threshold_sim", type=float, default=0.85, help="similarity threshold")
    parser.add_argument("--threshold_det", type=float, default=0.3, help="detection threshold (fraction)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        conf = json.load(f)
        
        
    
    # example usage
    print("------------------\nRunning SPU Watermarking Test\n------------------")
    original_text = 'We are not! Like men in the story of the Good Samaritan, they pass by on the other side...'
                     
    print("Original text:", original_text)
    
    r = args.rate
    print(f"Watermarking rate:\t\t\t r = {r}")
    
    
    inserter = LLM(
        llm_model="gpt2",
        llm_tokenizer="gpt2"
    )
    embedder = Embedder("intfloat/e5-base")
    
    spu_watermarking = SPU_Watermarking(conf, r, embedder, inserter)
    
    

    sectable, word2idx, idx2word = spu_watermarking.create_sectable_on_spu(
        args.vocab_path,
        args.emb_path
    )
    
    print("Sectable shape:", sectable.shape)
    spu_watermarking.privmark_insertion(original_text, sectable, word2idx, idx2word)
    
