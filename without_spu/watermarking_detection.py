import argparse
import numpy as np
import jax.numpy as jnp
import json
import os
import time
import random

from transformers import AutoTokenizer, FlaxAutoModel
from transformers.utils import logging

class Embedder:
    """
    A JAX-based class to handle text tokenization and embedding using a Hugging Face transformer model.
    This class is unchanged from the original.
    """
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
        mask_expanded = mask[..., None]
        summed = (hidden * mask_expanded).sum(axis=1)
        counts = mask_expanded.sum(axis=1).clip(min=1e-9)
        return summed / counts

    def l2_norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        L2 normalize the input tensor along the last dimension.
        """
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-9)
        return x / norm

    def embed(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Embed input_ids with attention_mask using the model's parameters.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, params=self.params, train=False)
        hidden = outputs.last_hidden_state
        pooled = self.mean_pool(hidden, attention_mask)
        return self.l2_norm(pooled)


class WatermarkingDetection:
    """
    A class to detect watermarks in text by comparing word embeddings using JAX locally.
    """
    @staticmethod
    def cosine_sim(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Compute cosine similarity between a vector `a` and each row of a matrix `b`.
        Returns a vector of similarities.
        """
        a_norm = a / jnp.linalg.norm(a)
        b_norm = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
        return jnp.dot(b_norm, a_norm)

    def detect(self,
               candidate_words: list[str],
               watermark_words: list[str],
               embedder: Embedder,
               theta_sim: float = 0.7,
               theta_det: float = 0.3
               ) -> bool:
        """
        Detect watermark presence in candidate words locally using JAX.
        :param candidate_words: List of words from the candidate text.
        :param watermark_words: List of words from the watermark.
        :param embedder: Embedder instance for generating embeddings.
        :param theta_sim: Similarity threshold for individual watermark word detection.
        :param theta_det: Detection threshold (fraction of watermark words present).
        :return: True if watermark is detected, False otherwise.
        """
        if not candidate_words or not watermark_words:
            return False

        cand_ids, cand_mask = embedder.tokenize(candidate_words)
        cand_embs = embedder.embed(cand_ids, cand_mask)
        
        wm_ids, wm_mask = embedder.tokenize(watermark_words)
        wm_embs = embedder.embed(wm_ids, wm_mask)

        present_count = 0
        for i, wm_emb in enumerate(wm_embs):
            similarities = self.cosine_sim(wm_emb, cand_embs)
            
            print(f"Watermark word '{watermark_words[i]}' → max similarity: {jnp.max(similarities):.4f}")

            if jnp.any(similarities >= theta_sim):
                present_count += 1
        
        print(f"\nWatermark words present: {present_count} out of {len(watermark_words)}")
        
        p = present_count / len(watermark_words)
        is_detected = p >= theta_det
        
        print(f"Detection ratio: {p:.4f} >= {theta_det} -> {is_detected}")
        
        return is_detected, present_count, p



        


if __name__ == "__main__":
    logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description='distributed watermark detection without SPU.')
    parser.add_argument('--input', type=str, required=True,
                   help='Path to input JSON file containing examples (list of dicts).')
    parser.add_argument('--out', type=str, default='results.jsonl', help='Path to JSONL results output file.')
    parser.add_argument("--threshold_sim", type=float, default=0.85, help="similarity threshold")
    parser.add_argument("--threshold_det", type=float, default=0.45, help="detection threshold (fraction)")
    args = parser.parse_args()

    

    # --- Example Inputs ---
    print(f"=== Local Watermark Detection with ===")
    # watermark_words = ["dominance", "drove", "surprised", "protective"]
    # print(f"Watermark words: {watermark_words}\n")
    
    # # Example 1: Text with watermark
    # candidate_text_with_wm = (
    #     "Despite the grim situation... The troops used the dense jungle as their protective cover, "
    #     "aiming to hit the Marine’s defenses from the side and rear... The dominance of the Marine defense "
    #     "drove the attackers back, leaving them surprised and demoralized... By daybreak, it had become apparent "
    #     "that the Kuma battalion had been effectively wiped out."
    # )
                     
    # # Example 2: Text after a removing attack (some watermark words removed)
    # removing_attack = (
    #     "Despite the grim situation... The troops used the dense jungle as their protective cover, "
    #     "aiming to hit the Marine’s defenses from the side and rear... By daybreak, it had become apparent "
    #     "that the Kuma battalion had been effectively wiped out."
    # )
                     
    # # Example 3: Text after a paraphrase attack (synonyms used)
    # paraphrase_attack = (
    #     "Despite the dire circumstances... The soldiers utilized the thick jungle as their shield, "
    #     "planning to assault the Marine’s fortifications from the flanks and rear... The Marine defense's "
    #     "superiority forced the attackers to retreat, leaving them astonished and disheartened... By dawn, it was clear "
    #     "that the Kuma battalion had been largely eliminated."
    # )
    
    # # --- Select a text to test ---
    # candidate_text = paraphrase_attack
    
    # # Extract candidate words by splitting on whitespace and cleaning punctuation
    # candidate_words = [w.strip(".,!?;:/\"#@&'([)`]{\\n") for w in candidate_text.split() if w]
    # print(f"Candidate words: {candidate_words}\n")

    # # --- Detect watermark ---
    # print("--- Starting Detection ---")
    # detected = detector.detect(candidate_words,
    #                            watermark_words,
    #                            embedder,
    #                            theta_sim=args.threshold_sim,
    #                            theta_det=args.threshold_det)
                               
    # print(f"\nFinal Result: Watermark detected = {detected}")
    
    
