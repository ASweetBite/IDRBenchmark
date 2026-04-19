import json
import logging
import random
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

_FAISS_AVAILABLE = False
_FT_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    logger.warning("faiss not installed; falling back to random substitution.")

try:
    import fasttext
    _FT_AVAILABLE = True
except ImportError:
    logger.warning("fasttext not installed; falling back to random substitution.")

from utils.ast_tools import IdentifierAnalyzer


class StaticCandidateGenerator:
    def __init__(self, config):
        """Initializes the generator by loading FastText and FAISS models for semantic search."""
        self.config = config.get("lightweight_candidate", {})
        self.top_m = self.config.get("top_m", 10)

        self.ft_model = None
        self.faiss_index = None
        self.vocab = []
        self.analyzer = IdentifierAnalyzer()

        ft_path = self.config.get("fasttext_model_path", "")
        if _FT_AVAILABLE and ft_path:
            try:
                self.ft_model = fasttext.load_model(ft_path)
                logger.info(f"FastText model loaded successfully: {ft_path}")
            except Exception as e:
                logger.warning(f"Failed to load FastText: {e}")

        faiss_path = self.config.get("faiss_index_path", "")
        vocab_path = self.config.get("faiss_vocab_path", "")
        if _FAISS_AVAILABLE and faiss_path:
            try:
                self.faiss_index = faiss.read_index(faiss_path)
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self.vocab = json.load(f)
                logger.info(f"FAISS index loaded successfully: {faiss_path}, vocab size: {len(self.vocab)}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS: {e}")

        self._fallback_pool = [
            f"var_{i}" for i in range(200)
        ] + [
            f"tmp_{i}" for i in range(50)
        ] + [
            f"arg_{i}" for i in range(50)
        ]

    def _get_faiss_neighbors(self, word: str) -> List[str]:
        """Retrieves semantically similar candidates from the FAISS index."""
        if self.ft_model is None or self.faiss_index is None:
            return []

        vec = self.ft_model.get_word_vector(word).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        _, indices = self.faiss_index.search(vec, self.top_m + 1)

        candidates = []
        for idx in indices[0]:
            if 0 <= idx < len(self.vocab):
                cand = self.vocab[idx]
                if cand != word:
                    candidates.append(cand)
        return candidates[: self.top_m]

    def _fallback_candidates(self, word: str) -> List[str]:
        """Provides a random set of fallback identifiers when semantic search fails."""
        pool = [w for w in self._fallback_pool if w != word]
        return random.sample(pool, min(self.top_m, len(pool)))

    def generate_candidates(self, code: str) -> Dict[str, List[str]]:
        """Maps each identifier in the provided code to a list of potential replacement candidates."""
        code_bytes = code.encode("utf-8") if isinstance(code, str) else code
        identifiers = self.analyzer.extract_identifiers(code_bytes)

        result = {}
        for ident in identifiers:
            neighbors = self._get_faiss_neighbors(ident)
            if not neighbors:
                neighbors = self._fallback_candidates(ident)
            result[ident] = neighbors

        return result

    def get_random_replacement(self, code: str, target_vars: List[str]) -> Dict[str, str]:
        """Selects a single random substitution for each specified target variable."""
        candidates = self.generate_candidates(code)
        mapping = {}
        for var in target_vars:
            cands = candidates.get(var, self._fallback_candidates(var))
            if cands:
                mapping[var] = random.choice(cands)
        return mapping