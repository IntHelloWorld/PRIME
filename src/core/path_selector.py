from typing import List, Tuple

import numpy as np
import requests


class PathSelector:
    def __init__(self, org, model, api_key, base_url):
        self.org = org
        assert org in [
            "jinaai",
            "vllm",
        ], "Unsupported organization. Use 'jinaai' or 'vllm'."
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings using Jina API.
        """
        data = {
            "model": self.model,
            "task": "text-matching",
            "input": [{"text": text} for text in texts],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            # Extract embeddings from response
            embeddings = []
            for item in result.get("data", []):
                embeddings.append(item.get("embedding", []))
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error embedding texts: {e}") from e

    def _embed_texts_vllm(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings using local deployed VLLM model.
        """
        data = {
            "model": self.model,
            "input": texts,
        }

        try:
            response = requests.post(self.base_url, json=data, timeout=20)
            response.raise_for_status()
            result = response.json()

            # Extract embeddings from response
            embeddings = []
            for item in result.get("data", []):
                embeddings.append(item.get("embedding", []))
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error embedding texts: {e}") from e

    def _cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Handle zero vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(v1, v2) / (norm1 * norm2)

    def _calculate_min_similarity(
        self,
        embedding: List[float],
        existing_embeddings: List[List[float]],
    ) -> float:
        """
        Calculate minimum cosine similarity between an embedding and all existing embeddings.
        """
        if not existing_embeddings:
            return 0.0  # No existing embeddings to compare against

        similarities = []
        for existing_embedding in existing_embeddings:
            similarity = self._cosine_similarity(embedding, existing_embedding)
            similarities.append(similarity)

        return min(similarities)

    def embed_paths(
        self,
        paths: List[str],
    ) -> List[Tuple[str, List[float]]]:
        """
        Embed a list of paths and return them with their embeddings.
        """
        if not paths:
            raise ValueError("Input paths list cannot be empty.")

        # Embed all input paths
        if self.org == "jinaai":
            path_embeddings = self._embed_texts(paths)
        else:
            path_embeddings = self._embed_texts_vllm(paths)

        # Return paths with their embeddings
        return list(zip(paths, path_embeddings))

    def select_paths(
        self,
        paths: List[str],
        existing_paths: List[Tuple[str, List[float]]],
        top_n: int = 2,
    ) -> Tuple[List[int], List[List[float]]]:
        """
        Select the top N most distinct paths from the input list.

        Args:
            paths: List of path strings to select from
            existing_paths: List of tuples (path_string, embedding) for existing paths
            top_n: Number of most distinct paths to select

        Returns:
            selected_indices: List of indices of selected paths in original list
            selected_embeddings: List of embeddings for selected paths
        """
        if not paths:
            raise ValueError("Input paths list cannot be empty.")

        # Embed all input paths
        if self.org == "jinaai":
            path_embeddings = self._embed_texts(paths)
        else:
            path_embeddings = self._embed_texts_vllm(paths)

        # Calculate distinctness scores (minimum similarity with existing paths)
        distinctness_scores = []
        existing_embeddings = [embedding for _, embedding in existing_paths]
        for embedding in path_embeddings:
            min_similarity = self._calculate_min_similarity(
                embedding, existing_embeddings
            )
            # Lower similarity means more distinct, so we negate for ranking
            distinctness_scores.append(min_similarity)

        # Get indices sorted by distinctness (most distinct first)
        sorted_indices = sorted(
            range(len(paths)),
            key=lambda i: distinctness_scores[i],
        )

        # Select top N most distinct paths
        selected_indices = sorted_indices[: min(top_n, len(paths))]
        selected_embeddings = [path_embeddings[i] for i in selected_indices]

        return selected_indices, selected_embeddings
