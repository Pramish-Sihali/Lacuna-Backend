"""
LLM-based extraction service for concepts and claims using Ollama.
"""
import httpx
import logging
import json
from typing import List, Dict, Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Handles concept and claim extraction using LLM via Ollama."""

    def __init__(self):
        """Initialize LLM extractor."""
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_LLM_MODEL
        self.timeout = httpx.Timeout(settings.OLLAMA_TIMEOUT, connect=10.0)

    async def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract key concepts from text.

        Args:
            text: Text to analyze

        Returns:
            List of concept dictionaries with 'name', 'description', 'generality_score'
        """
        prompt = f"""You are a research analyst. Extract key concepts from the following text.
For each concept, provide:
1. Name (concise, 2-5 words)
2. Description (1-2 sentences)
3. Generality score (0-1, where 0 is very specific and 1 is very general/abstract)

Return ONLY a JSON array of objects with keys: name, description, generality_score

Text:
{text[:2000]}  # Limit to first 2000 chars

JSON output:"""

        try:
            response_text = await self._call_llm(prompt)
            concepts = self._parse_json_response(response_text)

            if not isinstance(concepts, list):
                logger.warning("LLM did not return a list, wrapping response")
                concepts = [concepts] if concepts else []

            # Validate and clean concepts
            validated_concepts = []
            for concept in concepts:
                if isinstance(concept, dict) and "name" in concept:
                    validated_concepts.append({
                        "name": concept.get("name", "Unknown"),
                        "description": concept.get("description", ""),
                        "generality_score": float(concept.get("generality_score", 0.5))
                    })

            logger.info(f"Extracted {len(validated_concepts)} concepts")
            return validated_concepts

        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []

    async def extract_claims(
        self,
        text: str,
        concept_name: str
    ) -> List[Dict[str, Any]]:
        """
        Extract claims related to a specific concept.

        Args:
            text: Text to analyze
            concept_name: Concept to find claims for

        Returns:
            List of claim dictionaries with 'claim_text', 'claim_type', 'confidence'
        """
        prompt = f"""You are a research analyst. Extract claims about "{concept_name}" from the text below.
For each claim, provide:
1. claim_text: The actual claim or statement
2. claim_type: One of [supports, contradicts, extends, complements]
3. confidence: 0-1 score of extraction confidence

Return ONLY a JSON array of objects with keys: claim_text, claim_type, confidence

Text:
{text[:2000]}

JSON output:"""

        try:
            response_text = await self._call_llm(prompt)
            claims = self._parse_json_response(response_text)

            if not isinstance(claims, list):
                claims = [claims] if claims else []

            # Validate and clean claims
            valid_claim_types = {"supports", "contradicts", "extends", "complements"}
            validated_claims = []

            for claim in claims:
                if isinstance(claim, dict) and "claim_text" in claim:
                    claim_type = claim.get("claim_type", "supports").lower()
                    if claim_type not in valid_claim_types:
                        claim_type = "supports"

                    validated_claims.append({
                        "claim_text": claim.get("claim_text", ""),
                        "claim_type": claim_type,
                        "confidence": float(claim.get("confidence", 0.7))
                    })

            logger.info(f"Extracted {len(validated_claims)} claims for concept '{concept_name}'")
            return validated_claims

        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []

    async def analyze_relationships(
        self,
        concept1: str,
        concept2: str,
        context: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze relationship between two concepts.

        Args:
            concept1: First concept name
            concept2: Second concept name
            context: Optional context text

        Returns:
            Dictionary with 'relationship_type', 'strength', 'confidence', 'explanation'
        """
        prompt = f"""Analyze the relationship between these two concepts:
Concept 1: {concept1}
Concept 2: {concept2}

{f"Context: {context[:1000]}" if context else ""}

Determine:
1. relationship_type: One of [prerequisite, builds_on, contradicts, complements, similar, parent_child]
2. strength: 0-1 score of relationship strength
3. confidence: 0-1 confidence in this analysis
4. explanation: Brief explanation of the relationship

Return ONLY a JSON object with keys: relationship_type, strength, confidence, explanation

JSON output:"""

        try:
            response_text = await self._call_llm(prompt)
            relationship = self._parse_json_response(response_text)

            if not isinstance(relationship, dict):
                return None

            valid_types = {
                "prerequisite", "builds_on", "contradicts",
                "complements", "similar", "parent_child"
            }

            rel_type = relationship.get("relationship_type", "similar").lower()
            if rel_type not in valid_types:
                rel_type = "similar"

            return {
                "relationship_type": rel_type,
                "strength": float(relationship.get("strength", 0.5)),
                "confidence": float(relationship.get("confidence", 0.7)),
                "explanation": relationship.get("explanation", "")
            }

        except Exception as e:
            logger.error(f"Error analyzing relationship: {e}")
            return None

    async def generate_summary(self, texts: List[str], max_length: int = 500) -> str:
        """
        Generate a summary of multiple texts.

        Args:
            texts: List of texts to summarize
            max_length: Maximum length of summary in words

        Returns:
            Summary text
        """
        combined_text = "\n\n".join(texts)[:5000]  # Limit input length

        prompt = f"""Summarize the following research content in {max_length} words or less.
Focus on key findings, concepts, and insights.

Content:
{combined_text}

Summary:"""

        try:
            summary = await self._call_llm(prompt, max_tokens=max_length * 2)
            return summary.strip()

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

    async def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Call Ollama LLM API.

        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": 0.3,  # Lower temperature for more focused outputs
                        }
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    return ""

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return ""

    def _parse_json_response(self, response: str) -> Any:
        """
        Parse JSON from LLM response, handling markdown code blocks.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON object
        """
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            # Remove opening ```json or ```
            response = response.split("\n", 1)[1] if "\n" in response else response[3:]
            # Remove closing ```
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]

        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response}")
            return []
