"""
Candidate Generator for In-Context Reinforcement Learning

Implements Step 1: Generate candidates from training set
- Sample evenly across categories/types
- Generate code examples using modified FPP
- Create embeddings for context + question
- Save candidates with all required information
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import random
from openai import OpenAI
from dotenv import load_dotenv

from ..prompts import load_function_prototypes

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Generate training candidates for In-Context RL from datasets."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize CandidateGenerator.
        
        Args:
            api_key: OpenAI API key
            model: Model for code generation  
            embedding_model: Model for creating embeddings
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_model = embedding_model
        
        # Load function prototypes
        self.function_prototypes = load_function_prototypes()
        
        logger.info(f"CandidateGenerator initialized with {model}")
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def sample_by_type(self, data: List[Dict[str, Any]], 
                      n_total: int = 100,
                      type_field: str = "Type") -> List[Dict[str, Any]]:
        """
        Sample data evenly across types/categories.
        
        Args:
            data: Dataset items
            n_total: Total number of samples
            type_field: Field name for categorization
            
        Returns:
            Sampled items distributed evenly across types
        """
        # Group by type
        type_groups = defaultdict(list)
        for item in data:
            type_groups[item[type_field]].append(item)
        
        types = list(type_groups.keys())
        n_per_type = n_total // len(types)
        remainder = n_total % len(types)
        
        logger.info(f"Sampling {n_total} items across {len(types)} types:")
        for type_name, items in type_groups.items():
            logger.info(f"  {type_name}: {len(items)} available")
        
        sampled = []
        for i, (type_name, items) in enumerate(type_groups.items()):
            # Add extra sample to first few types if there's remainder
            n_samples = n_per_type + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(items))  # Don't exceed available
            
            type_samples = random.sample(items, n_samples)
            sampled.extend(type_samples)
            
            logger.info(f"  Sampled {len(type_samples)} from {type_name}")
        
        random.shuffle(sampled)  # Shuffle final order
        return sampled
    
    def create_code_generation_prompt(self, 
                                    context: str,
                                    question: str, 
                                    answer: float,
                                    explanation: str) -> str:
        """
        Create prompt for generating Python code using FPP approach.
        
        Args:
            context: Problem context (Body)
            question: Question text
            answer: Ground truth answer
            explanation: Explanation/equation
            
        Returns:
            Formatted prompt for code generation
        """
        prompt = f"""You are an expert programmer solving mathematical word problems using predefined functions.

Given the following information:
Context: {context if context else "N/A"}
Question: {question}
Ground Truth Answer: {answer}
Explanation: {explanation}

Available Functions:
{self.function_prototypes}

Your task:
Generate a Python program that solves this problem using the available functions. The program should:
1. Use only the predefined functions above
2. Be concise and readable
3. Calculate the exact answer: {answer}
4. Follow the logic in the explanation: {explanation}

Generate ONLY the Python code (no explanations):"""
        
        return prompt
    
    def generate_code(self, 
                     context: str,
                     question: str,
                     answer: float, 
                     explanation: str) -> Optional[str]:
        """
        Generate Python code for the given problem.
        
        Args:
            context: Problem context
            question: Question text
            answer: Ground truth answer
            explanation: Explanation/equation
            
        Returns:
            Generated Python code or None if failed
        """
        try:
            prompt = self.create_code_generation_prompt(context, question, answer, explanation)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean up code (remove markdown formatting if present)
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for text using OpenAI embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            return None
    
    def generate_candidates(self,
                          dataset_name: str,
                          dataset_path: str,
                          n_candidates: int = 100,
                          output_dir: str = "candidates") -> Dict[str, Any]:
        """
        Generate candidates for the specified dataset.
        
        Args:
            dataset_name: Name of dataset (e.g., "SVAMP")
            dataset_path: Path to dataset JSON file
            n_candidates: Number of candidates to generate
            output_dir: Directory to save candidates
            
        Returns:
            Summary statistics
        """
        logger.info(f"Generating {n_candidates} candidates for {dataset_name}")
        
        # Load and sample data
        data = self.load_dataset(dataset_path)
        sampled_data = self.sample_by_type(data, n_candidates)
        
        # Map dataset fields to standard format
        field_mapping = self._get_field_mapping(dataset_name)
        
        candidates = []
        failed_count = 0
        
        for i, item in enumerate(sampled_data):
            logger.info(f"Processing item {i+1}/{len(sampled_data)}")
            
            # Extract fields using mapping
            context = item.get(field_mapping["context"], "")
            question = item[field_mapping["question"]]
            answer = item[field_mapping["answer"]]
            explanation = item[field_mapping["explanation"]]
            
            # Generate code
            code = self.generate_code(context, question, answer, explanation)
            if not code:
                failed_count += 1
                logger.warning(f"Failed to generate code for item {i+1}")
                continue
            
            # Create embedding
            embedding_text = f"{context}\n\n{question}" if context else question
            embedding = self.create_embedding(embedding_text)
            if not embedding:
                failed_count += 1
                logger.warning(f"Failed to create embedding for item {i+1}")
                continue
            
            # Create candidate
            candidate = {
                "context": context,
                "question": question, 
                "answer": answer,
                "explanation": explanation,
                "code": code,
                "embedding": embedding,
                "type": item.get("Type", "Unknown"),  # Include type for analysis
                "id": item.get("ID", f"item_{i}")
            }
            
            candidates.append(candidate)
        
        # Save candidates
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dataset_name}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
        
        # Summary
        summary = {
            "dataset": dataset_name,
            "requested_candidates": n_candidates,
            "successful_candidates": len(candidates),
            "failed_generations": failed_count,
            "output_path": output_path,
            "type_distribution": Counter([c["type"] for c in candidates])
        }
        
        logger.info(f"Generated {len(candidates)} candidates for {dataset_name}")
        logger.info(f"Failed: {failed_count}, Success rate: {len(candidates)/(len(candidates)+failed_count)*100:.1f}%")
        logger.info(f"Saved to: {output_path}")
        
        return summary
    
    def _get_field_mapping(self, dataset_name: str) -> Dict[str, str]:
        """Get field mapping for different datasets."""
        mappings = {
            "SVAMP": {
                "context": "Body",
                "question": "Question", 
                "answer": "Answer",
                "explanation": "Equation"
            },
            # Add more dataset mappings as needed
        }
        
        return mappings.get(dataset_name, {
            "context": "context",
            "question": "question",
            "answer": "answer", 
            "explanation": "explanation"
        }) 