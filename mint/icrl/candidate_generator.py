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
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from ..prompts import load_function_prototypes
from ..config import load_config
from ..utils import execute_code, evaluate_result, clean_code

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Generate training candidates for In-Context RL from datasets."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize CandidateGenerator.
        
        Args:
            api_key: OpenAI API key
            model: Model for code generation (uses DEFAULT_MODEL from config if None)
            embedding_model: Model for creating embeddings
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        # Load config and use DEFAULT_MODEL if no model specified
        config = load_config()
        self.model = model or config['model']
        self.embedding_model = embedding_model
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Load function prototypes
        self.function_prototypes = load_function_prototypes()
        
        logger.info(f"CandidateGenerator initialized with model: {self.model}")
        logger.info(f"Using embedding model: {self.embedding_model}")
        
        self.dataset_configs = {
            "SVAMP": {
                "path": "datasets/SVAMP/train.json",
                "format": "json",
                "sampling": "type_based",
                "type_field": "Type",
                "question_field": "Body", 
                "answer_field": "Answer",
                "explanation_field": "Equation",
                "context_field": "Title",
                "clean_answer_method": "clean_svamp_answer"
            },
            "GSM8K": {
                "path": "datasets/GSM8K/train.jsonl", 
                "format": "jsonl",
                "sampling": "random",
                "question_field": "question",
                "answer_field": "answer", 
                "explanation_field": "answer",  # Use parse_gsm8k_answer to split
                "context_field": None,  # No context field
                "clean_answer_method": "clean_gsm8k_answer"
            },
            "TabMWP": {
                "path": "datasets/TabMWP/train.json",
                "format": "json_dict_keys",
                "sampling": "grade_based", 
                "grade_field": "grade",
                "question_field": "question",
                "answer_field": "answer",
                "explanation_field": "solution",
                "context_field": None,  # Use table processing
                "table_field": "table_for_pd",
                "table_title_field": "table_title", 
                "filter_field": "ques_type",
                "filter_value": "free_text",
                "clean_answer_method": "clean_tabmwp_answer"
            },
            "TAT-QA": {
                "path": "datasets/TAT-QA/train.json",
                "format": "json",
                "sampling": "random",  # Since expanded data will be pre-filtered
                "question_field": "question",
                "answer_field": "answer", 
                "explanation_field": "derivation",
                "context_field": None,  # Use create_tatqa_context
                "clean_answer_method": "clean_tatqa_answer",
                "requires_expansion": True  # Multiple questions per item
            },
            "FinQA": {
                "path": "datasets/FinQA/train.json",
                "format": "json",
                "sampling": "random",
                "question_field": "qa.question",  # Nested in qa dict
                "answer_field": "qa.answer",
                "explanation_field": "qa",  # Use create_finqa_explanation with full qa dict
                "context_field": None,  # Use create_finqa_context
                "clean_answer_method": "clean_finqa_answer"
            }
        }
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON or JSONL file."""
        data = []
        
        try:
            if dataset_path.endswith('.jsonl'):
                # Load JSONL format (GSM8K)
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line.strip()))
            else:
                # Load JSON format (SVAMP, TabMWP, etc.)
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    
                # Handle different JSON structures
                if isinstance(raw_data, list):
                    # Standard list format
                    data = raw_data
                elif isinstance(raw_data, dict):
                    # Dictionary with numbered keys (like TabMWP)
                    data = list(raw_data.values())
                else:
                    raise ValueError(f"Unsupported data format: {type(raw_data)}")
                    
            logger.info(f"Loaded {len(data)} items from {dataset_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            raise
    
    def filter_tabmwp_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter TabMWP data to only include free_text questions."""
        filtered = [item for item in data if item.get("ques_type") == "free_text"]
        logger.info(f"Filtered TabMWP: {len(filtered)}/{len(data)} free_text questions")
        return filtered

    def clean_tabmwp_answer(self, answer_str: str) -> float:
        """Clean TabMWP answer string and convert to float."""
        try:
            # Remove commas from numbers like "1,669" -> "1669"
            cleaned = str(answer_str).replace(",", "")
            return float(cleaned)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse TabMWP answer '{answer_str}': {e}")
            raise

    def table_to_markdown(self, table_data: Any) -> str:
        """Convert table data to markdown format."""
        try:
            if isinstance(table_data, list):
                # Table is list of lists (rows)
                if not table_data:
                    return "Empty table"
                
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                
                # Try to_markdown with tabulate
                try:
                    return df.to_markdown(index=False)
                except ImportError:
                    # Fallback to CSV-like format if tabulate not available
                    logger.warning("tabulate not available, using CSV fallback for table")
                    return df.to_csv(index=False, sep='|')
            
            elif isinstance(table_data, dict):
                # Table is dictionary format
                df = pd.DataFrame(table_data)
                
                # Try to_markdown with tabulate
                try:
                    return df.to_markdown(index=False)
                except ImportError:
                    # Fallback to CSV-like format if tabulate not available
                    logger.warning("tabulate not available, using CSV fallback for table")
                    return df.to_csv(index=False, sep='|')
            
            else:
                # Fallback for other formats
                return str(table_data)
                
        except Exception as e:
            logger.warning(f"Failed to convert table to markdown: {e}")
            return str(table_data)

    def create_tabmwp_context(self, item: Dict[str, Any]) -> str:
        """Create context for TabMWP item by combining title and table."""
        title = item.get("table_title", "")
        table_data = item.get("table_for_pd", item.get("table", ""))
        
        if not table_data:
            return title
        
        # Convert table to markdown
        table_markdown = self.table_to_markdown(table_data)
        
        # Combine title and table
        if title:
            return f"{title}\n\n{table_markdown}"
        else:
            return table_markdown
    
    def sample_by_type(self, data: List[Dict[str, Any]], 
                      n_total: int = 100,
                      type_field: str = "Type") -> List[Dict[str, Any]]:
        """
        Sample data evenly across types/categories, or randomly if no types.
        
        Args:
            data: Dataset items
            n_total: Total number of samples
            type_field: Field name for categorization (None for random sampling)
            
        Returns:
            Sampled items distributed evenly across types or randomly
        """
        # Check if type field exists in data
        has_types = type_field and len(data) > 0 and type_field in data[0]
        
        if not has_types:
            # Random sampling (for datasets like GSM8K without types)
            n_samples = min(n_total, len(data))
            sampled = random.sample(data, n_samples)
            logger.info(f"Random sampling {n_samples} items from {len(data)} total")
            return sampled
        
        # Group by type (for datasets like SVAMP)
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
    
    def validate_generated_code(self, code: str, expected_answer: float) -> Tuple[bool, str, Any]:
        """
        Validate generated code by executing it and checking result.
        
        Args:
            code: Generated Python code
            expected_answer: Expected numerical answer
            
        Returns:
            Tuple of (is_valid, cleaned_code, execution_result)
        """
        try:
            # Clean the code first
            cleaned_code = clean_code(code)
            
            # Execute the code
            result, error = execute_code(cleaned_code)
            
            if error:
                logger.debug(f"Code execution failed: {error}")
                return False, cleaned_code, None
            
            # Check if result matches expected answer
            if result is not None and evaluate_result(result, expected_answer):
                logger.debug(f"Code validation passed: {result} == {expected_answer}")
                return True, cleaned_code, result
            else:
                logger.debug(f"Code result mismatch: {result} != {expected_answer}")
                return False, cleaned_code, result
                
        except Exception as e:
            logger.debug(f"Code validation error: {e}")
            return False, code, None
    
    def generate_code(self, 
                     context: str,
                     question: str,
                     answer: float, 
                     explanation: str,
                     max_retries: int = 3) -> Optional[str]:
        """
        Generate and validate Python code for the given problem.
        
        Args:
            context: Problem context
            question: Question text
            answer: Ground truth answer
            explanation: Explanation/equation
            max_retries: Maximum number of generation attempts
            
        Returns:
            Validated Python code or None if failed
        """
        prompt = self.create_code_generation_prompt(context, question, answer, explanation)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                raw_code = response.choices[0].message.content.strip()
                
                # Clean up code (remove markdown formatting if present)
                if raw_code.startswith("```python"):
                    raw_code = raw_code[9:]
                if raw_code.startswith("```"):
                    raw_code = raw_code[3:]
                if raw_code.endswith("```"):
                    raw_code = raw_code[:-3]
                
                code = raw_code.strip()
                
                # Validate the generated code
                is_valid, cleaned_code, result = self.validate_generated_code(code, answer)
                
                if is_valid:
                    logger.debug(f"Code generation successful on attempt {attempt + 1}")
                    return cleaned_code
                else:
                    logger.debug(f"Code validation failed on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        logger.debug("Retrying code generation...")
                        
            except Exception as e:
                logger.error(f"Code generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.debug("Retrying due to API error...")
        
        logger.warning(f"Failed to generate valid code after {max_retries} attempts")
        return None
    
    def parse_gsm8k_answer(self, answer_text: str) -> tuple[str, str]:
        """
        Parse GSM8K answer field to extract explanation and final answer.
        
        Args:
            answer_text: Full answer text in format "explanation #### final_answer"
            
        Returns:
            Tuple of (explanation, final_answer)
        """
        if "####" in answer_text:
            parts = answer_text.split("####")
            explanation = parts[0].strip()
            final_answer = parts[1].strip()
            return explanation, final_answer
        else:
            # Fallback if no #### separator found
            return answer_text.strip(), "0"
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for text using OpenAI embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Clean text for embedding
            clean_text = text.strip()
            if not clean_text:
                logger.warning("Empty text for embedding")
                return None
                
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=clean_text
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
            dataset_name: Name of dataset (e.g., "SVAMP", "GSM8K", "TabMWP")
            dataset_path: Path to dataset JSON/JSONL file
            n_candidates: Number of candidates to generate
            output_dir: Directory to save candidates
            
        Returns:
            Summary statistics
        """
        logger.info(f"Generating {n_candidates} candidates for {dataset_name}")
        
        # Load and sample data
        raw_data = self.load_dataset(dataset_path)
        
        # Apply dataset-specific filtering
        if dataset_name == "TabMWP":
            # Filter for free_text questions only
            data = self.filter_tabmwp_data(raw_data)
            if len(data) == 0:
                raise ValueError("No free_text questions found in TabMWP dataset")
        elif dataset_name == "TAT-QA":
            # Expand TAT-QA data to individual arithmetic questions
            data = self.filter_tatqa_data(raw_data)
            if len(data) == 0:
                raise ValueError("No arithmetic questions found in TAT-QA dataset")
        else:
            data = raw_data
        
        # Get field mapping and type field
        field_mapping = self._get_field_mapping(dataset_name)
        type_field = field_mapping.get("type_field")
        
        sampled_data = self.sample_by_type(data, n_candidates, type_field)
        
        candidates = []
        failed_count = 0
        max_total_attempts = n_candidates * 5  # Prevent infinite loop
        total_attempts = 0
        used_indices = set()  # Track used samples to avoid duplicates
        
        i = 0
        while len(candidates) < n_candidates and total_attempts < max_total_attempts:
            total_attempts += 1
            
            # If we've exhausted initial samples, get random replacement
            if i >= len(sampled_data):
                # Get unused samples from full dataset
                available_indices = [idx for idx in range(len(data)) if idx not in used_indices]
                if not available_indices:
                    logger.warning("No more unused samples available")
                    break
                    
                # Random select from available samples  
                new_idx = random.choice(available_indices)
                item = data[new_idx]
                used_indices.add(new_idx)
                logger.info(f"Using replacement sample (index {new_idx}) after failures")
            else:
                item = sampled_data[i]
                # Track original sampled data indices if possible
                if hasattr(item, 'original_index'):
                    used_indices.add(item.original_index)
                i += 1
            
            logger.info(f"Processing candidate {len(candidates)+1}/{n_candidates} (attempt {total_attempts})")
            
            # Extract fields using mapping with dataset-specific handling
            if dataset_name == "TabMWP":
                context = self.create_tabmwp_context(item)
                question = item[field_mapping["question"]]
                answer = self.clean_tabmwp_answer(item[field_mapping["answer"]])
                explanation = item[field_mapping["explanation"]]
            elif dataset_name == "GSM8K":
                context = ""  # No context for GSM8K
                question = item[field_mapping["question"]]
                explanation, answer_str = self.parse_gsm8k_answer(item[field_mapping["answer"]])
                try:
                    answer = float(answer_str)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid answer format: {answer_str}, trying replacement")
                    failed_count += 1
                    continue
            elif dataset_name == "TAT-QA":
                # TAT-QA has special processing for expanded questions
                context = self.create_tatqa_context(item["context_data"])  # Use stored original context
                question = item[field_mapping["question"]]
                try:
                    answer = self.clean_tatqa_answer(item[field_mapping["answer"]])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid TAT-QA answer format: {item[field_mapping['answer']]}, error: {e}")
                    failed_count += 1
                    continue
                explanation = item[field_mapping["explanation"]]
            elif dataset_name == "FinQA":
                # FinQA has special processing for its structure
                context = self.create_finqa_context(item)
                question = self.get_nested_field(item, field_mapping["question"])
                try:
                    raw_answer = self.get_nested_field(item, field_mapping["answer"])
                    answer = self.clean_finqa_answer(raw_answer)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid FinQA answer format: {raw_answer}, error: {e}")
                    failed_count += 1
                    continue
                # Pass full qa dict to create_finqa_explanation
                qa_dict = item.get('qa', {})
                explanation = self.create_finqa_explanation(qa_dict)
            else:
                # For SVAMP and other datasets
                context = item.get(field_mapping["context"], "") if field_mapping["context"] else ""
                question = item[field_mapping["question"]]
                answer = item[field_mapping["answer"]]
                explanation = item[field_mapping["explanation"]]
            
            # Generate code with retries
            code = self.generate_code(context, question, answer, explanation)
            if not code:
                failed_count += 1
                logger.warning(f"Failed to generate valid code after retries, trying replacement")
                continue
            
            # Create embedding (context + question for datasets with context, question only for GSM8K)
            if context:
                embedding_text = f"{context}\n\n{question}"
            else:
                embedding_text = question
                
            embedding = self.create_embedding(embedding_text)
            if not embedding:
                failed_count += 1
                logger.warning(f"Failed to create embedding, trying replacement")
                continue
            
            # Create candidate
            candidate = {
                "context": context,
                "question": question, 
                "answer": answer,
                "explanation": explanation,
                "code": code,
                "embedding": embedding,
                "type": item.get(type_field, "Unknown") if type_field else "Unknown",
                "id": item.get("ID", item.get("id", f"item_{total_attempts}"))
            }
            
            candidates.append(candidate)
            logger.info(f"✅ Successfully generated candidate {len(candidates)}/{n_candidates}")
        
        # Log final statistics
        if len(candidates) < n_candidates:
            logger.warning(f"Could only generate {len(candidates)}/{n_candidates} candidates after {total_attempts} attempts")
        
        # Save candidates
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dataset_name}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
        
        # Summary
        type_distribution = Counter([c["type"] for c in candidates])
        summary = {
            "dataset": dataset_name,
            "requested_candidates": n_candidates,
            "successful_candidates": len(candidates),
            "failed_generations": failed_count,
            "total_attempts": total_attempts,
            "output_path": output_path,
            "type_distribution": type_distribution
        }
        
        logger.info(f"Generated {len(candidates)} candidates for {dataset_name}")
        logger.info(f"Failed attempts: {failed_count}, Total attempts: {total_attempts}")
        logger.info(f"Success rate: {len(candidates)/total_attempts*100:.1f}%")
        logger.info(f"Saved to: {output_path}")
        
        return summary
    
    def _get_field_mapping(self, dataset_name: str) -> Dict[str, str]:
        """Get field mapping for different datasets."""
        mappings = {
            "SVAMP": {
                "context": "Body",
                "question": "Question", 
                "answer": "Answer",
                "explanation": "Equation",
                "type_field": "Type"
            },
            "GSM8K": {
                "context": None,  # No context field
                "question": "question",
                "answer": "answer",  # Will be parsed to extract explanation and answer
                "explanation": None,  # Extracted from answer field
                "type_field": None  # No type field - random sampling
            },
            "TabMWP": {
                "context": None,  # Will be created from table_title + table_for_pd
                "question": "question",
                "answer": "answer",
                "explanation": "solution",
                "type_field": "grade"  # Sample by grade (1-8)
            },
            "TAT-QA": {
                "context": None,  # Use create_tatqa_context
                "question": "question",
                "answer": "answer",
                "explanation": "derivation",
                "type_field": None  # No type field - random sampling
            },
            "FinQA": {
                "context": None,  # Use create_finqa_context
                "question": "qa.question",  # Nested in qa dict
                "answer": "qa.answer",
                "explanation": "qa",  # Use create_finqa_explanation with full qa dict
                "type_field": None  # No type field - random sampling
            },
            # Add more dataset mappings as needed
        }
        
        return mappings.get(dataset_name, {
            "context": "context",
            "question": "question",
            "answer": "answer", 
            "explanation": "explanation",
            "type_field": "Type"
        }) 

    def process_tatqa_table(self, table_data: Dict[str, Any]) -> str:
        """Process TAT-QA table data to markdown string."""
        try:
            if 'table' not in table_data:
                return ""
            
            table_list = table_data['table']
            if not table_list or len(table_list) < 2:
                return str(table_list)
            
            # Convert list of lists to DataFrame
            # First row is usually headers
            headers = table_list[0]
            rows = table_list[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            
            # Try to_markdown with tabulate
            try:
                return df.to_markdown(index=False)
            except ImportError:
                # Fallback to CSV-like format if tabulate not available
                logger.warning("tabulate not available, using CSV fallback for TAT-QA table")
                return df.to_csv(index=False, sep='|')
                
        except Exception as e:
            logger.warning(f"Failed to process TAT-QA table: {e}")
            return str(table_data)

    def process_tatqa_paragraphs(self, paragraphs: List[Dict[str, Any]]) -> str:
        """Process TAT-QA paragraphs - sort by order and combine text."""
        try:
            if not paragraphs:
                return ""
            
            # Sort paragraphs by order
            sorted_paragraphs = sorted(paragraphs, key=lambda x: x.get('order', 0))
            
            # Combine text from all paragraphs
            combined_text = []
            for para in sorted_paragraphs:
                text = para.get('text', '').strip()
                if text:
                    combined_text.append(text)
            
            return '\n\n'.join(combined_text)
            
        except Exception as e:
            logger.warning(f"Failed to process TAT-QA paragraphs: {e}")
            return str(paragraphs)

    def create_tatqa_context(self, item: Dict[str, Any]) -> str:
        """Create context for TAT-QA item by combining table and paragraphs."""
        try:
            # Process table
            table_string = ""
            if 'table' in item:
                table_string = self.process_tatqa_table(item['table'])
            
            # Process paragraphs
            paragraph_str = ""
            if 'paragraphs' in item:
                paragraph_str = self.process_tatqa_paragraphs(item['paragraphs'])
            
            # Combine table and paragraphs
            context_parts = []
            if table_string.strip():
                context_parts.append(table_string.strip())
            if paragraph_str.strip():
                context_parts.append(paragraph_str.strip())
            
            return '\n\n'.join(context_parts)
            
        except Exception as e:
            logger.warning(f"Failed to create TAT-QA context: {e}")
            return str(item)

    def extract_tatqa_questions(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract arithmetic questions from TAT-QA item and create expanded items."""
        try:
            questions = item.get('questions', [])
            arithmetic_questions = []
            
            # Filter for arithmetic questions only
            for question_data in questions:
                if question_data.get('answer_type') == 'arithmetic':
                    # Create expanded item with single question
                    expanded_item = {
                        'context_data': item,  # Store original context data
                        'question': question_data.get('question', ''),
                        'answer': question_data.get('answer', ''),
                        'derivation': question_data.get('derivation', ''),
                        'question_uid': question_data.get('uid', ''),
                        'scale': question_data.get('scale', ''),
                        'order': question_data.get('order', 0)
                    }
                    arithmetic_questions.append(expanded_item)
            
            logger.debug(f"Extracted {len(arithmetic_questions)} arithmetic questions from TAT-QA item")
            return arithmetic_questions
            
        except Exception as e:
            logger.warning(f"Failed to extract TAT-QA questions: {e}")
            return []

    def clean_tatqa_answer(self, answer: Any) -> float:
        """Clean TAT-QA answer and convert to float."""
        try:
            if isinstance(answer, list):
                # Sometimes answer is a list, take first element
                answer = answer[0] if answer else "0"
            
            # Convert to string and clean
            answer_str = str(answer).strip()
            
            # Remove common formatting
            answer_str = answer_str.replace(",", "")  # Remove commas
            answer_str = answer_str.replace("$", "")  # Remove dollar signs
            answer_str = answer_str.replace("%", "")  # Remove percentages
            answer_str = answer_str.replace("(", "-").replace(")", "")  # Handle negatives
            
            # Handle empty or non-numeric strings
            if not answer_str or answer_str in ['-', 'n/a', 'N/A', 'nil', 'Nil']:
                return 0.0
            
            return float(answer_str)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse TAT-QA answer '{answer}': {e}")
            raise

    def filter_tatqa_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and expand TAT-QA data to include only arithmetic questions."""
        expanded_data = []
        arithmetic_count = 0
        total_items = len(data)
        
        for item in data:
            # Extract arithmetic questions from this item
            arithmetic_questions = self.extract_tatqa_questions(item)
            expanded_data.extend(arithmetic_questions)
            arithmetic_count += len(arithmetic_questions)
        
        logger.info(f"Filtered TAT-QA: {arithmetic_count} arithmetic questions from {total_items} items")
        return expanded_data 

    def process_finqa_text_array(self, text_array: List[str]) -> str:
        """Process FinQA text arrays (pre_text, post_text) - filter dots and combine."""
        try:
            if not text_array:
                return ""
            
            # Filter out dot-only elements and empty strings
            filtered_texts = []
            for text in text_array:
                cleaned_text = text.strip()
                if cleaned_text and cleaned_text != ".":
                    filtered_texts.append(cleaned_text)
            
            # Join with spaces
            combined_text = " ".join(filtered_texts)
            
            logger.debug(f"Processed text array: {len(text_array)} → {len(filtered_texts)} sentences")
            return combined_text
            
        except Exception as e:
            logger.warning(f"Failed to process FinQA text array: {e}")
            return " ".join(str(t) for t in text_array)

    def process_finqa_table(self, table_data: List[List[str]]) -> str:
        """Process FinQA table data to markdown string."""
        try:
            if not table_data or len(table_data) < 2:
                return ""
            
            # Convert list of lists to DataFrame
            # First row is usually headers
            headers = table_data[0]
            rows = table_data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            
            # Try to_markdown with tabulate
            try:
                return df.to_markdown(index=False)
            except ImportError:
                # Fallback to CSV-like format if tabulate not available
                logger.warning("tabulate not available, using CSV fallback for FinQA table")
                return df.to_csv(index=False, sep='|')
                
        except Exception as e:
            logger.warning(f"Failed to process FinQA table: {e}")
            return str(table_data)

    def create_finqa_context(self, item: Dict[str, Any]) -> str:
        """Create context for FinQA item by combining pre_text + table + post_text."""
        try:
            context_parts = []
            
            # Process pre_text
            pre_text = item.get('pre_text', [])
            if pre_text:
                pre_text_str = self.process_finqa_text_array(pre_text)
                if pre_text_str:
                    context_parts.append(pre_text_str)
            
            # Process table
            table_data = item.get('table', [])
            if table_data:
                table_str = self.process_finqa_table(table_data)
                if table_str:
                    context_parts.append(table_str)
            
            # Process post_text
            post_text = item.get('post_text', [])
            if post_text:
                post_text_str = self.process_finqa_text_array(post_text)
                if post_text_str:
                    context_parts.append(post_text_str)
            
            # Combine all parts with double newlines
            combined_context = '\n\n'.join(context_parts)
            
            logger.debug(f"Created FinQA context: {len(combined_context)} chars from {len(context_parts)} parts")
            return combined_context
            
        except Exception as e:
            logger.warning(f"Failed to create FinQA context: {e}")
            return str(item)

    def create_finqa_explanation(self, qa_data: Dict[str, Any]) -> str:
        """Create comprehensive explanation for FinQA by combining explanation + steps + program."""
        try:
            explanation_parts = []
            
            # Add base explanation
            explanation = qa_data.get('explanation', '').strip()
            if explanation:
                explanation_parts.append(f"Explanation: {explanation}")
            
            # Add steps if available
            steps = qa_data.get('steps', [])
            if steps and isinstance(steps, list):
                steps_str = " → ".join(str(step).strip() for step in steps if str(step).strip())
                if steps_str:
                    explanation_parts.append(f"Steps: {steps_str}")
            
            # Add program if available  
            program = qa_data.get('program', [])
            if program and isinstance(program, list):
                program_str = " | ".join(str(prog).strip() for prog in program if str(prog).strip())
                if program_str:
                    explanation_parts.append(f"Program: {program_str}")
            
            # Combine all parts
            if explanation_parts:
                return " | ".join(explanation_parts)
            else:
                return qa_data.get('explanation', '')
                
        except Exception as e:
            logger.warning(f"Failed to create FinQA explanation: {e}")
            return qa_data.get('explanation', '')

    def get_nested_field(self, item: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation (e.g., 'qa.question')."""
        try:
            fields = field_path.split('.')
            value = item
            for field in fields:
                value = value[field]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Failed to extract nested field '{field_path}' from item")
            return None

    def clean_finqa_answer(self, answer: Any) -> float:
        """Clean FinQA answer and convert to float."""
        try:
            # Convert to string and clean
            answer_str = str(answer).strip()
            
            # Remove common formatting
            answer_str = answer_str.replace(",", "")  # Remove commas
            answer_str = answer_str.replace("$", "")  # Remove dollar signs
            answer_str = answer_str.replace("%", "")  # Remove percentages
            answer_str = answer_str.replace("(", "-").replace(")", "")  # Handle negatives
            answer_str = answer_str.replace("million", "000000")  # Handle million
            answer_str = answer_str.replace("billion", "000000000")  # Handle billion
            
            # Handle empty or non-numeric strings
            if not answer_str or answer_str in ['-', 'n/a', 'N/A', 'nil', 'Nil']:
                return 0.0
            
            return float(answer_str)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse FinQA answer '{answer}': {e}")
            raise 