# Dataset Overview

This repository contains five mathematical reasoning datasets used for evaluating our novel prompting technique combined with a reinforcement learning Policy Network for few-shot prompting sample selection. Each dataset presents unique challenges in mathematical reasoning, from basic arithmetic to complex financial analysis.

## Datasets Summary

| Dataset | Type | Domain | Size (Train/Test) | Format |
|---------|------|--------|-------------------|---------|
| **SVAMP** | Arithmetic Word Problems | Elementary Math | 3,138 / 1,000 | JSON |
| **GSM8K** | Grade School Math | Elementary Math | 7,473 / 1,319 | JSONL |
| **TabMWP** | Tabular Math Word Problems | Math + Tables | 23,059 / 1,000 | JSON |
| **TAT-QA** | Table-and-Text QA | Financial/Tabular | 13,215 / 2,199 | JSON |
| **FinQA** | Financial QA | Financial Analysis | 8,281 / 1,147 | JSON |

---

## Dataset Descriptions

### 1. SVAMP (Simple Variations on Arithmetic Math word Problems)

**Overview**: SVAMP focuses on simple arithmetic word problems designed to test basic mathematical reasoning capabilities. The dataset contains problems involving addition, subtraction, multiplication, and division operations.

**Key Features**:
- **Domain**: Elementary arithmetic operations
- **Problem Structure**: Short word problems with clear numerical relationships
- **Operations**: Addition, Subtraction, Multiplication, Division
- **Format**: Each problem includes the body text, question, equation, answer, and operation type

**Example**:
```json
{
    "ID": "chal-363",
    "Body": "Mary is baking a cake. The recipe calls for 6 cups of flour 8 cups of sugar and 7 cups of salt. She already put in 5 cups of flour.",
    "Question": "How many more cups of sugar than cups of salt does she need to add now?",
    "Equation": "( 8.0 - 7.0 )",
    "Answer": 1.0,
    "Type": "Subtraction"
}
```

**Files**:
- `train.json` (3,138 problems)
- `test.json` (1,000 problems)
- `icl/representatives.json` (representative examples for few-shot learning)

---

### 2. GSM8K (Grade School Math 8K)

**Overview**: GSM8K is a dataset of high-quality, grade-school level math word problems requiring multi-step reasoning. Each problem comes with a natural language solution that breaks down the reasoning process.

**Key Features**:
- **Domain**: Grade school mathematics (elementary to middle school level)
- **Complexity**: Multi-step problems requiring sequential reasoning
- **Solutions**: Step-by-step natural language explanations
- **Answer Format**: Final numerical answer with step-by-step working

**Example**:
```json
{
    "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
}
```

**Files**:
- `train.jsonl` (7,473 problems)
- `test.jsonl` (1,319 problems)
- `example_model_solutions.jsonl` (model-generated solutions)
- `test_socratic.jsonl` and `train_socratic.jsonl` (Socratic questioning format)
- `icl/representatives.json`

---

### 3. TabMWP (Tabular Math Word Problems)

**Overview**: TabMWP combines mathematical reasoning with table comprehension. Problems require understanding tabular data and performing mathematical operations based on the information presented in tables.

**Key Features**:
- **Domain**: Mathematical reasoning with tabular data
- **Tables**: Various formats including statistical tables, charts, and data presentations
- **Question Types**: Both multiple choice and free-text answers
- **Grade Levels**: Problems span from elementary to high school level
- **Complexity**: Requires both table comprehension and mathematical computation

**Example**:
```json
{
    "question": "Some friends discussed the sizes of their coin collections. What is the mean of the numbers?",
    "table_title": "Coin collections",
    "table": "Name | Number of coins\nBraden | 76\nCamilla | 94\nRick | 86\nMary | 84\nHector | 80\nDevin | 83\nEmily | 82\nAvery | 87",
    "answer": "84",
    "solution": "Read the numbers from the table...\nThe mean is 84."
}
```

**Files**:
- `train.json` (23,059 problems)
- `test.json` (1,000 problems)
- `dev.json` and `dev1k.json` (development sets)
- `splits.json` (data split information)
- `icl/representatives.json`

---

### 4. TAT-QA (Table-and-Text Question Answering)

**Overview**: TAT-QA focuses on numerical reasoning over financial reports containing both tables and text. It requires understanding complex financial documents and performing calculations based on hybrid textual and tabular information.

**Key Features**:
- **Domain**: Financial document analysis
- **Input Format**: Tables and accompanying text from financial reports
- **Reasoning**: Numerical reasoning, arithmetic operations, and percentage calculations
- **Real-world Data**: Based on actual financial reports and documents
- **Multi-modal**: Combines textual understanding with tabular data processing

**Key Characteristics**:
- Problems require reading comprehension of financial contexts
- Numerical reasoning over multiple data sources
- Complex multi-step calculations involving financial metrics
- Integration of tabular and textual information

**Files**:
- `train.json` (13,215 examples)
- `test.json` (2,199 examples)
- `dev.json` (development set)
- `test_raw.json` (raw test data)
- `icl/representatives.json`

---

### 5. FinQA (Financial Question Answering)

**Overview**: FinQA is a large-scale dataset for numerical reasoning over financial reports. It requires understanding complex financial documents and performing multi-step numerical reasoning to answer questions about financial performance and metrics.

**Key Features**:
- **Domain**: Financial document analysis and numerical reasoning
- **Document Types**: Earnings reports, financial statements, and annual reports
- **Reasoning Types**: Arithmetic operations, percentage calculations, financial ratios
- **Multi-step Solutions**: Complex reasoning chains involving multiple calculations
- **Real Financial Data**: Based on actual corporate financial documents

**Key Challenges**:
- Long, complex financial documents with dense information
- Multi-hop reasoning across different parts of documents
- Financial domain knowledge requirements
- Integration of quantitative and qualitative information

**Files**:
- `train.json` (8,281 examples)
- `test.json` (1,147 examples)
- `dev.json` (development set)
- `private_test.json` (held-out test set)
- `icl/representatives.json`

---

## In-Context Learning (ICL) Components

Each dataset includes an `icl/` directory containing:

- **`representatives.json`**: Carefully selected representative examples for few-shot prompting
- These examples are used by our Policy Network to select optimal demonstration samples
- The representatives are chosen to cover diverse problem types and solution strategies within each dataset

## Vector Database Support

Some datasets (SVAMP and FinQA) include `vdb/` directories containing:
- **`faiss_index`**: FAISS vector index for similarity-based example retrieval
- **`id_mapping.pkl`**: Mapping between vector indices and example IDs
- These support efficient similarity-based example selection for few-shot learning

---

## Usage in Our Framework

These datasets are integrated into our prompting framework as follows:

1. **Training Phase**: The Policy Network learns to select optimal few-shot examples from the ICL representatives
2. **Evaluation Phase**: Problems from test sets are solved using our adaptive prompting technique
3. **Example Selection**: Vector databases and representatives enable efficient similarity-based and diversity-based example selection
4. **Multi-Dataset Learning**: The framework can leverage cross-dataset knowledge transfer

## Citation

If you use these datasets in your research, please cite the original papers:

- **SVAMP**: Patel et al. (2021)
- **GSM8K**: Cobbe et al. (2021) 
- **TabMWP**: Lu et al. (2022)
- **TAT-QA**: Zhu et al. (2021)
- **FinQA**: Chen et al. (2021)

For our specific preprocessing and ICL components, please cite our paper [Your Paper Citation]. 