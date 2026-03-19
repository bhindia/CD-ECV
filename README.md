# CD-ECV: Deterministic Evidence-Grounded Scientific Claim Verification

CD-ECV (Citation-Driven Evidence-Constrained Verification) is a deterministic framework designed to improve reliability in automated scientific claim verification systems. The system enforces evidence grounding by constraining predictions to verifiable sentences retrieved from source documents, reducing hallucination and unsupported reasoning in AI pipelines.

This project investigates how structured retrieval, deterministic safeguards, and evidence-constrained reasoning can improve trustworthiness in language-model-assisted scientific analysis.

---

## Project Motivation

Large language models often generate fluent explanations that are not fully grounded in verifiable evidence. This can introduce hallucinations and unsupported claims, especially in scientific or high-impact domains.

CD-ECV addresses this problem by introducing a **deterministic evidence-grounded pipeline**, where predictions are derived strictly from retrieved evidence sentences rather than unconstrained generation.

The goal is to explore how structured retrieval and verification mechanisms can improve factual reliability and transparency in automated reasoning systems.

---

## System Overview

The CD-ECV pipeline consists of several stages:

### 1. Retrieval
Candidate evidence sentences are retrieved from a document corpus using a hybrid retrieval approach.

- BM25 lexical retrieval
- Dense embedding retrieval
- Candidate document prefiltering

### 2. Evidence Filtering
Retrieved sentences are filtered using relevance and focus scoring methods.

Filtering criteria include:
- lexical overlap
- semantic similarity
- focus scoring

Low-density evidence claims may be rejected early to avoid unsupported predictions.

### 3. Verification
A cross-encoder Natural Language Inference (NLI) model evaluates claim–evidence pairs and predicts the relationship between them.

Possible labels include:

- **SUPPORT**
- **CONTRADICT**
- **NOT_ENOUGH_INFO**

### 4. Evidence-Grounded Output
Predictions are produced together with supporting evidence sentences to ensure traceability and interpretability.

## Installation

Clone the repository and install the required dependencies. 
git clone https://github.com/bhindia/CD-ECV.git
cd CD-ECV
pip install -r requirements.txt

## Running the System

To run the verification pipeline:

python main.py

This will:

Load the corpus and claim dataset

Retrieve candidate evidence sentences

Filter and rerank evidence

Run NLI verification

Produce predictions and evaluation metrics

``````bash