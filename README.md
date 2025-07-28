# Round 1B Submission: The Hybrid AI + Classic NLP Document Analyst

This project is a solution for the "Connecting the Dots" hackathon, Round 1B. It is an intelligent document analysis pipeline designed to process a collection of PDF documents and extract the most relevant sections based on a user's specific persona and job-to-be-done.

## Our Approach: A Fused, Multi-Module Hybrid Pipeline

After extensive testing across three diverse document collections (narrative travel guides, technical manuals, and structured recipe books), we concluded that no single strategy was sufficiently robust. Our final solution is a sophisticated **hybrid pipeline** that fuses the strengths of classic Natural Language Processing (NLP) with state-of-the-art Deep Learning models to achieve a superior, generalized result.

The system is a modular pipeline where each component is a specialized "brain" for its task:

1.  **`DocumentProcessor` (The "Structure Brain"):** A robust, rule-based parser using `pdfplumber`. It reliably extracts a clean, structured list of sections from the raw PDFs. It uses a powerful set of heuristics based on text patterns (numbering, capitalization), keywords, and layout to identify headings without relying on a brittle AI model.

2.  **`PersonaJobAnalyzer` (The "Requirements Brain"):** An NLP-powered module using `spaCy` and `NLTK`. It performs a deep, linguistic analysis of the user's request to extract a rich set of weighted keywords, key concepts, and the user's "domain focus." This provides a classic, keyword-based understanding of the query.

3.  **`QueryGenerator` (The "Creative Brain"):** A generative AI model (`google-t5/t5-small`). It acts as a brainstorming assistant, taking the user's goal and generating a new, creative set of semantically related search queries to find content that keywords alone might miss.

4.  **`HybridRelevanceScorer` (The "Ranking Brain"):** The core of our fusion strategy. This module ranks every parsed section by calculating and combining two distinct scores:
    *   A **keyword score** using classic `TF-IDF` and the weighted keywords from the `PersonaJobAnalyzer`.
    *   A **deep semantic score** using the powerful `BAAI/bge-base-en-v1.5` model and the creative queries from the `QueryGenerator`.
    This fusion makes the final ranking both precise (keyword-based) and context-aware (semantic).

5.  **`SubSectionAnalyzer` (The "Summarizer Brain"):** The final stage. It takes the top-ranked sections from the scorer and uses its advanced logic to perform a granular analysis, extracting the most relevant paragraphs and sentences to produce the final `refined_text`.

This hybrid architecture makes our solution both **resilient** (it works on diverse document layouts) and **intelligent** (it understands the user's full semantic intent).

## Models and Libraries Used

### AI Models
*   **Semantic Ranking Model:** `BAAI/bge-base-en-v1.5` (State-of-the-art sentence transformer for semantic search).
*   **Query Generation Model:** `google-t5/t5-small` (Generative model for dynamic query expansion).

### Core Libraries
*   **NLP:** `spaCy` (for linguistic analysis), `NLTK` (for tokenization).
*   **PDF Parsing:** `pdfplumber`.
*   **Keyword Scoring:** `scikit-learn (TfidfVectorizer)`.
*   **Deep Learning Framework:** `PyTorch`.
*   **Model Access:** `transformers`, `sentence_transformers`.

## How to Build and Run Your Solution

The solution is packaged as a Docker container for easy and reliable execution. The following instructions are for documentation purposes, as the solution will be run using the "Expected Execution" section of the hackathon brief.

### Prerequisites
*   Docker Desktop installed and running.

### Build the Docker Image

Navigate to the root directory of the project (where the `Dockerfile` is located) and run the following command. The `.` at the end is important.

```bash
docker build -t mysolutionname:somerandomidentifier .
```

This command will:
1.  Install all necessary system and Python dependencies from `requirements.txt`.
2.  Run the `download_models.py` script inside the container to pre-download all required AI models (`bge-base`, `t5-small`) and NLP data (`spaCy`, `NLTK`) for **fully offline execution.**
3.  Copy the complete application source code into the image.

### Run the Docker Container

The container is designed to process PDFs from an `input` directory and write its results to an `output` directory.

1.  Place your test PDFs and a corresponding `input.json` file inside a local `input` folder.
2.  Create an empty local `output` folder.
3.  From your project's root directory, run the command as specified in the hackathon brief:

```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none mysolutionname:somerandomidentifier
```

*   `--rm`: Automatically deletes the container after it finishes.
*   `-v "$(pwd)/input:/app/input"`: Mounts your local `input` folder to the container.
*   `-v "$(pwd)/output:/app/output"`: Mounts your local `output` folder from the container, allowing you to access the `output.json`.
*   `--network none`: **Crucially, this runs the container completely offline.**

The container will execute the `src/main.py` script, processing all documents and writing a final `output.json` to your local `output` folder.