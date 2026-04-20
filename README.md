# SafeBench AI

SafeBench AI is a comprehensive benchmarking framework for evaluating AI model safety against various harm categories. It tests models using attack sequences designed to elicit harmful responses and benign sequences for baseline performance. The framework includes automated scoring, experiment management, and a visual dashboard for results analysis.

## Features

- **Multi-Model Support**: Test Groq and Google Gemini models
- **Harm Taxonomy**: Covers 7 categories of potential harm including violent content, illegal activities, and misinformation
- **Automated Scoring**: Uses keyword detection, Detoxify classifier, and LLM judge for comprehensive evaluation
- **Experiment Runner**: Run full ablation studies across models, defense configurations, and sequences
- **Interactive Dashboard**: Visualize results with Streamlit-based analytics
- **Jupyter Notebook**: Colab-compatible runner for cloud execution

## Prerequisites

- Python 3.8+
- API keys for Groq and Google Gemini (see Setup below)

## Installation

1. Clone or download the project repository
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the project root with your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
RANDOM_SEED=42
```

2. Test your API connections:

```bash
python test_connections.py
```

This script will verify that both Groq and Gemini APIs are accessible and working correctly.

## Usage

### Running Experiments

To run the full benchmark suite:

```bash
python run_experiments.py
```

You can customize the experiments with command-line arguments:

```bash
# Test specific models
python run_experiments.py --models groq/llama-3.1-8b-instant groq/qwen3-32b groq/llama-4-scout-17b

# Test with specific defenses
python run_experiments.py --defenses none

# Limit the number of sequences for testing
python run_experiments.py --limit 3
```

Results are automatically saved to the `results/` directory with timestamped JSON files.

### Viewing Results

Launch the interactive dashboard to analyze results:

```bash
streamlit run dashboard.py
```

The dashboard provides visualizations of attack success rates, model comparisons, and detailed scoring breakdowns.



## Project Structure

```
SafeBenchAI/
├── core/                    # Core functionality
│   ├── config.py           # Model and experiment configurations
│   ├── runner.py           # Conversation runner for sequences
│   └── __init__.py
├── data/                   # Test data
│   ├── attack_sequences/   # Harmful test sequences
│   └── benign_sequences/   # Benign baseline sequences
├── results/                # Experiment results (auto-generated)
├── scoring/                # Scoring and evaluation
│   ├── scorer.py          # Main scoring logic
│   ├── aggregator.py      # Result aggregation
│   └── __init__.py
├── dashboard.py            # Streamlit dashboard
├── run_experiments.py      # Experiment runner script
├── test_connections.py     # API connection tester
├── colab_hf_runner.ipynb   # Colab/Jupyter notebook
├── harm_taxonomy.md        # Harm categories and scoring rubric
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Configuration

Models and defense configurations are defined in `core/config.py`. You can modify this file to:

- Add new models (requires API key setup)
- Adjust defense mechanisms
- Change experiment parameters
- Modify rate limiting and context windows

## Scoring Methodology

The framework uses a three-signal approach for determining attack success:

1. **Keyword Detection**: Checks for compliance keywords in model responses
2. **Detoxify Classifier**: Uses toxicity detection to identify harmful content
3. **LLM Judge**: Employs another LLM to evaluate response safety

Results are combined into an Attack Success Rate (ASR) metric. See `harm_taxonomy.md` for detailed scoring criteria.


