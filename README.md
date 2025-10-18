# LLM vs Traditional ML: Clinical Prediction Validation

## 🎯 Project Overview
This project validates the research findings that traditional ML models outperform Large Language Models (LLMs) for clinical prediction tasks using NHANES health data.

## 🔬 Research Question
**Do traditional ML models outperform pre-trained LLMs for health status prediction?**

## 📊 Dataset
- **Source**: NHANES health status data
- **Size**: 5,331 samples
- **Features**: 20 features (physical activity, demographics, socioeconomic)
- **Target**: Health_status (0=Good Health, 1=Poor Health)

## 🤖 Models Evaluated

### Traditional ML Models (4 models)
- **Random Forest**: Ensemble method with balanced class weights
- **Logistic Regression**: Linear classifier with balanced class weights
- **Support Vector Machine**: Non-linear classifier with balanced class weights
- **XGBoost**: Gradient boosting with scale_pos_weight for imbalance

### LLM Models (2 models)
- **Llama 3.1 8B**: Meta's large language model via Ollama
- **Mistral 7B**: Mistral AI's language model via Ollama

## 🚀 Quick Start

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install and setup Ollama (for LLM evaluation)
python scripts/setup_ollama.py
```

### Run Complete Experiment
```bash
python scripts/run_full_experiment.py
```

### Run Individual Components
```bash
# Data preparation
python -m src.data_preparation.data_loader

# ML model training
python -m src.traditional_ml.model_trainer

# LLM evaluation
python -m src.llm_evaluation.llm_evaluator

# Analysis and comparison
python -m src.analysis.performance_comparison
```

## 📁 Project Structure
```
Phase_two_project/
├── data/                    # Data files (symbolic links to avoid duplication)
├── src/                     # Source code modules
│   ├── data_preparation/    # Data loading and validation
│   ├── traditional_ml/      # ML model implementations
│   ├── llm_evaluation/      # LLM evaluation system
│   ├── analysis/           # Performance comparison and statistics
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks for exploration
├── experiments/            # Experimental scripts
├── results/               # Output results and visualizations
├── docs/                  # Documentation
├── tests/                 # Unit tests
└── scripts/               # Setup and execution scripts
```

## 🔬 Methodology
This project replicates the methodology from research papers that compare traditional ML with LLMs for clinical prediction tasks:

1. **Data Processing**: Load, validate, and split NHANES data
2. **Traditional ML**: Train 4 models using scikit-learn
3. **LLM Evaluation**: Use Ollama (free, local) to evaluate 2 LLM models
4. **Comparison**: Statistical analysis of performance differences
5. **Validation**: Test if traditional ML > LLMs for this task

## 📈 Results
*Results will be updated as experiments are completed*

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add type hints where appropriate
- Write unit tests for all modules

## 📄 License
MIT License - see LICENSE file for details

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Documentation
- [Methodology](docs/methodology.md)
- [Ollama Setup](docs/ollama_setup.md)
- [Results Interpretation](docs/results_interpretation.md)
- [Paper Validation Report](docs/paper_validation_report.md)
