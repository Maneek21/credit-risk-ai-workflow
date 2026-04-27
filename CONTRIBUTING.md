# Contributing to Credit Risk AI Workflow

Thank you for your interest in contributing! This project bridges AI research
and credit risk practice, so we welcome contributions from both domains.

## How to Contribute

### Reporting Issues

- Use GitHub Issues for bugs, feature requests, or questions.
- For bugs: include Python version, OS, and the full error traceback.
- For feature requests: describe the use case and expected behavior.

### Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Install dev dependencies: `pip install -r requirements.txt`
3. Make your changes with clear, descriptive commits.
4. Add or update tests if applicable.
5. Ensure all existing tests pass: `python -m pytest tests/`
6. Submit a PR with a clear description of what and why.

### Code Style

- Python: follow PEP 8, use type hints for public functions.
- Docstrings: Google style.
- Line length: 88 characters (Black formatter).
- All random operations must use `random_state=42` for reproducibility.

### Areas We'd Love Help With

**New LLM benchmarks:**
- Gemini, Llama 3, Mistral, Command R on the credit evaluation task
- Add your model to `benchmarks/src/llm_zero_shot_eval.py`

**Additional credit models:**
- LightGBM, CatBoost, TabNet
- Add to `workflow/scripts/train_classical.py`

**Regulatory frameworks:**
- EU AI Act compliance checks
- Basel III/IV capital requirement considerations
- Additional FCRA/ECOA requirements

**Production hardening:**
- Rate limiting and retry logic for API calls
- Caching layer for repeated evaluations
- Async/batch processing for large portfolios

**Documentation:**
- Tutorials for specific use cases
- Integration guides for common credit platforms

### What We Won't Merge

- Changes that remove or weaken safety layers (SHAP filter, FCRA injection,
  protected attribute filter, uncertainty flagging, cross-model grading).
- LLMs making credit decisions directly (the core thesis is that they shouldn't).
- Proprietary data or API keys committed to the repo.
- Dependencies without clear licenses compatible with Apache 2.0.

## Development Setup

```bash
git clone https://github.com/maneek21/credit-risk-ai-workflow.git
cd credit-risk-ai-workflow
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Code of Conduct

Be respectful, constructive, and inclusive. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0.
