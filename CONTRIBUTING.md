# Contributing to Care Phenotype Analyzer

Thank you for your interest in contributing to the Care Phenotype Analyzer project! This guide will help you get started with contributing to the project.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- A GitHub account
- Basic understanding of healthcare data analysis

### Development Setup

1. Fork the repository:
   - Go to [https://github.com/MIT-LCP/care-phenotypic-label-creation](https://github.com/MIT-LCP/care-phenotypic-label-creation)
   - Click the "Fork" button in the top-right corner
   - Clone your fork locally:
     ```bash
     git clone https://github.com/YOUR_USERNAME/care-phenotypic-label-creation.git
     cd care-phenotypic-label-creation
     ```

2. Set up your development environment:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Development Workflow

### 1. Create a New Branch

Create a new branch for your feature or bug fix:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, well-documented code
- Follow the project's coding style (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

Before submitting your changes, run the test suite:
```bash
pytest
```

### 4. Commit Your Changes

Follow these commit message guidelines:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring
- test: Adding or modifying tests
- chore: Maintenance tasks

Example:
```
feat(analyzer): Add new pattern detection algorithm

- Implemented wavelet-based pattern detection
- Added support for multi-scale analysis
- Updated documentation with new features

Closes #123
```

### 5. Push Your Changes

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select the base branch (usually `main`)
4. Write a clear description of your changes
5. Link any related issues
6. Request review from maintainers

## Code Style Guidelines

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable and function names

Example:
```python
def analyze_patterns(
    data: pd.DataFrame,
    pattern_column: str,
    time_column: str
) -> Dict[str, Any]:
    """
    Analyze temporal patterns in care delivery.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing care pattern data
    pattern_column : str
        Column containing the pattern to analyze
    time_column : str
        Column containing temporal information

    Returns
    -------
    Dict[str, Any]
        Dictionary containing analysis results

    Raises
    ------
    ValueError
        If required columns are missing
    """
    # Implementation
```

### Documentation Style

- Use clear, concise language
- Include code examples where appropriate
- Keep documentation up to date with code changes
- Use proper markdown formatting

## Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Include both unit tests and integration tests
- Use meaningful test names
- Follow the test structure in existing tests

Example:
```python
def test_pattern_analysis_with_clinical_factors():
    """Test pattern analysis with clinical factor adjustment."""
    # Setup
    data = create_test_data()
    analyzer = CarePatternAnalyzer(data)
    
    # Execute
    results = analyzer.analyze_patterns(
        pattern_column='lab_test_frequency',
        clinical_factors=['sofa_score']
    )
    
    # Assert
    assert 'adjusted_frequency' in results
    assert 'unexplained_variation' in results
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analyzer.py

# Run with coverage report
pytest --cov=care_phenotype_analyzer
```

## Review Process

1. Your pull request will be reviewed by maintainers
2. Address any feedback and make requested changes
3. Once approved, your changes will be merged
4. The CI/CD pipeline will run tests and checks

## Getting Help

- Open an issue for bug reports or feature requests
- Join our community discussions
- Contact maintainers for questions

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a respectful and inclusive environment.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 