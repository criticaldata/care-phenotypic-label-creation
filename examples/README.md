# Care Phenotype Analyzer Examples

This directory contains example scripts demonstrating how to use the care phenotype analyzer package.

## Example Scripts

### lab_test_analysis_example.py

This example demonstrates:
- Creating care phenotype labels based on lab test measurement patterns
- Analyzing measurement frequencies and patterns
- Accounting for clinical factors (SOFA score, Charlson score)
- Identifying unexplained variations
- Visualizing patterns and results
- Performing fairness evaluation

To run the example:
```bash
python lab_test_analysis_example.py
```

The script will:
1. Generate sample data with clinical factors and care patterns
2. Create care phenotype labels
3. Analyze measurement frequencies
4. Generate visualizations (saved as 'lab_test_analysis.png')
5. Analyze unexplained variations
6. Perform fairness evaluation

## Output

The example script generates:
- Console output with analysis results
- Visualization plots saved as 'lab_test_analysis.png'

## Requirements

Make sure you have installed the required dependencies:
```bash
pip install -r ../requirements.txt
```

## Customization

You can modify the example script to:
- Use your own data instead of sample data
- Adjust the number of phenotypes
- Change the clinical factors considered
- Modify visualization settings
- Add additional analysis metrics 