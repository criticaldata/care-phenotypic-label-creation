"""
Tests for the lab test frequency analyzer module.
"""

import pytest
import pandas as pd
from lab_test_frequency_analyzer.analyzer import LabTestFrequencyAnalyzer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'condition': ['A', 'B', 'A', 'C'],
        'test': ['test1', 'test2', 'test1', 'test3'],
        'timestamp': pd.date_range(start='2023-01-01', periods=4)
    })

def test_analyzer_initialization(sample_data):
    """Test that the analyzer initializes correctly."""
    analyzer = LabTestFrequencyAnalyzer(sample_data)
    assert analyzer.data.equals(sample_data)

# More tests will be added as we implement the functionality 