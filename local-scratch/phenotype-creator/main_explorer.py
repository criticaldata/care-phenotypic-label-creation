import os
import sys
from pathlib import Path

def main():
    """Main function to run all exploration scripts."""
    print("=" * 50)
    print("CarePhenotypeCreator Exploration Suite")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Check if we're in the right directory
    scripts = [
        'data_generator.py',
        'explore_preprocessing.py',
        'explore_clustering.py',
        'explore_clinical_factors.py',
        'explore_validation.py'
    ]
    
    # Make sure all scripts exist
    for script in scripts:
        if not (script_dir / script).exists():
            print(f"Error: {script} not found. Please make sure all exploration scripts are in {script_dir}")
            return
    
    # Run data generator first
    print("\nStep 1: Generating synthetic data")
    print("-" * 50)
    os.system(f"python {script_dir / 'data_generator.py'}")
    
    # Run other exploration scripts
    steps = [
        ("Exploring data preprocessing", "explore_preprocessing.py"),
        ("Exploring clustering", "explore_clustering.py"),
        ("Exploring clinical factors", "explore_clinical_factors.py"),
        ("Exploring validation metrics", "explore_validation.py")
    ]
    
    for i, (description, script) in enumerate(steps, 2):
        print(f"\nStep {i}: {description}")
        print("-" * 50)
        os.system(f"python {script_dir / script}")
    
    print("\n" + "=" * 50)
    print("Exploration complete! Check the generated visualizations.")
    print("=" * 50)

if __name__ == "__main__":
    main()