# BioLink: Lightweight Protein-Ligand Binding Predictor

A specialized AI system for predicting protein-ligand binding affinities using efficient machine learning instead of massive computational resources.

## Project Overview

While AlphaFold revolutionized protein structure prediction, it requires enormous computational resources. BioLink tackles a focused subproblem: predicting how strongly small molecule drugs bind to protein targets using lightweight models that run on standard hardware.

## Features

- XGBoost-based binding affinity prediction with hyperparameter optimization
- Interactive 3D protein visualization using Plotly and PDB structures  
- 2D molecular structure rendering from SMILES strings
- Real-time web interface built with Gradio
- Scaffold-based data splitting to prevent overfitting

## Installation

```bash
# Clone the repository
git clone https://github.com/Alexander-Baker-1/biolink-ai.git
cd biolink-ai

# Install dependencies
pip install -r requirements.txt

# Run the application
python biolink.py
```

## Usage

### Training a New Model
```python
from biolink import BioLinkApp

app = BioLinkApp()
app.setup_data()
app.predictor.tune_hyperparameters(app.df)
```

### Making Predictions
```python
pic50, confidence = app.predictor.predict_with_confidence(
    smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    pdb_id="3N8Y"  # COX-1 protein
)
print(f"Predicted pIC50: {pic50:.2f} (Confidence: {confidence:.2f})")
```

### Web Interface
```bash
python biolink.py
```
Then open the provided URL to use the interactive web interface.

## Dependencies

- pandas==2.0.3
- numpy<2.0
- scikit-learn>=1.3.0
- xgboost>=1.7.0
- rdkit-pypi==2022.9.5
- biopython>=1.81
- gradio>=3.0.0
- plotly>=5.0.0
- requests>=2.25.0
