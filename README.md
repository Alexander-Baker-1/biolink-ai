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
git clone https://github.com/[your-username]/biolink-ai.git
cd biolink-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install pandas numpy scikit-learn xgboost rdkit biopython gradio plotly requests

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

- pandas>=2.1.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- xgboost>=1.7.0
- rdkit
- biopython>=1.81
- gradio>=4.0.0
- plotly>=5.15.0
- requests>=2.28.0

## Performance

- Test Set Spearman Correlation: 0.678
- Test Set RMSE: 0.639 pIC50 units
- Training Data: 135 protein-ligand pairs from BindingDB
- Feature Space: 4111 dimensions (Morgan fingerprints + physicochemical + target encoding)
