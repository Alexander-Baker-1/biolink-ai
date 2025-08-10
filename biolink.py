#!/usr/bin/env python3
"""
BioLink: Lightweight Protein-Ligand Binding Predictor
Professional version with 3D protein viewer and machine learning predictor
"""

import warnings
warnings.filterwarnings('ignore')

import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import io
import zipfile
import gzip
import requests
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from Bio.PDB import PDBParser, MMCIFParser, DSSP, is_aa, PDBList
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("BioLink: Ligand-Protein Binding Predictor")
print("=" * 50)

# =========================
# UTILITY FUNCTIONS
# =========================
def smiles_to_svg_html(smiles, size=(320, 320)):
    """Create 2D ligand structure visualization from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return "<div style='color:#b00'>Invalid SMILES</div>"
        
        mol = Chem.RemoveHs(mol)
        rdDepictor.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.bondLineWidth = 2
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        svg = drawer.GetDrawingText()
        if "xmlns" not in svg:
            svg = svg.replace("<svg ", "<svg xmlns='http://www.w3.org/2000/svg' ", 1)

        return f"<div style='width:{size[0]}px;height:{size[1]}px'>{svg}</div>"
    except Exception as e:
        return f"<div style='color:#b00'>2D render failed: {e}</div>"

def create_protein_3d_figure(pdb_id, local_path):
    """Create 3D protein visualization using Plotly"""
    pdb_id = (pdb_id or "").strip().upper()

    if not local_path or not Path(local_path).exists():
        fig = go.Figure()
        fig.update_layout(
            title=f"{pdb_id} - Structure not available",
            annotations=[{
                'text': f"No structure file for {pdb_id}",
                'x': 0.5, 'y': 0.5, 'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 14}
            }],
            width=500, height=400
        )
        return fig

    try:
        # Parse atoms from PDB file
        atoms = []
        ca_atoms = []

        with open(local_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    try:
                        atom_name = line[12:16].strip()
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        element = atom_name[0].upper()

                        atoms.append({'x': x, 'y': y, 'z': z, 'element': element, 'name': atom_name})

                        if atom_name == 'CA':
                            ca_atoms.append({'x': x, 'y': y, 'z': z})
                    except:
                        continue

        if not atoms:
            fig = go.Figure()
            fig.update_layout(title=f"No atoms found in {pdb_id}")
            return fig

        fig = go.Figure()

        # Add backbone trace
        if len(ca_atoms) > 1:
            fig.add_trace(go.Scatter3d(
                x=[a['x'] for a in ca_atoms],
                y=[a['y'] for a in ca_atoms],
                z=[a['z'] for a in ca_atoms],
                mode='lines',
                line=dict(color='blue', width=6),
                name='Backbone'
            ))

        # Sample atoms if too many for performance
        max_atoms = 2000
        if len(atoms) > max_atoms:
            step = len(atoms) // max_atoms
            atoms = atoms[::step]

        # Group atoms by element for coloring
        element_colors = {'C': '#808080', 'N': '#0000FF', 'O': '#FF0000', 'S': '#FFFF00', 'P': '#FFA500'}
        by_element = {}
        for atom in atoms:
            element = atom['element']
            if element not in by_element:
                by_element[element] = []
            by_element[element].append(atom)

        # Add atoms by element
        for element, element_atoms in by_element.items():
            if element == 'H':  # Skip hydrogen atoms
                continue
            color = element_colors.get(element, '#FF1493')
            size = 2 if element == 'C' else 3

            fig.add_trace(go.Scatter3d(
                x=[a['x'] for a in element_atoms],
                y=[a['y'] for a in element_atoms],
                z=[a['z'] for a in element_atoms],
                mode='markers',
                marker=dict(color=color, size=size, opacity=0.7),
                name=f'{element} ({len(element_atoms)})'
            ))

        # Style the figure
        fig.update_layout(
            title=f"{pdb_id} - 3D Structure ({len(atoms)} atoms)",
            scene=dict(
                xaxis_title='X (Å)', 
                yaxis_title='Y (Å)', 
                zaxis_title='Z (Å)',
                bgcolor='white', 
                aspectmode='data'
            ),
            width=500, height=400, 
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True, 
            legend=dict(x=0.02, y=0.98)
        )

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error: {str(e)[:40]}")
        return fig

def make_feature_importance_plot(predictor, top_n=20):
    """Plot top feature importances"""
    model = getattr(predictor, "model", None)
    fi = getattr(model, "feature_importances_", None)
    if model is None or fi is None:
        return go.Figure()

    fi = np.asarray(fi, dtype=float).ravel()
    n_total = int(fi.shape[0])

    cfg = getattr(predictor, "feat_cfg", {})
    fp_bits = int(cfg.get("fp_bits", n_total))
    use_physchem = bool(cfg.get("use_physchem", True))
    physchem_names = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB", "Rings", "AromRings"]
    target_keys = list(getattr(predictor, "target_keys", []) or [])

    # Build feature names
    names = []
    
    # Fingerprint features
    fp_take = min(fp_bits, n_total)
    names.extend([f"FP[{i}]" for i in range(fp_take)])
    remaining = n_total - len(names)

    # Physicochemical features
    if use_physchem and remaining > 0:
        pc_take = min(len(physchem_names), remaining)
        names.extend(physchem_names[:pc_take])
    remaining = n_total - len(names)

    # Target one-hot features
    if remaining > 0:
        if target_keys:
            tk_take = min(len(target_keys), remaining)
            names.extend([f"target:{target_keys[i]}" for i in range(tk_take)])
            remaining = n_total - len(names)
        if remaining > 0:
            start = 0 if not target_keys else len(target_keys)
            names.extend([f"target_[{start+i}]" for i in range(remaining)])

    # Ensure names length matches feature vector length
    if len(names) != n_total:
        names = (names + [f"feat_{i}" for i in range(n_total - len(names))])[:n_total]

    k = max(1, min(top_n, n_total))
    idx = np.argsort(fi)[-k:][::-1]

    y_labels = [names[i] if i < len(names) else f"feat_{int(i)}" for i in idx]
    x_vals = fi[idx]

    fig = go.Figure(go.Bar(x=x_vals, y=y_labels, orientation="h"))
    fig.update_layout(
        title=f"Top {len(idx)} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=420,
        margin=dict(l=140, r=20, t=40, b=40),
    )
    return fig

# =========================
# DATA DOWNLOADER
# =========================
class DataDownloader:
    """Handles downloading and managing protein and binding data"""
    
    def __init__(self):
        self.data_dir = Path("mini_alphafold_data")
        self.data_dir.mkdir(exist_ok=True)
        self.pdb_dir = self.data_dir / "pdb"
        self.pdb_dir.mkdir(exist_ok=True)
        self.pdbl = PDBList()
        self._pdb_cache = {}

    def download_bindingdb_sample(self, force_refresh=False):
        """Download or use cached BindingDB sample data"""
        import shutil

        frozen = self.data_dir / "bindingdb_frozen.csv"
        latest = self.data_dir / "sample_bindingdb.csv"

        print("Downloading BindingDB sample data...")

        if (not force_refresh) and frozen.exists() and frozen.stat().st_size > 0:
            shutil.copy2(frozen, latest)
            print(f"Using cached dataset: {frozen.name}")
            return latest

        # Create fresh dataset
        out = self.create_bindingdb(per_target_limit=25)
        try:
            shutil.copy2(out, frozen)
            print(f"Dataset cached: {frozen.name}")
        except Exception as e:
            print(f"Warning: Could not cache dataset: {e}")
        return out

    def _sha256(self, path):
        """Calculate SHA256 hash of file"""
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""):
                h.update(chunk)
        return h.hexdigest()

    def create_bindingdb(self, per_target_limit=25):
        """Create BindingDB dataset by streaming and filtering data"""
        import os
        import csv
        import re
        from datetime import datetime
        import pandas as pd
        from rdkit.Chem import rdMolDescriptors

        out = self.data_dir / "sample_bindingdb.csv"

        # Target proteins and their PDB IDs
        targets = [
            "COX-1", "Adenosine receptor", "COX-2", "CYP", "Alcohol dehydrogenase",
            "Beta-2 adrenergic receptor", "Neuraminidase", "Androgen receptor",
            "PPAR gamma", "Fatty acid binding protein", "Histamine H1 receptor",
            "Retinoic acid receptor", "HERG", "Lipase", "Dihydrofolate reductase",
            "Sphingomyelinase", "Peptidase", "Adenosine deaminase"
        ]
        
        pdb_map = {
            "COX-1": "3N8Y", "Adenosine receptor": "1H9M", "COX-2": "4PH9", "CYP": "4YOP",
            "Alcohol dehydrogenase": "4YOP", "Beta-2 adrenergic receptor": "2R9W",
            "Neuraminidase": "1BNA", "Androgen receptor": "1I37", "PPAR gamma": "2PRG",
            "Fatty acid binding protein": "1ICE", "Histamine H1 receptor": "2F01",
            "Retinoic acid receptor": "5L7D", "HERG": "2YCW", "Lipase": "3P0G",
            "Dihydrofolate reductase": "4QI7", "Sphingomyelinase": "1BV7",
            "Peptidase": "3HLL", "Adenosine deaminase": "2E5L",
        }

        # Build regex pattern for target matching
        target_pattern = re.compile("|".join(re.escape(t) for t in targets), re.I)

        # Try to download recent BindingDB data
        def month_tags(n=3):
            now = datetime.utcnow()
            tags = []
            y, m = now.year, now.month
            for i in range(n):
                yy = y if m - i >= 1 else y - 1
                mm = m - i if m - i >= 1 else m - i + 12
                tags.append(f"{yy}{mm:02d}")
            return tags

        urls = [f"https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_{t}_tsv.zip"
                for t in month_tags(3)]

        # Download data
        zbytes = None
        for url in urls:
            try:
                print(f"Trying BindingDB URL: {url}")
                r = requests.get(url, timeout=60, allow_redirects=True)
                r.raise_for_status()
                if r.content.startswith(b"PK"):
                    zbytes = r.content
                    print("Successfully downloaded BindingDB data")
                    break
                else:
                    print("Invalid ZIP file, trying next URL...")
            except Exception as e:
                print(f"Download failed: {e}")
                continue

        if zbytes is None:
            raise RuntimeError(f"Could not download BindingDB data from any URL")

        # Process the downloaded data
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            tsv_name = next(n for n in zf.namelist() if n.lower().endswith(".tsv"))
            with zf.open(tsv_name) as tf:
                import io as _io
                txt = _io.TextIOWrapper(tf, encoding="utf-8", newline="")
                reader = csv.reader(txt, delimiter="\t")

                # Find required column indices
                header = next(reader)
                def find_column_index(*candidates):
                    norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): i for i, c in enumerate(header)}
                    for c in candidates:
                        k = re.sub(r"[^a-z0-9]+", "", c.lower())
                        if k in norm:
                            return norm[k]
                    return None

                i_smiles = find_column_index("Ligand SMILES", "SMILES", "Canonical SMILES")
                i_tname = find_column_index("Target Name", "Protein Name", "Target")
                i_lname = find_column_index("Ligand Name", "Ligand", "Compound Name")
                i_ic50 = find_column_index("IC50 (nM)", "IC50 (nM, converted)", "IC50 (nM,converted)")

                if i_smiles is None or i_tname is None or i_ic50 is None:
                    raise RuntimeError("Required columns not found in BindingDB TSV.")

                # Filter and save data
                with open(out, "w", newline="", encoding="utf-8") as fout:
                    writer = csv.writer(fout)
                    writer.writerow(["Ligand SMILES", "PDB ID", "IC50 (nM)", "Protein Name", "Ligand Name"])

                    keep_count = {t: 0 for t in targets}
                    line_no = 0
                    scan_cap = 200_000

                    for row in reader:
                        line_no += 1
                        if line_no % 50_000 == 0:
                            print(f"Processed {line_no:,} lines, kept {sum(keep_count.values())} records")

                        if line_no > scan_cap or all(keep_count[t] >= per_target_limit for t in targets):
                            break

                        try:
                            smiles = row[i_smiles].strip()
                            tname = row[i_tname].strip()
                            ic50 = row[i_ic50].strip()
                            lname = row[i_lname].strip() if i_lname is not None and row[i_lname] else ""
                        except:
                            continue

                        if not smiles or not ic50 or not tname:
                            continue
                        if not target_pattern.search(tname):
                            continue

                        match_key = next((t for t in targets if t.lower() in tname.lower()), None)
                        if match_key is None or keep_count[match_key] >= per_target_limit:
                            continue

                        try:
                            ic50_val = float(ic50)
                        except:
                            continue

                        # Find corresponding PDB ID
                        pdb_id = None
                        for k, v in pdb_map.items():
                            if k.lower() in tname.lower():
                                pdb_id = v
                                break

                        writer.writerow([smiles, pdb_id or "", ic50_val, tname, lname])
                        keep_count[match_key] += 1

        # Clean and format the final dataset
        df = pd.read_csv(
            out,
            dtype={"Ligand SMILES": "string", "PDB ID": "string", "Protein Name": "string", "Ligand Name": "string"}
        )
        df["IC50 (nM)"] = pd.to_numeric(df["IC50 (nM)"], errors="coerce")
        df = df.dropna(subset=["Ligand SMILES", "IC50 (nM)"]).copy()

        # Clean text columns
        for col in ["Ligand SMILES", "Ligand Name", "PDB ID", "Protein Name"]:
            df[col] = df[col].astype("string").fillna("")

        # Generate pretty ligand names using molecular formulas
        def generate_ligand_name(row):
            name = str(row["Ligand Name"])
            name = "" if name in ("<NA>", "NA", "NaN", "nan") else name.strip()
            if name:
                return name
            smi = str(row["Ligand SMILES"]).strip()
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    return rdMolDescriptors.CalcMolFormula(mol)
            except:
                pass
            return smi[:24] if smi else "Ligand"

        df["Ligand Name"] = df.apply(generate_ligand_name, axis=1)

        # Final cleanup
        for col in ["Ligand SMILES", "PDB ID", "Protein Name", "Ligand Name"]:
            df[col] = df[col].astype("string").fillna("").str.strip().replace({"nan": "", "NA": "", "<NA>": ""})

        df.to_csv(out, index=False)
        print(f"Created BindingDB dataset: {out} (≤{per_target_limit} per target)")
        return out

    def ensure_pdb_file(self, pdb_id):
        """Download PDB file if not already cached"""
        pdb_id = str(pdb_id).strip().lower()

        # Check cache first
        cached = self._pdb_cache.get(pdb_id)
        if cached and Path(cached).exists() and Path(cached).stat().st_size > 0:
            return cached

        out_pdb = self.pdb_dir / f"{pdb_id}.pdb"

        # Check if file already exists
        if out_pdb.exists() and out_pdb.stat().st_size > 0:
            self._pdb_cache[pdb_id] = str(out_pdb)
            return str(out_pdb)

        # Download from PDB
        try:
            print(f"Downloading PDB structure: {pdb_id.upper()}")
            fn = self.pdbl.retrieve_pdb_file(pdb_id, pdir=str(self.pdb_dir), file_format='pdb')
            src = Path(fn)
            if src.exists():
                if src.suffix == ".gz":
                    with gzip.open(src, "rb") as f_in, open(out_pdb, "wb") as f_out:
                        f_out.write(f_in.read())
                    src.unlink()
                else:
                    src.rename(out_pdb)

                if out_pdb.exists() and out_pdb.stat().st_size > 0:
                    print(f"PDB {pdb_id.upper()} downloaded successfully")
                    self._pdb_cache[pdb_id] = str(out_pdb)
                    return str(out_pdb)
        except Exception as e:
            print(f"Warning: PDB download failed for {pdb_id.upper()}: {e}")

        return None

  # =========================
# FEATURE EXTRACTOR
# =========================
class MolecularFeatureExtractor:
    """Extract molecular features from SMILES and protein structures"""
    
    def __init__(self):
        self.scaler = StandardScaler()

    def extract_ligand_features(self, smiles):
        """Extract molecular descriptors from SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)

            features = {
                'mol_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'h_donors': Descriptors.NumHDonors(mol),
                'h_acceptors': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'polar_surface_area': Descriptors.TPSA(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
            }

            # Add molecular fingerprint
            try:
                base = Chem.RemoveHs(mol)
                fp = AllChem.GetMorganFingerprintAsBitVect(base, radius=2, nBits=256)
                for i in range(256):
                    features[f'ecfp2_{i}'] = int(fp[i])
            except:
                pass

            return features
        except Exception as e:
            print(f"Ligand feature extraction error: {e}")
            return None

    def extract_protein_features(self, pdb_path):
        """Extract protein features from PDB file"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_path)

            features = {
                'num_residues': 0, 'num_atoms': 0,
                'hydrophobic_residues': 0, 'polar_residues': 0,
                'charged_residues': 0, 'molecular_weight': 0.0
            }

            hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'TYR', 'PRO', 'MET'}
            polar = {'SER', 'THR', 'ASN', 'GLN', 'CYS'}
            charged = {'ARG', 'LYS', 'HIS', 'ASP', 'GLU'}

            for model in structure:
                for chain in model:
                    for residue in chain:
                        if is_aa(residue.get_resname(), standard=True):
                            features['num_residues'] += 1
                            resname = residue.get_resname()
                            if resname in hydrophobic:
                                features['hydrophobic_residues'] += 1
                            if resname in polar:
                                features['polar_residues'] += 1
                            if resname in charged:
                                features['charged_residues'] += 1
                            features['num_atoms'] += sum(1 for _ in residue.get_atoms())
                            features['molecular_weight'] += 110  # Average amino acid weight

            return features
        except:
            return self.get_default_protein_features()

    def get_default_protein_features(self):
        """Return default protein features when structure is unavailable"""
        return {
            'num_residues': 350, 'num_atoms': 2800,
            'hydrophobic_residues': 120, 'polar_residues': 80,
            'charged_residues': 70, 'molecular_weight': 38000
        }

  # =========================
# BINDING PREDICTOR
# =========================
class LightweightBindingPredictor:
    """Machine learning model for predicting protein-ligand binding affinity"""
    
    def __init__(self, downloader=None):
        self.downloader = downloader
        self.model = None
        self.scaler = StandardScaler()
        
        # Configuration defaults
        self.feat_cfg = dict(
            fp_radius=2, 
            fp_bits=4096, 
            fp_chiral=True, 
            use_physchem=True
        )
        self.target_keys = []
        self.target_index = {}
        self.use_delta = False
        self.target_mean = {}

    def prepare_features(self, df, fp_radius=2, fp_bits=2048, fp_chiral=True, 
                        use_physchem=True, use_target_onehot=True):
        """
        Build feature matrix from dataframe with molecular fingerprints,
        physicochemical properties, and target one-hot encoding
        """
        from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski
        from rdkit import DataStructs

        X_fp = []
        X_pc = [] if use_physchem else None
        want_onehot = bool(use_target_onehot and len(getattr(self, "target_keys", [])) > 0)
        X_t = [] if want_onehot else None
        y = []

        n_targets = len(getattr(self, "target_keys", [])) if use_target_onehot else 0

        for _, row in df.iterrows():
            smiles = str(row.get("Ligand SMILES", "")).strip()
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Target variable (pIC50)
            pic50 = float(row.get("pIC50", 0.0))

            # Morgan fingerprint
            bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, fp_radius, nBits=fp_bits, useChirality=fp_chiral
            )
            arr = np.zeros((fp_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
            X_fp.append(arr)

            # Physicochemical properties
            if use_physchem:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                rot = Lipinski.NumRotatableBonds(mol)
                rings = rdMolDescriptors.CalcNumRings(mol)
                arom = rdMolDescriptors.CalcNumAromaticRings(mol)
                X_pc.append([mw, logp, tpsa, hbd, hba, rot, rings, arom])

            # Target one-hot encoding
            if want_onehot and n_targets > 0:
                key = str(row.get("PDB ID", "")).strip().upper()
                if not key:
                    key = str(row.get("Protein Name", "")).strip().lower()
                vec = np.zeros((n_targets,), dtype=np.float32)
                j = getattr(self, "target_index", {}).get(key, None)
                if j is not None:
                    j = int(j)
                    if 0 <= j < n_targets:
                        vec[j] = 1.0
                X_t.append(vec)

            y.append(pic50)

        # Validate data
        if len(X_fp) == 0:
            raise ValueError("No valid molecules found for feature extraction")

        # Combine feature blocks
        X_fp = np.asarray(X_fp, dtype=np.float32)
        blocks = [X_fp]

        if use_physchem:
            X_pc = np.asarray(X_pc, dtype=np.float32)
            if X_pc.shape[0] != X_fp.shape[0]:
                n = min(X_pc.shape[0], X_fp.shape[0], len(y))
                X_fp = X_fp[:n]
                X_pc = X_pc[:n]
                y = y[:n]
                if want_onehot:
                    X_t = X_t[:n]
            blocks.append(X_pc)

        if want_onehot and n_targets > 0:
            X_t = np.asarray(X_t, dtype=np.float32)
            if X_t.shape[0] != X_fp.shape[0]:
                n = min(X_t.shape[0], X_fp.shape[0], len(y))
                X_fp = X_fp[:n]
                y = y[:n]
                if use_physchem:
                    blocks[1] = blocks[1][:n]
                X_t = X_t[:n]
            blocks.append(X_t)

        X = np.hstack(blocks)
        y = np.asarray(y, dtype=np.float32)

        # Log feature information
        pc_dim = 0
        if use_physchem and X_pc is not None and X_pc.size:
            pc_dim = int(X_pc.shape[1])
        additional_info = (f" + physchem({pc_dim})" if use_physchem else "") + \
                         (f" + target_onehot({n_targets})" if want_onehot and n_targets > 0 else "")

        print(f"Extracted features: {len(y)} samples × {X.shape[1]} dimensions "
              f"(radius={fp_radius}, bits={fp_bits}, chiral={fp_chiral}{additional_info})")
        return X, y

    def train(self, df):
        """Train the binding prediction model using scaffold splits"""
        print("Training binding predictor...")

        # Split data by scaffold groups
        if "split" in df.columns:
            df_train = df[df["split"] == "train"].copy()
            df_val = df[df["split"] == "val"].copy()
            df_test = df[df["split"] == "test"].copy()
        else:
            # Fallback to random split
            df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
            df_test = df_val.iloc[0:0].copy()

        # Extract features and labels
        cfg = getattr(self, "feat_cfg", dict(fp_radius=2, fp_bits=4096, fp_chiral=True, use_physchem=True))
        X_train, y_train = self.prepare_features(df_train, **cfg)
        X_val, y_val = self.prepare_features(df_val, **cfg)

        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        y_pred_val = self.model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)
        
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(y_val, y_pred_val)
        except:
            rho = float("nan")

        print(f"Validation metrics - RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f} | Spearman: {rho:.3f}")

        # Retrain on combined training and validation data
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        X_combined_scaled = self.scaler.fit_transform(X_combined)
        self.model.fit(X_combined_scaled, y_combined)
        print("Model retrained on combined train+validation data")

        # Store validation metrics
        self._val_metrics = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": rho}
        self._sizes = {"train": len(df_train), "val": len(df_val), "test": len(df_test)}

        return self.model

    def predict_with_confidence(self, smiles, pdb_id=None):
        """Predict binding affinity with confidence estimate"""
        if smiles is None or str(smiles).strip() == "":
            raise ValueError("Ligand SMILES is empty. Please enter a valid SMILES string.")

        key, name = self._parse_target_key(pdb_id) if pdb_id else ("", "")

        cfg = getattr(self, "feat_cfg", dict(fp_radius=2, fp_bits=4096, fp_chiral=True, use_physchem=True))
        tmp_df = pd.DataFrame([{
            "Ligand SMILES": str(smiles),
            "PDB ID": key,
            "Protein Name": name,
            "pIC50": 0.0
        }])

        # Extract features
        X, _ = self.prepare_features(tmp_df, **cfg, use_target_onehot=True)

        # Ensure feature dimensions match training
        expected_features = getattr(self.scaler, "n_features_in_", X.shape[1])
        if X.shape[1] != expected_features:
            if X.shape[1] < expected_features:
                X = np.hstack([X, np.zeros((X.shape[0], expected_features - X.shape[1]), dtype=np.float32)])
            else:
                X = X[:, :expected_features]

        X_scaled = self.scaler.transform(X)
        prediction_delta = float(self.model.predict(X_scaled)[0])

        # Apply baseline correction if using delta predictions
        baseline = 0.0
        if getattr(self, "use_delta", False):
            baseline = float(getattr(self, "target_mean", {}).get(key or name, 0.0))

        prediction = prediction_delta + baseline
        
        # Calculate confidence based on validation RMSE
        rmse = (getattr(self, "_val_metrics", {}) or {}).get("rmse", None)
        confidence = float(max(0.0, 1.0 - (rmse or 1.0) / 2.0))
        
        return prediction, confidence

    def get_feature_importance(self, top_n=20):
        """Get feature importance from trained model"""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return []
        
        importances = np.asarray(self.model.feature_importances_, dtype=float)
        if importances.ndim != 1 or importances.size == 0:
            return []
        
        indices = np.argsort(importances)[::-1][:top_n]
        feature_names = getattr(self, "feature_names", None)
        
        result = []
        for i in indices:
            if feature_names and i < len(feature_names):
                fname = feature_names[i]
            else:
                fname = f"feature_{int(i)}"
            result.append({"feature": fname, "importance": float(importances[i])})
        
        return result

    def tune_hyperparameters(self, df):
        """Hyperparameter tuning using XGBoost with grid search"""
        import time
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Safe Spearman correlation calculation
        try:
            from scipy.stats import spearmanr
            def spearman_safe(y_true, y_pred):
                return float(spearmanr(y_true, y_pred).statistic)
        except:
            def spearman_safe(y_true, y_pred):
                y_true_arr = np.asarray(y_true)
                y_pred_arr = np.asarray(y_pred)
                rank_true = np.argsort(np.argsort(y_true_arr))
                rank_pred = np.argsort(np.argsort(y_pred_arr))
                return float(np.corrcoef(rank_true, rank_pred)[0, 1])

        # Install XGBoost if needed
        try:
            import xgboost as xgb
        except:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
            import xgboost as xgb

        # Prepare data splits
        df_train = df[df["split"] == "train"].copy()
        df_val = df[df["split"] == "val"].copy()
        if df_train.empty or df_val.empty:
            raise RuntimeError("Empty train/val split. Run scaffold splits before tuning.")

        # Set up target vocabulary for one-hot encoding
        if hasattr(self, "_fit_target_vocab"):
            self._fit_target_vocab(df_train)
        if hasattr(self, "_fit_target_stats"):
            self._fit_target_stats(df_train)
            self.use_delta = False

        # Define hyperparameter grids
        feature_configs = [
            dict(fp_radius=2, fp_bits=2048, fp_chiral=True, use_physchem=True),
            dict(fp_radius=2, fp_bits=4096, fp_chiral=True, use_physchem=True),
            dict(fp_radius=3, fp_bits=2048, fp_chiral=True, use_physchem=True),
        ]
        
        model_configs = [
            dict(n_estimators=1500, learning_rate=0.03, max_depth=5,
                subsample=0.7, colsample_bytree=0.6,
                reg_lambda=2.0, min_child_weight=3, gamma=0.1),
            dict(n_estimators=1200, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.7,
                reg_lambda=1.5, min_child_weight=2, gamma=0.0),
        ]

        leaderboard = []
        best_config = None
        success_count = 0
        base_params = dict(objective="reg:squarederror", random_state=42, n_jobs=-1, tree_method="hist")

        # Grid search
        for feat_cfg in feature_configs:
            # Extract features for current configuration
            X_train, y_train_raw = self.prepare_features(df_train, **feat_cfg)
            X_val, y_val_raw = self.prepare_features(df_val, **feat_cfg)

            # Apply delta labels if configured
            if getattr(self, "use_delta", False):
                mean_train = np.array([self.target_mean.get(self._key_for_row(r), 0.0) 
                                     for _, r in df_train.iterrows()], dtype=float)
                mean_val = np.array([self.target_mean.get(self._key_for_row(r), 0.0) 
                                   for _, r in df_val.iterrows()], dtype=float)
                y_train, y_val = y_train_raw - mean_train, y_val_raw - mean_val
            else:
                y_train, y_val = y_train_raw, y_val_raw

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Check for invalid labels
            if not (np.isfinite(y_train).all() and np.isfinite(y_val).all()):
                print("Skipping configuration due to invalid labels")
                continue

            for model_cfg in model_configs:
                start_time = time.time()
                try:
                    # Configure XGBoost model
                    import inspect
                    fit_signature = inspect.signature(xgb.XGBRegressor.fit)
                    supports_eval_metric = "eval_metric" in fit_signature.parameters
                    supports_early_stopping = "early_stopping_rounds" in fit_signature.parameters

                    model = xgb.XGBRegressor(**base_params, **model_cfg)
                    fit_kwargs = dict(eval_set=[(X_val_scaled, y_val)], verbose=False)
                    
                    if supports_eval_metric:
                        fit_kwargs["eval_metric"] = "rmse"
                    else:
                        model = xgb.XGBRegressor(**base_params, **model_cfg, eval_metric="rmse")
                    
                    if supports_early_stopping:
                        fit_kwargs["early_stopping_rounds"] = 50

                    try:
                        model.fit(X_train_scaled, y_train, **fit_kwargs)
                        best_iteration = getattr(model, "best_iteration", None)
                        best_iter = int(best_iteration if best_iteration is not None else model_cfg.get("n_estimators", 1000))
                    except TypeError as e:
                        print(f"Warning: Early stopping disabled for this configuration: {e}")
                        model = xgb.XGBRegressor(**base_params, **model_cfg)
                        model.fit(X_train_scaled, y_train)
                        best_iter = model_cfg.get("n_estimators", 1000)

                    # Evaluate model
                    y_pred = model.predict(X_val_scaled)
                    mse = mean_squared_error(y_val, y_pred)
                    rmse = float(np.sqrt(mse))
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    spearman = spearman_safe(y_val, y_pred)
                    training_time = time.time() - start_time

                    config_result = {
                        "rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman,
                        "feat": feat_cfg, "model": {"backend": "xgb", **model_cfg},
                        "n_features": int(X_train.shape[1]), "train_time": round(training_time, 2),
                        "best_iteration": int(best_iter),
                    }
                    leaderboard.append(config_result)
                    success_count += 1

                    # Update best configuration
                    if (best_config is None or 
                        (rmse < best_config["rmse"] - 1e-6) or 
                        (abs(rmse - best_config["rmse"]) <= 1e-6 and spearman > best_config["spearman"])):
                        best_config = config_result

                    print(f"Config: RMSE={rmse:.3f}, Spearman={spearman:.3f}, Features={config_result['n_features']}, Time={training_time:.2f}s")

                except Exception as e:
                    print(f"Skipping configuration due to error: {e}")

        if success_count == 0 or best_config is None:
            raise RuntimeError("Hyperparameter tuning failed. Check data quality and XGBoost installation.")

        print(f"\nBest configuration: RMSE={best_config['rmse']:.3f}, Spearman={best_config['spearman']:.3f}")
        print(f"Features: {best_config['feat']}")
        print(f"Model: {best_config['model']}")

        # Store best configuration
        self.feat_cfg = best_config["feat"]

        # Retrain best model on combined train+validation data
        X_train, y_train_raw = self.prepare_features(df_train, **self.feat_cfg)
        X_val, y_val_raw = self.prepare_features(df_val, **self.feat_cfg)
        
        if getattr(self, "use_delta", False):
            mean_train = np.array([self.target_mean.get(self._key_for_row(r), 0.0) 
                                 for _, r in df_train.iterrows()], dtype=float)
            mean_val = np.array([self.target_mean.get(self._key_for_row(r), 0.0) 
                               for _, r in df_val.iterrows()], dtype=float)
            y_combined = np.concatenate([y_train_raw - mean_train, y_val_raw - mean_val])
        else:
            y_combined = np.concatenate([y_train_raw, y_val_raw])

        X_combined = np.vstack([X_train, X_val])
        X_combined_scaled = self.scaler.fit_transform(X_combined)

        # Final model training
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            **{k: v for k, v in best_config["model"].items() if k != "backend"}
        ).set_params(n_estimators=best_config.get("best_iteration", best_config["model"].get("n_estimators", 1000)))
        
        self.model.fit(X_combined_scaled, y_combined)

        print("Model retrained on combined train+validation data")
        leaderboard.sort(key=lambda d: (d["rmse"], -d["spearman"]))
        self._tune_leaderboard = leaderboard
        return leaderboard

    def evaluate_test_set(self, df, verbose=True):
        """Evaluate model performance on test set"""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        if getattr(self, "model", None) is None or getattr(self, "scaler", None) is None:
            raise RuntimeError("Model not trained. Call train() or tune_hyperparameters() first.")
        
        if "split" not in df.columns:
            raise ValueError("No 'split' column found. Run scaffold splits first.")

        df_test = df[df["split"] == "test"].copy()
        if df_test.empty:
            if verbose:
                print("Warning: Test set is empty")
            return {}

        # Safe Spearman correlation
        try:
            from scipy.stats import spearmanr
            def spearman_safe(y_true, y_pred):
                return float(spearmanr(y_true, y_pred).statistic)
        except:
            def spearman_safe(y_true, y_pred):
                y_true_arr = np.asarray(y_true)
                y_pred_arr = np.asarray(y_pred)
                rank_true = np.argsort(np.argsort(y_true_arr))
                rank_pred = np.argsort(np.argsort(y_pred_arr))
                return float(np.corrcoef(rank_true, rank_pred)[0, 1])

        # Use same feature configuration as training
        if not hasattr(self, "feat_cfg"):
            raise RuntimeError("Feature configuration missing. Train model first.")

        cfg = getattr(self, "feat_cfg", dict(fp_radius=2, fp_bits=4096, fp_chiral=True, use_physchem=True))
        X_test, y_test = self.prepare_features(df_test, **cfg)

        # Verify feature dimensions
        expected_features = getattr(self.scaler, "n_features_in_", X_test.shape[1])
        if X_test.shape[1] != expected_features:
            raise ValueError(
                f"Feature dimension mismatch: {X_test.shape[1]} vs expected {expected_features}. "
                f"Current config: {cfg}"
            )

        X_test_scaled = self.scaler.transform(X_test)
        y_pred_delta = self.model.predict(X_test_scaled)
        
        # Apply baseline correction if using delta predictions
        mean_test = np.array([self.target_mean.get(self._key_for_row(r), 0.0) 
                             for _, r in df_test.iterrows()], dtype=float) \
                   if getattr(self, "use_delta", False) else 0.0
        y_pred = y_pred_delta + mean_test

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        spearman = spearman_safe(y_test, y_pred)

        if verbose:
            print(f"Test set evaluation: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}, Spearman={spearman:.3f} (n={len(df_test)})")
        
        return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman, "n": int(len(df_test))}

    def save_model_artifacts(self, app, test_metrics=None):
        """Save model configuration and training artifacts"""
        import json
        import time
        import hashlib
        from pathlib import Path

        data_dir = Path(getattr(getattr(app, "downloader", None), "data_dir", "mini_alphafold_data"))
        data_dir.mkdir(exist_ok=True)

        def file_hash(path):
            if not path.exists():
                return "N/A"
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1<<20), b""):
                    h.update(chunk)
            return h.hexdigest()

        # Save paths
        config_path = data_dir / "model_config.json"
        leaderboard_path = data_dir / "tune_leaderboard.json"
        metadata_path = data_dir / "run_metadata.json"

        # Save feature configuration
        feat_cfg = getattr(self, "feat_cfg", {})
        with open(config_path, "w") as f:
            json.dump(feat_cfg, f, indent=2)

        # Save tuning leaderboard
        leaderboard = getattr(self, "_tune_leaderboard", [])
        with open(leaderboard_path, "w") as f:
            json.dump(leaderboard, f, indent=2)

        # Save metadata
        frozen_csv = data_dir / "bindingdb_frozen.csv"
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_file": str(frozen_csv.resolve()) if frozen_csv.exists() else "N/A",
            "dataset_hash": file_hash(frozen_csv)[:12] if frozen_csv.exists() else "N/A",
            "split_file": str((data_dir / "scaffold_splits_seed42.json").resolve()),
            "split_seed": 42,
            "feature_config": feat_cfg,
            "test_metrics": test_metrics or {},
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("Model artifacts saved:")
        print(f"  Config: {config_path.resolve()}")
        print(f"  Leaderboard: {leaderboard_path.resolve()}")
        print(f"  Metadata: {metadata_path.resolve()}")

    def _fit_target_vocab(self, df_train):
        """Build target vocabulary for one-hot encoding"""
        keys = []
        for _, row in df_train.iterrows():
            key = str(row.get("PDB ID", "")).strip().upper()
            if not key:
                key = str(row.get("Protein Name", "")).strip().lower()
            if key:
                keys.append(key)
        self.target_keys = sorted(set(keys))
        self.target_index = {k: i for i, k in enumerate(self.target_keys)}
        print(f"Target one-hot vocabulary: {len(self.target_keys)} dimensions")

    def _key_for_row(self, row):
        """Extract target key from dataframe row"""
        key = str(row.get("PDB ID", "")).strip().upper()
        if not key:
            key = str(row.get("Protein Name", "")).strip().lower()
        return key

    def _parse_target_key(self, raw_input):
        """Parse target identifier from user input"""
        import re
        if raw_input is None:
            return "", ""
        s = str(raw_input).strip()
        if not s:
            return "", ""
        parts = [p.strip() for p in re.split(r"\s*[—-]\s*", s, maxsplit=1) if p.strip()]
        
        # Handle "PDB_ID — Protein_Name" format
        if len(parts) == 2 and 4 <= len(parts[0]) <= 5 and parts[0].isalnum():
            return parts[0].upper(), parts[1].lower()
        # Handle PDB ID only
        if 4 <= len(s) <= 5 and s.isalnum():
            return s.upper(), ""
        # Handle protein name only
        return "", s.lower()

    def _fit_target_stats(self, df_train):
        """Fit target statistics for delta predictions"""
        import pandas as pd
        keys = [self._key_for_row(r) for _, r in df_train.iterrows()]
        series = pd.Series(df_train["pIC50"].values, index=keys)
        grouped = series.groupby(level=0)
        self.target_mean = grouped.mean().to_dict()
        self.target_std = {k: (float(v) if v == v else 1.0) 
                          for k, v in grouped.std(ddof=0).fillna(1.0).to_dict().items()}
        print(f"Target statistics fitted for {len(self.target_mean)} targets")

  # =========================
# MAIN APP
# =========================
class BioLinkApp:
    """Main application class for Mini AlphaFold binding predictor"""
    
    def __init__(self):
        self.downloader = DataDownloader()
        self.predictor = LightweightBindingPredictor(self.downloader)
        self.df = None

    def setup_data(self):
        """Setup and prepare training data"""
        print("Setting up data...")
        csv_path = self.downloader.download_bindingdb_sample()
        
        # Load data with appropriate data types
        df = pd.read_csv(
            csv_path,
            dtype={"Ligand SMILES": "string", "PDB ID": "string", 
                   "Protein Name": "string", "Ligand Name": "string"}
        )
        df["IC50 (nM)"] = pd.to_numeric(df["IC50 (nM)"], errors="coerce")

        # Clean and process data
        df = self._clean_dataframe(df)
        df = df.reset_index(drop=True)
        self.df = df

        # Create lookup maps for UI
        self.pdb_name_map = {
            str(pid).upper(): pname
            for pid, pname in zip(self.df["PDB ID"], self.df["Protein Name"])
            if str(pid).strip() != ""
        }
        self.ligand_name_map = dict(zip(self.df["Ligand SMILES"], self.df["Ligand Name"]))

        # Create scaffold-based splits
        self.make_scaffold_splits(test_size=0.20, val_size=0.20, seed=42, save=True)

        print(f"Data setup complete: {len(self.df)} records loaded")
        return True

    def _clean_dataframe(self, df):
        """Clean and standardize the dataframe"""
        from rdkit.Chem import rdMolDescriptors
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize
            standardize_available = True
        except:
            standardize_available = False

        # Ensure proper data types
        for col in ["Ligand SMILES", "PDB ID", "Protein Name", "Ligand Name"]:
            df[col] = df.get(col, "").astype("string").fillna("").str.strip()
        df["IC50 (nM)"] = pd.to_numeric(df["IC50 (nM)"], errors="coerce")

        # Remove invalid entries
        initial_count = len(df)
        df = df.dropna(subset=["IC50 (nM)"]).copy()
        df = df[df["IC50 (nM)"] > 0]
        df = df[df["IC50 (nM)"] < 1e8]  # Remove unrealistic values
        after_basic_filter = len(df)

        # Canonicalize SMILES strings
        def canonicalize_smiles(smiles_str):
            try:
                mol = Chem.MolFromSmiles(str(smiles_str))
                if not mol:
                    return None
                
                if standardize_available:
                    # Remove salts and small fragments
                    fragment_chooser = rdMolStandardize.LargestFragmentChooser()
                    mol = fragment_chooser.choose(mol)
                    # Neutralize charges
                    uncharger = rdMolStandardize.Uncharger()
                    mol = uncharger.uncharge(mol)
                    # Canonicalize tautomers
                    tautomer_enum = rdMolStandardize.TautomerEnumerator()
                    mol = tautomer_enum.Canonicalize(mol)
                
                return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            except:
                return None

        df["canonical_smiles"] = df["Ligand SMILES"].apply(canonicalize_smiles)
        df = df.dropna(subset=["canonical_smiles"]).copy()

        # Calculate pIC50
        df["pIC50"] = 9.0 - np.log10(df["IC50 (nM)"].astype(float))

        # Remove duplicates (keep best IC50 per target-ligand pair)
        key_columns = ["PDB ID", "canonical_smiles"] if (df["PDB ID"].astype(str).str.len() > 0).any() else ["Protein Name", "canonical_smiles"]
        df = df.sort_values("IC50 (nM)", ascending=True)
        df = df.drop_duplicates(subset=key_columns, keep="first").copy()

        # Update SMILES with canonical version
        df["Ligand SMILES"] = df["canonical_smiles"]
        df = df.drop(columns=["canonical_smiles"])

        # Clean text columns
        for col in ["Ligand SMILES", "PDB ID", "Protein Name", "Ligand Name"]:
            df[col] = df[col].astype("string").fillna("").str.strip().replace({"nan": ""})

        final_count = len(df)
        print(f"Data cleaning: {initial_count} → {after_basic_filter} → {final_count} records")
        return df

    def _calculate_scaffold(self, smiles_str):
        """Calculate Murcko scaffold for a SMILES string"""
        from rdkit.Chem.Scaffolds import MurckoScaffold
        try:
            mol = Chem.MolFromSmiles(str(smiles_str))
            if not mol:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if not scaffold or scaffold.GetNumAtoms() == 0:
                return None
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except:
            return None

    def make_scaffold_splits(self, test_size=0.20, val_size=0.20, seed=42, save=True):
        """Create scaffold-based train/validation/test splits"""
        import json
        import random
        from pathlib import Path
        
        assert hasattr(self, "df") and len(self.df) > 0, "Must call setup_data() first"

        df = self.df.reset_index(drop=True).copy()

        # Calculate scaffolds
        df["scaffold"] = df["Ligand SMILES"].apply(self._calculate_scaffold)
        df["scaffold"] = df["scaffold"].fillna(df["Ligand SMILES"].astype(str).str[:64])

        # Group by scaffold
        scaffold_groups = {scaffold: indices.tolist() 
                          for scaffold, indices in df.groupby("scaffold").groups.items()}
        ordered_groups = sorted(scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True)

        # Calculate target sizes
        total_samples = len(df)
        target_train = int(round((1.0 - test_size - val_size) * total_samples))
        target_val = int(round(val_size * total_samples))
        target_test = total_samples - target_train - target_val

        # Shuffle groups for reproducible randomness
        random.Random(seed).shuffle(ordered_groups)

        # Assign scaffolds to splits
        splits = {"train": [], "val": [], "test": []}
        counts = {"train": 0, "val": 0, "test": 0}
        targets = {"train": target_train, "val": target_val, "test": target_test}

        for scaffold, indices in ordered_groups:
            remaining = {k: targets[k] - counts[k] for k in ("train", "val", "test")}
            best_split = max(remaining, key=remaining.get) if max(remaining.values()) > 0 else min(counts, key=counts.get)
            splits[best_split].extend(indices)
            counts[best_split] += len(indices)

        # Clean up splits
        splits = {k: sorted(set(map(int, v))) for k, v in splits.items()}

        # Assign split labels to dataframe
        self.df = df
        self.df["split"] = "train"
        self.df.loc[splits["val"], "split"] = "val"
        self.df.loc[splits["test"], "split"] = "test"
        self.splits = splits

        print(f"Scaffold splits created:")
        print(f"  Train: {len(splits['train'])} samples")
        print(f"  Validation: {len(splits['val'])} samples") 
        print(f"  Test: {len(splits['test'])} samples")
        print(f"  Unique scaffolds - Train: {self.df.loc[splits['train'],'scaffold'].nunique()}, "
              f"Val: {self.df.loc[splits['val'],'scaffold'].nunique()}, "
              f"Test: {self.df.loc[splits['test'],'scaffold'].nunique()}")

        if save:
            save_path = Path(self.downloader.data_dir) / f"scaffold_splits_seed{seed}.json"
            with open(save_path, "w") as f:
                json.dump({
                    "seed": seed,
                    "sizes": {k: len(v) for k, v in splits.items()},
                    "indices": splits
                }, f)
            print(f"Splits saved to: {save_path.resolve()}")
        
        return splits

    def train_model(self):
        """Train the binding prediction model"""
        if self.df is None:
            self.setup_data()
        self.predictor.train(self.df)
        return True

  # =========================
# GRADIO UI
# =========================
def launch_ui(app):
    """Launch the Gradio web interface"""
    import gradio as gr
    from rdkit.Chem import rdMolDescriptors

    # Ensure model is ready
    model_path = Path("trained_model.pkl")
    if model_path.exists():
        import pickle
        with open(model_path, "rb") as f:
            saved_app = pickle.load(f)
        # Transfer trained components
        app.predictor = saved_app.predictor
        app.df = getattr(saved_app, "df", getattr(app, "df", None))
        app.splits = getattr(saved_app, "splits", getattr(app, "splits", None))
        app.downloader = getattr(saved_app, "downloader", getattr(app, "downloader", None))
        if hasattr(saved_app.predictor, "feat_cfg"):
            app.predictor.feat_cfg = saved_app.predictor.feat_cfg
        print("Loaded pretrained model")
    else:
        # Train new model
        app.setup_data()
        app.predictor.tune_hyperparameters(app.df)
        app.predictor.evaluate_test_set(app.df, verbose=True)
        # Save trained model
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(app, f)
        print("Model trained and saved")

    # Prepare UI data
    def smiles_to_formula(smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            return rdMolDescriptors.CalcMolFormula(mol) if mol else ""
        except:
            return ""

    # Build ligand dropdown options
    ligand_df = app.df[["Ligand SMILES", "IC50 (nM)", "PDB ID"]].dropna(subset=["Ligand SMILES"]).copy()
    ligand_df["Formula"] = ligand_df["Ligand SMILES"].apply(smiles_to_formula)
    ligand_options = sorted({f"{row.Formula} — {row['Ligand SMILES']}" 
                           for _, row in ligand_df.iterrows()})

    # Quick selection examples (best binders with PDB structures)
    quick_examples_df = (
        ligand_df[ligand_df["PDB ID"].astype(str).str.len() > 0]
        .drop_duplicates(subset=["Ligand SMILES"])
        .sort_values("IC50 (nM)", ascending=True)
        .head(6)
    )
    quick_examples = [f"{r.Formula} — {r['Ligand SMILES']}" for _, r in quick_examples_df.iterrows()]

    # Build protein dropdown options
    protein_options = sorted({f"{app.pdb_name_map.get(str(pid).upper(), 'Unknown')} — {str(pid).upper()}"
                            for pid in app.df["PDB ID"].tolist() if str(pid).strip()})

    def predict_binding(smiles_input, protein_input):
        """Main prediction function for the UI"""
        # Parse inputs
        if '—' in smiles_input:
            smiles = smiles_input.split('—', 1)[1].strip()
        else:
            smiles = smiles_input.strip()

        # Generate ligand display name
        try:
            mol = Chem.MolFromSmiles(smiles)
            ligand_formula = rdMolDescriptors.CalcMolFormula(mol) if mol else smiles[:24]
        except:
            ligand_formula = smiles[:24]

        # Parse protein input
        if '—' in protein_input:
            pdb_id = protein_input.split('—', 1)[1].strip().upper()
        else:
            pdb_id = (protein_input or "").strip().upper()

        try:
            # Make prediction
            pic50_pred, confidence = app.predictor.predict_with_confidence(smiles, pdb_id)
            if pic50_pred is None:
                return (
                    "<div style='color:#b00; padding:20px; text-align:center;'>Invalid SMILES string</div>",
                    "**Error:** Invalid SMILES string provided",
                    go.Figure(),
                    go.Figure(),
                    pdb_id, smiles
                )

            # Convert to IC50 and determine binding strength
            ic50_nM = 10**(-pic50_pred) * 1e9
            if pic50_pred >= 8:
                strength = "Very Strong (IC50 < 10 nM)"
            elif pic50_pred >= 7:
                strength = "Strong (IC50 < 100 nM)"
            elif pic50_pred >= 6:
                strength = "Moderate (IC50 < 1 μM)"
            elif pic50_pred >= 5:
                strength = "Weak (IC50 < 10 μM)"
            else:
                strength = "Very Weak (IC50 > 10 μM)"

            # Generate visualizations
            ligand_html = smiles_to_svg_html(smiles, size=(320, 320))
            
            protein_structure_path = app.downloader.ensure_pdb_file(pdb_id)
            protein_figure = create_protein_3d_figure(pdb_id, protein_structure_path)
            
            importance_figure = make_feature_importance_plot(app.predictor, top_n=12)

            # Prepare result summary
            protein_name = app.pdb_name_map.get(pdb_id.upper(), "Unknown")
            result_text = (
                f"**Protein:** {pdb_id} — {protein_name}\n\n"
                f"**Ligand:** {ligand_formula}\n\n"
                f"**Predicted pIC50:** {pic50_pred:.2f}\n\n"
                f"**Predicted IC50:** {ic50_nM:.1f} nM\n\n"
                f"**Confidence:** {confidence:.2f}\n\n"
                f"**Binding Strength:** {strength}"
            )

            return (ligand_html, result_text, protein_figure, importance_figure, pdb_id, smiles)

        except Exception as e:
            print(f"Prediction error: {e}")
            return (
                "<div style='color:#b00'>Prediction failed</div>",
                f"**Error:** {str(e)}",
                go.Figure(), go.Figure(), pdb_id, smiles
            )

    # Create Gradio interface
    with gr.Blocks(title="Mini AlphaFold Binding Predictor") as interface:
        gr.Markdown("## Mini AlphaFold: Protein-Ligand Binding Predictor\nEnter a SMILES string and select a protein target to predict binding affinity.")

        with gr.Row():
            with gr.Column():
                ligand_input = gr.Dropdown(
                    choices=ligand_options,
                    value=ligand_options[0] if ligand_options else None,
                    label="Ligand (Formula — SMILES)",
                    allow_custom_value=True
                )

                gr.Examples(
                    examples=[[x] for x in quick_examples],
                    inputs=ligand_input,
                    label="Try these high-affinity ligands:"
                )

            protein_input = gr.Dropdown(
                choices=protein_options,
                value=protein_options[0] if protein_options else None,
                label="Protein Target (Name — PDB ID)"
            )

        predict_button = gr.Button("Predict Binding Affinity", variant="primary")

        with gr.Row():
            ligand_visualization = gr.HTML(label="Ligand Structure (2D)")
            prediction_results = gr.Markdown(label="Prediction Results")

        with gr.Row():
            protein_visualization = gr.Plot(label="Protein Structure (3D)")
            feature_importance = gr.Plot(label="Model Feature Importance")

        with gr.Accordion("Debug Information", open=False):
            debug_pdb = gr.Textbox(label="Selected PDB ID")
            debug_smiles = gr.Textbox(label="Processed SMILES")

        # Connect UI components
        predict_button.click(
            predict_binding,
            inputs=[ligand_input, protein_input],
            outputs=[ligand_visualization, prediction_results, protein_visualization, 
                    feature_importance, debug_pdb, debug_smiles]
        )

    # Launch interface
    interface.queue().launch(share=True, debug=True, server_name="0.0.0.0", server_port=7860)

# =========================
# TRAINING BLOCK
# =========================
if __name__ == "__main__":
    print("Mini AlphaFold Training Pipeline")

    try:
        # Initialize application
        app = BioLinkApp()
        app.setup_data()

        # Ensure predictor exists
        if not hasattr(app, "predictor") or app.predictor is None:
            app.predictor = LightweightBindingPredictor()

        # Hyperparameter tuning on train/validation sets
        leaderboard = app.predictor.tune_hyperparameters(app.df)
        app.predictor.evaluate_test_set(app.df, verbose=True)

        # Save trained model
        import pickle
        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(app, f)

        print("Training completed successfully. Model saved as 'trained_model.pkl'")

    except Exception as e:
        print(f"Training failed: {e}")
        raise

# Application Launch
if __name__ == "__main__":
    print("Loading Mini AlphaFold Application...")

    try:
        # Load pre-trained model
        import pickle
        import os

        if os.path.exists('trained_model.pkl'):
            with open('trained_model.pkl', 'rb') as f:
                app = pickle.load(f)
            print("Pre-trained model loaded successfully")
        else:
            print("No trained model found. Please run the training script first.")
            raise FileNotFoundError("trained_model.pkl not found. Run training first.")

        # Launch web interface
        launch_ui(app)

    except Exception as e:
        print(f"Application launch failed: {e}")
        if "trained_model.pkl" in str(e):
            print("Solution: Run the training script first to create the model")
        raise
