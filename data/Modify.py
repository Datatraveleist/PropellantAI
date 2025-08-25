import pickle
import joblib
import pandas as pd
import os
from rdkit import Chem
from collections import Counter
from joblib import dump, load

# # === 1. Set directory path ===
# folder_path = "./"  # Change to your folder path, e.g. "data/"
# prefix = "High_throughput_HTPB_part"
# output_file = "High_throughput_HTPB_all.pkl"

# # === 2. Find all CSV files with the prefix ===
# csv_files = [f for f in os.listdir(folder_path)
#              if f.startswith(prefix) and f.endswith('.csv')]

# # === 3. Merge all CSV files ===
# df_list = []
# for file in csv_files:
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_csv(file_path)
#     df_list.append(df)

# # Merge into one DataFrame
# merged_df = pd.concat(df_list, ignore_index=True)

# # === 4. Save as .pkl file ===
# merged_df.to_pickle(os.path.join(folder_path, output_file))

# print(f"✅ Merge complete, {len(csv_files)} files merged, saved as: {output_file}")

# === 1. Read CSV file ===
# df = pd.read_pickle("input_modify.pkl")
def Modify1(data_name="High_throughput_HTPB.xlsx"):
    df = pd.read_excel(data_name)  # If you need to read from CSV file
# High_throughput_GAP
# === 4. Define oxygen balance calculation function ===
    def calc_mol_weight(row):
        atomic_weights = {
        'Al': 26.98, 'O': 16.00, 'C': 12.01, 'H': 1.008, 'N': 14.01,
        'Cl': 35.45, 'S': 32.06, 'F': 18.998, 'I': 126.90, 'Br': 79.90
    }
        return sum(row[elem] * atomic_weights[elem] for elem in atomic_weights)

    def calc_OB_CO2(row):
        mol_wt = calc_mol_weight(row)
        O_val = row['O']
        C_val = row['C']
        H_val = row['H']
    
    # Halogen consumes oxygen
        halogen_penalty = row.get('Cl', 0) + row.get('F', 0) + row.get('Br', 0) + row.get('I', 0)
        effective_O = O_val - halogen_penalty
    
        OB_CO2 = 1600 * (effective_O - (2 * C_val + H_val / 2)) / mol_wt
        return OB_CO2

    def calc_OB_CO(row):
        mol_wt = calc_mol_weight(row)
        O_val = row['O']
        C_val = row['C']
        H_val = row['H']

    # Halogen consumes oxygen
        halogen_penalty = row.get('Cl', 0) + row.get('F', 0) + row.get('Br', 0) + row.get('I', 0)
        effective_O = O_val - halogen_penalty
    
        OB_CO = 1600 * (effective_O - (C_val + H_val / 2)) / mol_wt
        return OB_CO


# === 5. Parse element counts from smiles and add to DataFrame ===
    def get_CHNOFClBrI_counts_and_molwt_from_smiles(smiles):
        atomic_weights = {
        'C': 12.01, 'H': 1.008, 'N': 14.01, 'O': 16.00,
        'F': 18.998, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90,
        'S': 32.06, 'Al': 26.98
    }
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {elem: 0 for elem in atomic_weights}, 0.0
        mol = Chem.AddHs(mol)
        atom_counts = Counter()
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] += 1
    # Add implicit hydrogens
        counts = {elem: atom_counts.get(elem, 0) for elem in atomic_weights}
        mol_wt = sum(counts[elem] * atomic_weights[elem] for elem in atomic_weights)
        return counts, mol_wt

# Parse smiles, add element counts and molecular weight columns (CHNO with ECs_ prefix)
    def parse_smiles_and_add_counts(row):
        counts, mol_wt = get_CHNOFClBrI_counts_and_molwt_from_smiles(row['smiles'])
        for elem in counts:
            row[f'ECs_{elem}'] = counts[elem]
        row['ECs_mol_wt'] = mol_wt
        return row

    df = df.apply(parse_smiles_and_add_counts, axis=1)

# === 6. Calculate oxygen balance and element mass fractions ===
    def EMS_calc_OB_CO2(row):
        mol_wt = row['ECs_mol_wt']
        O_val = row.get('ECs_O', row.get('O', 0))
        C_val = row.get('ECs_C', row.get('C', 0))
        H_val = row.get('ECs_H', row.get('H', 0))
        print(f"Processing row: {row['smiles']}, O: {O_val}, C: {C_val}, H: {H_val}, mol_wt: {mol_wt}")
        halogen_penalty = row.get('ECs_Cl', 0) + row.get('ECs_F', 0) + row.get('ECs_Br', 0) + row.get('ECs_I', 0)
        effective_O = O_val - halogen_penalty
        return 1600 * (effective_O - (2 * C_val + H_val / 2)) / mol_wt if mol_wt else 0

    def EMS_calc_OB_CO(row):
        mol_wt = row['ECs_mol_wt']
        O_val = row.get('ECs_O', row.get('O', 0))
        C_val = row.get('ECs_C', row.get('C', 0))
        H_val = row.get('ECs_H', row.get('H', 0))
        halogen_penalty = row.get('ECs_Cl', 0) + row.get('ECs_F', 0) + row.get('ECs_Br', 0) + row.get('ECs_I', 0)
        effective_O = O_val - halogen_penalty
        return 1600 * (effective_O - (C_val + H_val / 2)) / mol_wt if mol_wt else 0

    def EMS_calc_mass_fraction(row, elem):
        atomic_weights = {'C': 12.01, 'H': 1.008, 'O': 16.00, 'N': 14.01, 'F': 18.998, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90, 'S': 32.06, 'Al': 26.98}
        mol_wt = row['ECs_mol_wt']
        key = f'ECs_{elem}' if f'ECs_{elem}' in row else elem
        return row.get(key, 0) * atomic_weights[elem] / mol_wt if mol_wt else 0

    df['ECs_OB%_CO2'] = df.apply(EMS_calc_OB_CO2, axis=1)
    df['ECs_OB%_CO'] = df.apply(EMS_calc_OB_CO, axis=1)
    df['ECs_C%'] = df.apply(lambda row: EMS_calc_mass_fraction(row, 'C'), axis=1)
    df['ECs_H%'] = df.apply(lambda row: EMS_calc_mass_fraction(row, 'H'), axis=1)
    df['ECs_O%'] = df.apply(lambda row: EMS_calc_mass_fraction(row, 'O'), axis=1)
    df['ECs_N%'] = df.apply(lambda row: EMS_calc_mass_fraction(row, 'N'), axis=1)

# === 6. Save the modified file ===
    df.to_pickle("input_modify_f_ECs.pkl")  # You can change the file name as needed
    print("✅ Oxygen balance calculation complete, file saved as input_modify_f_ECs.pkl")
    # df.to_excel(f"f_ECs_{data_name}", index=False)  # Save as Excel file
    # print(f"✅ Oxygen balance calculation complete, file saved as f_ECs_{data_name}")
def Modify2(data_name="input_modify_f_ECs.pkl"):
    print("Converting input_modify_f_ECs.pkl to joblib format...")
    joblib_file = 'input_modify_f_ECs.joblib'
    with open(data_name, 'rb') as f:
        try:
            data = pickle.load(f)
        except ModuleNotFoundError as e:
            print(f"❌ Load failed, module not found: {e}")
            exit()
        except Exception as e:
            print(f"❌ Load failed: {e}")
            exit()
        # Save as joblib format for better compatibility
        joblib.dump(data, joblib_file)
        print(f"✅ Successfully saved as joblib format: {joblib_file}")