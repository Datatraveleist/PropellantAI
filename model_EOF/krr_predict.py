import pandas as pd
import joblib  # for loading model and scaler
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from descriptors_generator import generator  # function to generate descriptors

# Define selected features
selected_features = ['nHbondA', 'nHbondD', 'nNH2', 'nAHC', 'nACC', 'nHC', 'nRbond', 'nR', 'nNNO2', 'nONO2', 'nNO2', 'nC(NO2)3', 'nC(NO2)2', 'nC(NO2)',
                     'MinPartialCharge', 'MaxPartialCharge', 'MOLvolume', 'nH', 'nC', 'nN', 'nO', 'PBF', 'TPSA', 'ob', 'total energy', 'molecular weight', 
                     'PMI3', 'nOCH3', 'nCH3', 'Eccentricity', 'PMI2', 'PMI1', 'NPR1', 'NPR2', 'ESTATE_0', 'ESTATE_1', 'ESTATE_2', 'ESTATE_4', 'ESTATE_5',
                     'ESTATE_6', 'ESTATE_8', 'ESTATE_9', 'ESTATE_10', 'ESTATE_11', 'ESTATE_12', 'ESTATE_14', 'ESTATE_16', 'ESTATE_17', 'ESTATE_18', 
                     'ESTATE_19', 'ESTATE_21', 'ESTATE_22', 'ESTATE_23', 'ESTATE_24', 'ESTATE_25', 'ESTATE_27', 'ESTATE_28', 'ESTATE_29', 'ESTATE_30',
                     'ESTATE_35', 'ESTATE_36', 'ESTATE_37', 'ESTATE_39', 'ESTATE_40', 'ESTATE_41', 'ESTATE_43', 'ESTATE_44', 'ESTATE_45', 'ESTATE_46',
                     'ESTATE_47', 'ESTATE_49', 'ESTATE_51', 'ESTATE_52', 'ESTATE_53', 'ESTATE_54', 'ESTATE_56', 'ESTATE_57', 'ESTATE_58', 'ESTATE_59',
                     'ESTATE_60', 'ESTATE_62', 'ESTATE_63', 'ESTATE_64', 'ESTATE_65']

def predict_enthalpy(smiles):
    """
    Input: smiles (list or pd.Series), each item is a SMILES string representing a molecular structure
    Output: predicted enthalpy
    """
    # If input is a list, convert to pd.Series
    if isinstance(smiles, list):
        smiles = pd.Series(smiles)
    
    # Generate descriptors
    descriptors = generator(smiles)
    # If needed, save generated descriptors to CSV file
    # descriptors.to_csv('descriptors.csv', index=False)
    
    # Keep only features required by the model
    descriptors = descriptors[selected_features]
    
    # Load scaler and normalize descriptors
    scaler = joblib.load('scaler.pkl')
    input_data = scaler.transform(descriptors)
    
    # Load saved KRR model
    model_filename = 'best_krr_model.pkl'
    best_krr = joblib.load(model_filename)
    
    # Make predictions using the model
    predictions = best_krr.predict(input_data)
    return predictions

# Example usage:
if __name__ == '__main__':
    sample_smiles = ['O=[N+]([O-])c1ccc2c(c1)c1nonc1n[n+]2[O-]', 'CC1=C([N+]([O-])=O)C=C([N+]([O-])=O)C=C1[N+]([O-])=O']  # Replace with your SMILES list to predict
    preds = predict_enthalpy(sample_smiles)
    # print("Predicted Enthalpy:", preds)
    # Convert SMILES to RDKit Mol objects, then back to SMILES
    # sample_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smile)) for smile in sample_smiles if Chem.MolFromSmiles(smile) is not None]
    # print("Converted SMILES:", sample_smiles)