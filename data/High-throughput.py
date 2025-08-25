import numpy as np
import pandas as pd
import warnings
from itertools import permutations
from functools import lru_cache
from rdkit import Chem
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
# from krr_predict import predict_enthalpy
warnings.filterwarnings("ignore")
from CEA_Wrap import Fuel, RocketProblem, DataCollector, utils
import re
import os
import ast
# import screening
# from screening import process_file
def find_combinations(formulations, target_sum):
    result = []
    num_loops = len(formulations)

    def get_precision(step):
        """Calculate required precision based on the decimal part of the step size"""
        if isinstance(step, float):
            return len(str(step).split('.')[1]) if '.' in str(step) else 0
        return 0  # If step is integer, no decimal places

    # Use cache to avoid repeated calculation of the same state
    @lru_cache(None)
    def recursive_search(index, current_sum, precision):
        # If all formulations have been traversed, check if current sum equals target sum
        if index == num_loops:
            if round(current_sum, precision) == target_sum:
                return [()]
            return []

        formulation = formulations[index]
        current_combinations = []

        # If fixed value
        if len(formulation) == 1:
            n = formulation[0]
            if current_sum + n <= target_sum:
                sub_combinations = recursive_search(index + 1, current_sum + n, precision)
                for sub_comb in sub_combinations:
                    current_combinations.append([round(n, precision)] + list(sub_comb))

        else:
            # Otherwise, it's a range, iterate over each value in the range
            start, end, step = formulation
            step_precision = get_precision(step)

            # Calculate precision value, convert all values to integers
            precision_value = 10 ** step_precision
            start_int = int(round(start * precision_value))
            end_int = int(round(end * precision_value))
            step_int = int(round(step * precision_value))

            # Round start, end, step to avoid floating point errors
            for n in range(start_int, end_int + step_int, step_int):
                if current_sum + n / precision_value > target_sum:
                    break
                sub_combinations = recursive_search(index + 1, current_sum + n / precision_value, step_precision)
                for sub_comb in sub_combinations:
                    current_combinations.append([round(n / precision_value, step_precision)] + list(sub_comb))

        return current_combinations

    result = recursive_search(0, 0, 0)  # Start from the first formulation, initial sum is 0, precision is 0
    return result


class FDP_Fuel:
    def __init__(self, pressure=69.8, pressure_units="bar", pip=69.8, analysis_type="equilibrium", nfz=None, custom_nfz=None, ae_at=None):
        self.pressure = pressure
        self.pressure_units = pressure_units
        self.pip = pip
        self.ae_at = ae_at
        self.analysis_type = analysis_type
        self.nfz = nfz
        self.custom_nfz = custom_nfz

    def cal_isp_optimized_HTPB(self, percent, ems_f, ems_hf, prod=None):
        AL = Fuel("AL(cr)", wt_percent=percent[0])
        EMs = Fuel("EMs", wt_percent=percent[1], chemical_composition=ems_f, hf=float(ems_hf))
        # print(ems_f, ems_hf)
        HTPB = Fuel("HTPB", wt_percent=percent[2])
        AP = Fuel("NH4CLO4(I)", wt_percent=percent[3])
        m_list = [AL, EMs, HTPB, AP]

        if self.ae_at:
            problem = RocketProblem(
                materials=m_list,
                massf=True,
                pressure=self.pressure,
                pressure_units=self.pressure_units,
                analysis_type=self.analysis_type,
                nfz=self.nfz,
                custom_nfz=self.custom_nfz,
                ae_at=self.ae_at,
            )
        else:
            problem = RocketProblem(
                materials=m_list,
                massf=True,
                pressure=self.pressure,
                pressure_units=self.pressure_units,
                pip=self.pip,
                analysis_type=self.analysis_type,
                nfz=self.nfz,
                custom_nfz=self.custom_nfz,
            )

        problem.set_absolute_o_f()
        collector = DataCollector("c_t", "ivac", "isp", "cstar", "c_m", "c_gamma", "cf")
        collector.add_data(problem.run())
        return {
            "ivac": collector.ivac[0],
            "isp": collector.isp[0],
            "c_t": collector.c_t[0],
            "cstar": collector.cstar[0],
            "c_m": collector.c_m[0],
            "c_gamma": collector.c_gamma[0],
            "cf": collector.cf[0],
        }

    def cal_isp_optimized_GAP(self, percent, ems_f, ems_hf, prod=None):
        AL = Fuel("AL(cr)", wt_percent=percent[0])
        EMs = Fuel("EMs", wt_percent=percent[1], chemical_composition=ems_f, hf=float(ems_hf))
        GAP = Fuel("GAP", wt_percent=percent[2], chemical_composition='C 3.0 H 5.0 N 3.0 O 1.0', hf=138)
        AP = Fuel("NH4CLO4(I)", wt_percent=percent[3])
        m_list = [AL, EMs, GAP, AP]

        if self.ae_at:
            problem = RocketProblem(
                materials=m_list,
                massf=True,
                pressure=self.pressure,
                pressure_units=self.pressure_units,
                analysis_type=self.analysis_type,
                nfz=self.nfz,
                custom_nfz=self.custom_nfz,
                ae_at=self.ae_at,
            )
        else:
            problem = RocketProblem(
                materials=m_list,
                massf=True,
                pressure=self.pressure,
                pressure_units=self.pressure_units,
                pip=self.pip,
                analysis_type=self.analysis_type,
                nfz=self.nfz,
                custom_nfz=self.custom_nfz,
            )

        problem.set_absolute_o_f()
        collector = DataCollector("c_t", "ivac", "isp", "cstar", "c_m", "c_gamma", "cf")
        collector.add_data(problem.run())
        return {
            "ivac": collector.ivac[0],
            "isp": collector.isp[0],
            "c_t": collector.c_t[0],
            "cstar": collector.cstar[0],
            "c_m": collector.c_m[0],
            "c_gamma": collector.c_gamma[0],
            "cf": collector.cf[0],
        }

    def cal_isp_optimized_NEPE(self, percent, ems_f, ems_hf, prod=None):
        # print(percent, ems_f, ems_hf)
        AL = Fuel("AL(cr)", wt_percent=percent[0])
        EMs = Fuel("EMs", wt_percent=percent[1], chemical_composition=ems_f, hf=float(ems_hf))
        PEG = Fuel("PEG", wt_percent=percent[2], chemical_composition='C 45.0 H 91.0 O 23.0', hf=-442.8)
        NG = Fuel("NG", wt_percent=percent[3], chemical_composition='C 3.0 H 5.0 O 9.0 N 3.0', hf=-372.5)
        BTTN = Fuel("BTTN", wt_percent=percent[4], chemical_composition='C 4.0 H 7.0 O 9.0 N 3.0', hf=-406.98)
        AP = Fuel("NH4CLO4(I)", wt_percent=percent[5])
        ALH3 = Fuel("ALH3", wt_percent=percent[6])
        m_list = [AL, EMs, PEG, NG, BTTN, AP, ALH3]

        if self.ae_at:
            problem = RocketProblem(
                materials=m_list,
                massf=True,
                pressure=self.pressure,
                pressure_units=self.pressure_units,
                analysis_type=self.analysis_type,
                nfz=self.nfz,
                custom_nfz=self.custom_nfz,
                ae_at=self.ae_at,
            )
        else:
            problem = RocketProblem(
                materials=m_list,
                massf=True,
                pressure=self.pressure,
                pressure_units=self.pressure_units,
                pip=self.pip,
                analysis_type=self.analysis_type,
                nfz=self.nfz,
                custom_nfz=self.custom_nfz,
            )

        problem.set_absolute_o_f()
        collector = DataCollector("c_t", "ivac", "isp", "cstar", "c_m", "c_gamma", "cf")
        collector.add_data(problem.run())
        print(percent, ems_f, ems_hf,collector.isp[0])
        return {
            "ivac": collector.ivac[0],
            "isp": collector.isp[0],
            "c_t": collector.c_t[0],
            "cstar": collector.cstar[0],
            "c_m": collector.c_m[0],
            "c_gamma": collector.c_gamma[0],
            "cf": collector.cf[0],
        }


class FDP:
    # Define propellant formula information as class attributes
    compounds = {
        "AL": {"formula": "Al 1", "Hf": 0},
        "HTPB": {"formula": "C 10.00 H 15.40 O 0.070", "Hf": -51.880},
        "AP": {"formula": "N 1.00 H 4.00 Cl 1.00 O 4.00", "Hf": -295.767000},
        "GAP": {"formula": "C 3.0 H 5.0 N 3.0 O 1.0", "Hf": 138},
        "NG": {"formula": "C 3.0 H 5.0 O 9.0 N 3.0", "Hf": -372.5},
        "PEG": {"formula": "C 45.0 H 91.0 O 23.0", "Hf": -442.8},
        "CL20": {"formula": "C 6.0 H 6.0 O 12.0 N 12.0", "Hf": 415.5},
        "ADN": {"formula": "N 4 H 4 O 4", "Hf": -150.0},
        "BTTN": {"formula": "C 4.0 H 7.0 O 9.0 N 3.0", "Hf": -406.98},
        "ALH3": {"formula": "Al 1 H 3", "Hf": 128.896080}
    }
    ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.01, "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18,
    "Na": 22.99, "Mg": 24.31, "Al": 26.98, "Si": 28.09, "P": 30.97,
    "S": 32.07, "Cl": 35.45, "Ar": 39.95,
                    "K": 39.10, "Ca": 40.08, "Sc": 44.96, "Ti": 47.87, "V": 50.94,}
    def __init__(self, fuel_data):
        self.fuel_data = fuel_data
    @staticmethod
    def parse_chemical_composition(chemical_composition):
        """
        Parses a chemical composition string into a dictionary.
        :param chemical_composition: str, chemical formula (e.g., "C 1 H 10 N 12 O 2")
        :return: dict, e.g., {"C": 1, "H": 10, "N": 12, "O": 2}
        """
        pattern = r'([A-Z][a-z]?)\s*(\d+)'  # Match element symbols and counts
        matches = re.findall(pattern, chemical_composition)
        
        if not matches:
            raise ValueError(f"Invalid chemical composition format: {chemical_composition}")
        
        return {element: int(count) for element, count in matches}
    def smiles_to_composition(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        atom_counts = defaultdict(int)
        for atom in mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1
        composition = " ".join(f"{el} {count}" for el, count in atom_counts.items())
        # Also return a simple element mole dictionary for subsequent calculation
        return composition, dict(atom_counts)

    @staticmethod
    def chemical_composition_to_dict(chemical_compositions, wt_percents, hfs):
        print(wt_percents)
        FIXED_ELEMENTS = {'Al', 'C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'S', 'I'}
        """
        Computes the total molar count for each fixed element across multiple chemical formulas.
        
        :param chemical_compositions: list, chemical formula strings (e.g., ["C 1 H 10 N 12 O 2", "C 2 H 4 O 1"])
        :param wt_percents: list, corresponding weight percentages (e.g., [50.0, 50.0])
        :param hfs: list, corresponding formation enthalpies (e.g., [-200.5, -100.2])
        :return: dict, total molar counts for each fixed element (non-fixed elements default to zero)
        """
        if len(chemical_compositions) != len(wt_percents):
            raise ValueError("The number of chemical compositions must match the number of weight percentages.")
        
        total_element_moles = defaultdict(float)
        total_hf = 0.0  # Accumulate total enthalpy
    
        for chemical_composition, wt_percent, hf in zip(chemical_compositions, wt_percents, hfs):
            composition_dict = FDP.parse_chemical_composition(chemical_composition)
            
            # Calculate molecular weight
            try:
                molecular_weight = sum(FDP.ATOMIC_WEIGHTS[element] * count 
                                       for element, count in composition_dict.items())
                if molecular_weight == 0:
                    raise ValueError(f"Molecular weight is zero for composition: {chemical_composition}")
            except KeyError as e:
                raise ValueError(f"Unknown element '{e.args[0]}' in composition: {chemical_composition}")
    
            # Calculate total moles for this composition
            mol = (1000 * wt_percent * 0.01) / molecular_weight  # 1000 for mg to g; 0.01 for percentage to fraction
            total_hf += mol * hf  # Accumulate formation enthalpy
    
            # Accumulate element moles for fixed elements only
            for element in FIXED_ELEMENTS:
                if element in composition_dict:
                    total_element_moles[element] += composition_dict[element] * mol
    
        total_element_moles['Hf'] = total_hf
    
        # Ensure all fixed elements are present with zero values if missing
        for element in FIXED_ELEMENTS:
            total_element_moles[element] = total_element_moles.get(element, 0.0)
    
        return dict(total_element_moles)

    @staticmethod
    def safe_log_ratio(numerator, denominator):
        EPSILON = 1e-6
        return np.log((numerator + EPSILON) / (denominator + EPSILON))

    @staticmethod
    def calculate_element_ratios(input_data):
        # Atomic weights table
        ATOMIC_WEIGHTS = {
            'C': 12.01, 'H': 1.008, 'N': 14.01, 'O': 16.00,
            'Al': 26.98, 'F': 18.998, 'Cl': 35.45, 'Br': 79.90,
            'S': 32.06, 'I': 126.90
        }
        def safe_log_ratio(numerator, denominator):
            EPSILON = 1e-6
            """ Compute log-ratio safely with epsilon to avoid infinity issues. """
            return np.log((numerator + EPSILON) / (denominator + EPSILON))
        
        # Fixed element set
        FIXED_ELEMENTS = set(ATOMIC_WEIGHTS.keys())
        # Extract element moles, use `.get()` to provide default value
        element_moles = {el: input_data.get(el, 0) for el in FIXED_ELEMENTS}
        
        # Calculate total relative molecular mass
        total_molecular_mass = sum(element_moles[el] * ATOMIC_WEIGHTS[el] for el in FIXED_ELEMENTS)

        # Calculate element mass fractions
        mass_fractions = {
            f'{el}_wt%': (element_moles[el] * ATOMIC_WEIGHTS[el] / total_molecular_mass) * 100
            for el in FIXED_ELEMENTS
        }

        # Calculate oxygen balance (OB%)
        n_C = element_moles.get('C', 0)
        n_H = element_moles.get('H', 0)
        n_O = element_moles.get('O', 0)
        n_N = element_moles.get('N', 0)
        n_F = element_moles.get('F', 0)
        n_Cl = element_moles.get('Cl', 0)
        n_Br = element_moles.get('Br', 0)
        n_I = element_moles.get('I', 0)

        oxygen_balance_co2 = 1600 * (n_O - (2 * n_C + 0.5 * n_H + n_F + n_Cl + n_Br + n_I)) / total_molecular_mass
        oxygen_balance_co = 1600 * (n_O - (n_C + 0.5 * n_H + n_F + n_Cl + n_Br + n_I)) / total_molecular_mass

        # Calculate all pairwise log-ratios (mole ratio) for C, H, O, N
        elements_to_combine = ['C', 'H', 'O', 'N']
        mole_ratios = {
            f'log_mole({a}/{b})': safe_log_ratio(element_moles[a], element_moles[b])
            for a, b in permutations(elements_to_combine, 2)
        }

        # Calculate all pairwise log-ratios (mass ratio) for C, H, O, N
        mass_ratios = {
            f'log_mass({a}/{b})': safe_log_ratio(
                element_moles[a] * ATOMIC_WEIGHTS[a], 
                element_moles[b] * ATOMIC_WEIGHTS[b]
            )
            for a, b in permutations(elements_to_combine, 2)
        }

        # Aggregate results
        result = {
            'OB%_CO2': round(oxygen_balance_co2, 2),
            'OB%_CO': round(oxygen_balance_co, 2),
            **{k: round(v, 4) for k, v in mole_ratios.items()},  # mole ratio
            **{k: round(v, 4) for k, v in mass_ratios.items()},  # mass ratio
            **{k: round(v, 2) for k, v in mass_fractions.items()}  # keep mass fraction
        }

        return result

    def optimize_calculate_HTPB(self, flag="HTPB"):
        compounds = FDP.compounds
        all_results = []
        formulations = ([10, 30, 1], [0, 70, 5], [0, 30, 1], [0, 100, 10])
        formulation_all = find_combinations(formulations, 100)

        save_interval = 500000  # Save every 500,000 combinations
        save_count = 0
        process = 0

        for idx, row in self.fuel_data.iterrows():
            comp_str, elem_moles = self.smiles_to_composition(row['SMILES'])
            EMs_formulas = comp_str
            EMs_hf = row['EOF']
            formula_list = [compounds["AL"]["formula"], EMs_formulas, compounds["HTPB"]["formula"], compounds["AP"]["formula"]]
            Hf_list = [compounds["AL"]["Hf"], EMs_hf, compounds["HTPB"]["Hf"], compounds["AP"]["Hf"]]

            for i in formulation_all:
                if process % 10000 == 0:
                    print(process, len(formulation_all) * len(self.fuel_data), '*****************')
                process += 1
                try:
                    ratio = [i[0], i[1], i[2], i[3]]
                    input_data = FDP.chemical_composition_to_dict(formula_list, ratio, Hf_list)
                    descriptor = FDP.calculate_element_ratios(input_data)
                    input_data.update(descriptor)

                    fdp_Fuel = FDP_Fuel()
                    result_batch = fdp_Fuel.cal_isp_optimized_HTPB(ratio, EMs_formulas, EMs_hf)

                    input_data.update({
                        'Isp': result_batch['isp'],
                        'T_c': result_batch['c_t'],
                        'Cstar': result_batch['cstar'],
                        'smiles': row['SMILES'],
                        'ratio': ratio,
                        'EMs_hf': row['EOF'],
                        'EMs_OB_CO': row['OB_CO'],
                        'EMs_OB_CO2': row['OB_CO2']
                    })

                    all_results.append(input_data)

                except Exception as e:
                    # print(f"Error with ratio {ratio}: {e}")
                    continue

                # Save every 500,000 samples
                if len(all_results) >= save_interval:
                    partial_df = pd.DataFrame(all_results)
                    partial_filename = f'High_throughput_{flag}_part{save_count}.csv'
                    partial_df.to_csv(partial_filename, index=False)
                    print(f"✅ Saved {partial_filename} with {len(all_results)} rows.")
                    save_count += 1
                    all_results.clear()  # Clear list to free memory

        # Save remaining data less than 500,000
        if all_results:
            partial_df = pd.DataFrame(all_results)
            partial_filename = f'High_throughput_{flag}_part{save_count}.csv'
            partial_df.to_csv(partial_filename, index=False)
            print(f"✅ Saved final {partial_filename} with {len(all_results)} rows.")

    def optimize_calculate_GAP(self, flag="GAP"):
        compounds = FDP.compounds
        all_results = []
        formulations = ([10, 30, 1], [0, 70, 5], [0, 30, 1], [0, 100, 10])
        formulation_all = find_combinations(formulations, 100)
        process = 0
        for idx, row in self.fuel_data.iterrows():
            comp_str, elem_moles = self.smiles_to_composition(row['SMILES'])
            EMs_formulas = comp_str
            EMs_hf = row['EOF']
            formula_list = [compounds["AL"]["formula"], EMs_formulas, compounds["GAP"]["formula"], compounds["AP"]["formula"]]
            Hf_list = [compounds["AL"]["Hf"], EMs_hf, compounds["GAP"]["Hf"], compounds["AP"]["Hf"]]
            for i in formulation_all:
                if process % 10000 == 0:
                    print(process, len(formulation_all) * len(self.fuel_data), '*****************')
                process += 1
                try:
                    ratio = [i[0], i[1], i[2], i[3]]
                    input_data = FDP.chemical_composition_to_dict(formula_list, ratio, Hf_list)
                    # input_data['element_moles'] = elem_moles
                    descriptor = FDP.calculate_element_ratios(input_data)
                    input_data.update(descriptor)
                    fdp_Fuel = FDP_Fuel()
                    result_batch = fdp_Fuel.cal_isp_optimized_GAP(ratio, EMs_formulas, EMs_hf)
                    input_data['Isp'] = result_batch['isp']
                    input_data['T_c'] = result_batch['c_t']
                    input_data['Cstar'] = result_batch['cstar']
                    input_data['smiles'] = row['SMILES']
                    input_data['ratio'] = ratio
                    input_data['EMs_hf'] = row['EOF']
                except Exception as e:
                    print(f"Error with ratio {ratio}: {e}")
                    continue
                all_results.append(input_data)
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(f'High_throughput_{flag}.csv', index=False)

    def optimize_calculate_NEPE(self, flag="NEPE"):
        compounds = FDP.compounds
        all_results = []
        formulations = ([10,30,1], [0, 70, 5], [6], [7,8,1], [9], [0, 100, 1])
        formulation_all = find_combinations(formulations, 100)
        print(formulation_all, '***************')
        process = 0
        for idx, row in self.fuel_data.iterrows():
            comp_str, elem_moles = self.smiles_to_composition(row['SMILES'])
            EMs_formulas = comp_str
            EMs_hf = row['HSOLID']
            formula_list = [
                compounds["AL"]["formula"],
                EMs_formulas,
                compounds["PEG"]["formula"],
                compounds["NG"]["formula"],
                compounds["BTTN"]["formula"],
                compounds["AP"]["formula"],
                compounds["ALH3"]["formula"]
            ]
            Hf_list = [
                compounds["AL"]["Hf"],
                EMs_hf,
                compounds["PEG"]["Hf"],
                compounds["NG"]["Hf"],
                compounds["BTTN"]["Hf"],
                compounds["AP"]["Hf"],
                compounds["ALH3"]["Hf"]
            ]
            for i in formulation_all:
                if process % 10000 == 0:
                    print(process, len(formulation_all) * len(self.fuel_data), '*****************')
                process += 1
                try:
                    ratio = [i[0] * 0.6, i[1], i[2], i[3], i[4], i[5], i[0] * 0.4]
                    input_data = FDP.chemical_composition_to_dict(formula_list, ratio, Hf_list)
                    # input_data['element_moles'] = elem_moles
                    descriptor = FDP.calculate_element_ratios(input_data)
                    input_data.update(descriptor)
                    fdp_Fuel = FDP_Fuel()
                    result_batch = fdp_Fuel.cal_isp_optimized_NEPE(ratio, EMs_formulas, EMs_hf)
                    input_data['c_Isp'] = result_batch['isp']
                    input_data['c_T_c'] = result_batch['c_t']
                    input_data['c_Cstar'] = result_batch['cstar']
                    input_data['smiles'] = row['SMILES']
                    input_data['ratio'] = ratio
                    input_data['EMs_hf'] = row['HSOLID']
                except Exception as e:
                    print(f"Error with ratio {ratio}: {e}")
                    continue
                all_results.append(input_data)
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(f'High_throughput_{flag}.csv', index=False)

if __name__ == "__main__":
    print('Start')
    fuel_data = pd.read_excel("CEA_data.xlsx")
    fDP = FDP(fuel_data)
    fDP.optimize_calculate_HTPB()
    fDP.optimize_calculate_GAP()
    fDP.optimize_calculate_NEPE()
    print('Finished normally')
