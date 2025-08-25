import numpy as np
import pandas as pd
import warnings
from itertools import permutations
from functools import lru_cache
import pickle
from rdkit import Chem
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
from krr_predict import predict_enthalpy
warnings.filterwarnings("ignore")
from CEA_Wrap import Fuel, RocketProblem, DataCollector, utils
import re
import os
import ast
# 2. 构建神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # Output 3 values (Isp, T_c, Cstar)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def predict_isp_from_features(feature_dict_or_df):
    # 统一转为 DataFrame
    if isinstance(feature_dict_or_df, dict):
        df = pd.DataFrame([feature_dict_or_df])
    elif isinstance(feature_dict_or_df, pd.DataFrame):
        df = feature_dict_or_df.copy()
    else:
        raise ValueError("Input must be a dict or DataFrame")

    # 只选用模型所需的特征列（保持顺序）
    X_name = [
        "Al",
        "C",
        "N",
        "O",
        "H",
        "Cl",
        "Hf",
        "S",
        "F",
        "Br",
        "I",
        "OB%_CO2",
        'OB%_CO',
        "log_mass(C/H)",
        "log_mass(C/O)",
        "log_mass(C/N)",
        "log_mass(H/C)",
        "log_mass(H/O)",
        "log_mass(H/N)"
        # 'EMs_hf',
        # 'ECs_OB%_CO2',
        # 'ECs_OB%_CO',
        # 'ECs_C%',
        # 'ECs_H%',
        # 'ECs_O%',
        # 'ECs_N%'
    ]
    df = df[X_name]
    # 加载 scaler 和模型
    with open("/public/home/wrh/IAGFDP/model_train/isp_X.pkl", "rb") as f:
        scaler = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(X_name)
    hidden_dim = 256
    model = NeuralNet(input_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load("/public/home/wrh/IAGFDP/model_train/trained_model.pth", map_location=device))
    model.eval()

    # 标准化并预测
    X_scaled = scaler.transform(df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(X_tensor).cpu().numpy().flatten()
    return prediction  # 返回 Isp 预测值数组（与输入行数对应）
def find_combinations(formulations, target_sum):
    result = []
    num_loops = len(formulations)

    def get_precision(step):
        """根据步长的小数部分计算所需的精度"""
        if isinstance(step, float):
            return len(str(step).split('.')[1]) if '.' in str(step) else 0
        return 0  # 如果步长是整数，则没有小数位

    # 使用缓存，避免重复计算相同的状态
    @lru_cache(None)
    def recursive_search(index, current_sum, precision):
        # 如果已经遍历到所有配方，检查当前和是否等于目标和
        if index == num_loops:
            # 直接判断当前和是否与目标和相等
            if round(current_sum, precision) == target_sum:
                return [()]
            return []

        formulation = formulations[index]
        current_combinations = []

        # 如果是固定值
        if len(formulation) == 1:
            n = formulation[0]
            if current_sum + n <= target_sum:
                sub_combinations = recursive_search(index + 1, current_sum + n, precision)
                for sub_comb in sub_combinations:
                    current_combinations.append([round(n, precision)] + list(sub_comb))

        else:
            # 否则是一个范围，遍历该范围中的每个值
            start, end, step = formulation
            step_precision = get_precision(step)

            # 计算精度值，将所有值乘以精度值转为整数
            precision_value = 10 ** step_precision
            start_int = int(round(start * precision_value))
            end_int = int(round(end * precision_value))
            step_int = int(round(step * precision_value))

            # 对 start, end, step 进行四舍五入，避免浮点数误差
            for n in range(start_int, end_int + step_int, step_int):
                # 当前和加上该整数值时，不超过目标和
                if current_sum + n / precision_value > target_sum:
                    break
                sub_combinations = recursive_search(index + 1, current_sum + n / precision_value, step_precision)
                for sub_comb in sub_combinations:
                    current_combinations.append([round(n / precision_value, step_precision)] + list(sub_comb))

        return current_combinations

    result = recursive_search(0, 0, 0)  # 从第一个配方开始，初始和为0，精度为0
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
        try:
            AL = Fuel("AL(cr)", wt_percent=percent[0])
            EMs = Fuel("EMs", wt_percent=percent[1], chemical_composition=ems_f, hf=float(ems_hf))
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

        except Exception as e:
            print(f"[Error in cal_isp_optimized_HTPB]: {e}")
            return {
                "ivac": 0.0,
                "isp": 0.0,
                "c_t": 0.0,
                "cstar": 0.0,
                "c_m": 0.0,
                "c_gamma": 0.0,
                "cf": 0.0,
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
    # 将各推进剂的配方信息定义为类属性
    def __init__(self):
        self.compounds = {
        "AL": {"formula": "Al 1", "Hf": 0},
        "HTPB": {"formula": "C 10.00 H 15.40 O 0.070", "Hf": -51.880},
        "AP": {"formula": "N 1.00 H 4.00 Cl 1.00 O 4.00", "Hf": -295.767000},
        "GAP": {"formula": "C 3.0 H 5.0 N 3.0 O 1.0", "Hf": 138},
        "NG": {"formula": "C 3.0 H 5.0 O 9.0 N 3.0", "Hf": -372.5},
        "PEG": {"formula": "C 45.0 H 91.0 O 23.0", "Hf": -442.8},
        "CL20": {"formula": "C 6.0 H 6.0 O 12.0 N 12.0", "Hf": 415.5},
        "ADN": {"formula": "N 4 H 4 O 4", "Hf": -150.0},
        "BTTN": {"formula": "C 4.0 H 7.0 O 9.0 N 3.0", "Hf": -406.98},
        "ALH3": {"formula": "Al 1 H 3", "Hf": 128.896080}}
        self.ATOMIC_WEIGHTS = {
            "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.01,
            "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18, "Na": 22.99, "Mg": 24.31,
            "Al": 26.98, "Si": 28.09, "P": 30.97, "S": 32.07, "Cl": 35.45, "Ar": 39.95,
            "K": 39.10, "Ca": 40.08, "Sc": 44.96, "Ti": 47.87, "V": 50.94, "Cr": 52.00,
            "Mn": 54.94, "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38,
            "Ga": 69.72, "Ge": 72.63, "As": 74.92, "Se": 78.96, "Br": 79.90, "Kr": 83.80,
            "Rb": 85.47, "Sr": 87.62, "Y": 88.91, "Zr": 91.22, "Nb": 92.91, "Mo": 95.95,
            "Tc": 98.00, "Ru": 101.1, "Rh": 102.9, "Pd": 106.4, "Ag": 107.9, "Cd": 112.4,
            "In": 114.8, "Sn": 118.7, "Sb": 121.8, "I": 126.9, "Te": 127.6, "Xe": 131.3,
            "Cs": 132.9, "Ba": 137.3, "La": 138.9, "Ce": 140.1, "Pr": 140.9, "Nd": 144.2,
            "Pm": 145.0, "Sm": 150.4, "Eu": 152.0, "Gd": 157.3, "Tb": 158.9, "Dy": 162.5,
            "Ho": 164.9, "Er": 167.3, "Tm": 168.9, "Yb": 173.0, "Lu": 175.0, "Hf": 178.5,
            "Ta": 180.9, "W": 183.8, "Re": 186.2, "Os": 190.2, "Ir": 192.2, "Pt": 195.1,
            "Au": 197.0, "Hg": 200.6, "Tl": 204.4, "Pb": 207.2, "Bi": 208.9, "Po": 209.0,
            "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.0,
            "Pa": 231.0, "U": 238.0, "Np": 237.0, "Pu": 244.0, "Am": 243.0, "Cm": 247.0,
            "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0, "Md": 258.0, "No": 259.0,
            "Lr": 262.0, "Rf": 267.0, "Db": 270.0, "Sg": 271.0, "Bh": 270.0, "Hs": 277.0,
            "Mt": 276.0, "Ds": 281.0, "Rg": 280.0, "Cn": 285.0, "Nh": 284.0, "Fl": 289.0,
            "Mc": 288.0, "Lv": 293.0, "Ts": 294.0, "Og": 294.0
        }
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
        # 同时返回一个简单的元素摩尔数字典用于后续计算
        return composition, dict(atom_counts)

    @staticmethod
    def chemical_composition_to_dict(chemical_compositions, wt_percents, hfs):
        ATOMIC_WEIGHTS = {
            "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.01,
            "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18, "Na": 22.99, "Mg": 24.31,
            "Al": 26.98, "Si": 28.09, "P": 30.97, "S": 32.07, "Cl": 35.45, "Ar": 39.95,
            "K": 39.10, "Ca": 40.08, "Sc": 44.96, "Ti": 47.87, "V": 50.94, "Cr": 52.00,
            "Mn": 54.94, "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38,
            "Ga": 69.72, "Ge": 72.63, "As": 74.92, "Se": 78.96, "Br": 79.90, "Kr": 83.80,
            "Rb": 85.47, "Sr": 87.62, "Y": 88.91, "Zr": 91.22, "Nb": 92.91, "Mo": 95.95,
            "Tc": 98.00, "Ru": 101.1, "Rh": 102.9, "Pd": 106.4, "Ag": 107.9, "Cd": 112.4,
            "In": 114.8, "Sn": 118.7, "Sb": 121.8, "I": 126.9, "Te": 127.6, "Xe": 131.3,
            "Cs": 132.9, "Ba": 137.3, "La": 138.9, "Ce": 140.1, "Pr": 140.9, "Nd": 144.2,
            "Pm": 145.0, "Sm": 150.4, "Eu": 152.0, "Gd": 157.3, "Tb": 158.9, "Dy": 162.5,
            "Ho": 164.9, "Er": 167.3, "Tm": 168.9, "Yb": 173.0, "Lu": 175.0, "Hf": 178.5,
            "Ta": 180.9, "W": 183.8, "Re": 186.2, "Os": 190.2, "Ir": 192.2, "Pt": 195.1,
            "Au": 197.0, "Hg": 200.6, "Tl": 204.4, "Pb": 207.2, "Bi": 208.9, "Po": 209.0,
            "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.0,
            "Pa": 231.0, "U": 238.0, "Np": 237.0, "Pu": 244.0, "Am": 243.0, "Cm": 247.0,
            "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0, "Md": 258.0, "No": 259.0,
            "Lr": 262.0, "Rf": 267.0, "Db": 270.0, "Sg": 271.0, "Bh": 270.0, "Hs": 277.0,
            "Mt": 276.0, "Ds": 281.0, "Rg": 280.0, "Cn": 285.0, "Nh": 284.0, "Fl": 289.0,
            "Mc": 288.0, "Lv": 293.0, "Ts": 294.0, "Og": 294.0
        }
        # print(wt_percents)
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
                molecular_weight = sum(ATOMIC_WEIGHTS[element] * count 
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
        # 原子量表
        ATOMIC_WEIGHTS = {
            'C': 12.01, 'H': 1.008, 'N': 14.01, 'O': 16.00,
            'Al': 26.98, 'F': 18.998, 'Cl': 35.45, 'Br': 79.90,
            'S': 32.06, 'I': 126.90
        }
        def safe_log_ratio(numerator, denominator):
            # 小正数以避免 log(0) 或除零
            EPSILON = 1e-6
            """ Compute log-ratio safely with epsilon to avoid infinity issues. """
            return np.log((numerator + EPSILON) / (denominator + EPSILON))
        
        # 固定元素集
        FIXED_ELEMENTS = set(ATOMIC_WEIGHTS.keys())
        # 提取元素摩尔数，使用 `.get()` 提供默认值
        element_moles = {el: input_data.get(el, 0) for el in FIXED_ELEMENTS}
        
        # 计算总相对分子质量
        total_molecular_mass = sum(element_moles[el] * ATOMIC_WEIGHTS[el] for el in FIXED_ELEMENTS)

        # 计算元素质量分数
        mass_fractions = {
            f'{el}_wt%': (element_moles[el] * ATOMIC_WEIGHTS[el] / total_molecular_mass) * 100
            for el in FIXED_ELEMENTS
        }

        # 计算氧平衡 (OB%)
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

        # 计算所有 C、H、O、N 两两组合的 log-ratio (摩尔比)
        elements_to_combine = ['C', 'H', 'O', 'N']
        mole_ratios = {
            f'log_mole({a}/{b})': safe_log_ratio(element_moles[a], element_moles[b])
            for a, b in permutations(elements_to_combine, 2)
        }

        # 计算所有 C、H、O、N 两两组合的 log-ratio (质量比)
        mass_ratios = {
            f'log_mass({a}/{b})': safe_log_ratio(
                element_moles[a] * ATOMIC_WEIGHTS[a], 
                element_moles[b] * ATOMIC_WEIGHTS[b]
            )
            for a, b in permutations(elements_to_combine, 2)
        }

        # 汇总结果
        result = {
            'OB%_CO2': round(oxygen_balance_co2, 2),
            'OB%_CO': round(oxygen_balance_co, 2),
            **{k: round(v, 4) for k, v in mole_ratios.items()},  # 摩尔比
            **{k: round(v, 4) for k, v in mass_ratios.items()},  # 质量比
            **{k: round(v, 2) for k, v in mass_fractions.items()}  # 保留质量分数
        }

        return result

    def optimize_calculate_HTPB(self,smile,EOF,ratio = [18, 40, 12, 30],flag="HTPB"):
        compounds = self.compounds
        all_results = []
        # formulations = ([18], [0, 70, 15], [12], [0, 100, 1])
        # formulation_all = find_combinations(formulations, 100)
        # print(formulation_all,'***************')
        process = 0
        # 对每个样本获取 SMILES 对应的组成信息及元素摩尔数
        comp_str, elem_moles = self.smiles_to_composition(smile)
        EMs_formulas = comp_str
        EMs_hf = EOF
        # 构造配方：AL、EMs、HTPB、AP
        formula_list = [compounds["AL"]["formula"], EMs_formulas, compounds["HTPB"]["formula"], compounds["AP"]["formula"]]
        Hf_list = [compounds["AL"]["Hf"], EMs_hf, compounds["HTPB"]["Hf"], compounds["AP"]["Hf"]]
        try:
            input_data = FDP.chemical_composition_to_dict(formula_list, ratio, Hf_list)
            fdp_Fuel = FDP_Fuel()
            result_batch = fdp_Fuel.cal_isp_optimized_HTPB(ratio, EMs_formulas, EMs_hf)
        except Exception as e:
            print(f"Error with ratio {ratio}: {e}")
            result_batch['isp'] = 0
        # final_df = pd.DataFrame(all_results)
        # final_df.to_csv(f'High_throughput_{flag}.csv', index=False)
        return result_batch['isp']
    def descriptor_HTPB(self,smile,EOF,ratio = [18, 20, 12, 50],flag="HTPB"):
        compounds = self.compounds
        all_results = []
        comp_str, elem_moles = self.smiles_to_composition(smile)
        EMs_formulas = comp_str
        EMs_hf = EOF
        # 构造配方：AL、EMs、HTPB、AP
        formula_list = [compounds["AL"]["formula"], EMs_formulas, compounds["HTPB"]["formula"], compounds["AP"]["formula"]]
        Hf_list = [compounds["AL"]["Hf"], EMs_hf, compounds["HTPB"]["Hf"], compounds["AP"]["Hf"]]
        input_data = FDP.chemical_composition_to_dict(formula_list, ratio, Hf_list)
        descriptor = FDP.calculate_element_ratios(input_data)
        input_data.update(descriptor)
        return input_data
    def optimize_calculate_GAP(self, flag="GAP"):
        compounds = FDP.compounds
        all_results = []
        formulations = ([18], [0, 70, 15], [12], [0, 100, 1])
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
        # final_df = pd.DataFrame(all_results)
        # final_df.to_csv(f'High_throughput_{flag}.csv', index=False)
        return all_results
    def optimize_calculate_NEPE(self, flag="NEPE"):
        compounds = FDP.compounds
        all_results = []
        formulations = ([15], [35, 50, 3], [6], [7], [9], [0, 100, 1])
        formulation_all = find_combinations(formulations, 100)
        print(formulation_all, '***************')
        process = 0
        for idx, row in self.fuel_data.iterrows():
            comp_str, elem_moles = self.smiles_to_composition(row['SMILES'])
            EMs_formulas = comp_str
            EMs_hf = row['EOF']
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
                    ratio = [i[0] * 0.7, i[1], i[2], i[3], i[4], i[5], i[0] * 0.3]
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
                    input_data['EMs_hf'] = row['EOF']
                except Exception as e:
                    print(f"Error with ratio {ratio}: {e}")
                    continue
                all_results.append(input_data)
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(f'High_throughput_{flag}.csv', index=False)
def predict_HTPB(SMILES_list,ratio=None):
    isps = []
    smiles = SMILES_list
    EOFs = predict_enthalpy(smiles)
    fDP = FDP()
    ratios = []
    for smile, EOF in zip(smiles,EOFs):
        print(smile,EOF)
        isp_ = []
        if ratio:
            # isp_CEA = fDP.optimize_calculate_HTPB(smile,EOF,ratio)
            descriptor = fDP.descriptor_HTPB(smile,EOF,ratio)
            result = predict_isp_from_features(descriptor)
            isp = result[::3] 
            isps.append(isp)
            max_ratio = ratio
        # isp = fDP.optimize_calculate_HTPB(smiles,EOF)
        else:
            formulations = ([18], [10, 70, 10], [12], [0, 100, 1])
            formulation_all = find_combinations(formulations, 100)
            max_isp = None
            max_ratio = None
            for ratio_ in formulation_all:
                # isp_value = fDP.optimize_calculate_HTPB(smile,EOF,ratio_)
                # print(isp_value)
                descriptor = fDP.descriptor_HTPB(smile, EOF, ratio_)
                result = predict_isp_from_features(descriptor)
                isp_value = result[::3]
                # isp_value = isp[0] if isinstance(isp, (list, np.ndarray)) else isp
                if (max_isp is None) or (isp_value > max_isp):
                    max_isp = isp_value
                    max_ratio = ratio_
                # print(f"ratio: {ratio_}, isp: {isp_value}")
            isps.append(max_isp)
            # ratio = max_ratio
        ratios.append(max_ratio)
    isps = [i[0] for i in isps]
    return isps,ratios,EOFs
    # return isp['isp']

if __name__ == "__main__":
    # fDP = FDP()
    # SMILES_list = ['CONc1c(N([N+]([O-])=O)CN([N+]([O-])=O)c2c(n3c([N+]([O-])=O)nc(N)n3)non2)non1','O=C(c1c(O)cccc1)Cc2c(O)c([N+]([O-])=O)cc([N+]([O-])=O)c2']
    # # isp = fDP.optimize_calculate_HTPB(SMILES_list[0],100,[18, 40, 12, 30])
    # isp,ratio,EOFs = predict_HTPB(SMILES_list)
    # # Save to table
    # # df = pd.DataFrame({"SMILES": SMILES_list, "Isp": isp})
    # # df.to_csv("smiles_isp.csv", index=False)
    # print(isp,ratio,EOFs)
    # # Read SMILES list from txt file, one SMILES per line



    df_input = pd.read_csv("bo_ceshi_deduplicated_CHNO.csv")
    SMILES_list = df_input["std_smiles"].dropna().astype(str).tolist()
    isp, ratios,EOFs = predict_HTPB(SMILES_list)
    # # Save to table
    print("Number of SMILES:", len(SMILES_list))
    print("Number of Isp:", len(isp))
    print("Number of EOFs:", len(EOFs))
    # Add results to original DataFrame and save
    df_input["Isp"] = isp
    df_input["EOF"] = EOFs
    df_input["ratios"] = ratios
    df_input.to_csv("generated_molecules_prediction.csv", index=False)
