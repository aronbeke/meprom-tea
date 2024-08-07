from rdkit import Chem
import pandas as pd
import model.energetics

def read_input(input_file_path):
    '''
    Reads inputs provided in .txt file.
    '''
    data_dict = {}
    with open(input_file_path, 'r') as file:
        for line in file:
            # Split each line by the first '=' character
            key, value = line.strip().split('=', 1)
            # Convert numerical values to float
            try:
                value = float(value)
            except ValueError:
                pass
            data_dict[key] = value
    return data_dict

def correct_case(original_str, string_set):
    '''
    Takes a string and searches through a set of strings. 
    If it finds an element that is the same string but with different cases, 
    it changes the original string to that.
    '''
    for s in string_set:
        if s.lower() == original_str.lower():
            return s
    return original_str

def check_input_data(data_dict):
    error = False
    error_message = ''
    if data_dict['separation_type'] not in ('solute_concentration','solvent_recovery','solute_separation','impurity_removal'):
        error = True
        error_message = 'Unknown separation type.'
        return error, error_message
    if data_dict['separation_type'] in ('solute_concentration','solvent_recovery'):
        binary, ternary = True, False
    else:
        binary, ternary = False, True
    if data_dict['solvent'].lower() not in {s.lower() for s in set(pd.read_csv('data/solvent_properties.csv')['solvent'])}:
        error = True
        error_message = 'Unknown solvent.'
        return error, error_message
    
    data_dict['solvent'] = correct_case(data_dict['solvent'], set(pd.read_csv('data/solvent_properties.csv')['solvent']))
    extractor = model.energetics.solvent_property('extractor',data_dict['solvent'])

    if pd.isna(extractor):
        error = True
        error_message = 'Extraction impossible with chosen solvent.'
        return error, error_message   
    
    if (Chem.MolFromSmiles(data_dict['solute1_smiles']) is None) or (ternary and Chem.MolFromSmiles(data_dict['solute2_smiles']) is None):
        error = True
        error_message = 'One or more solute SMILES are invalid.'
        return error, error_message
    
    if (not isinstance(data_dict['solute1_feed_concentration'], float)) or (ternary and not isinstance(data_dict['solute2_feed_concentration'], float)):
        error = True
        error_message = 'Invalid solute feed concentration(s).'
        return error, error_message

    if not data_dict['solute1_feed_concentration'] > 0 or (ternary and not data_dict['solute2_feed_concentration'] > 0):
        error = True
        error_message = 'Invalid solute feed concentration(s).'
        return error, error_message
    
    if binary and not data_dict['solute1_target_concentration'] > 0:
        error = True
        error_message = 'Invalid solute feed concentration(s).'
        return error, error_message
    
    if binary and (not isinstance(data_dict['solute1_target_concentration'], float)):
        error = True
        error_message = 'Invalid solute target concentration.'
        return error, error_message
    
    if binary and not data_dict['solute1_target_concentration'] > 0:
        error = True
        error_message = 'Invalid solute target concentration.'
        return error, error_message
    
    if ternary and (not isinstance(data_dict['solute_target_concentration_ratio'], float)):
        error = True
        error_message = 'Invalid solute target concentration ratio.'
        return error, error_message
    
    if ternary and not data_dict['solute_target_concentration_ratio'] > 0:
        error = True
        error_message = 'Invalid solute target concentration ratio.'
        return error, error_message    
    
    if (not isinstance(data_dict['solute1_rejection'], float)) or (ternary and not isinstance(data_dict['solute2_rejection'], float)):
        error = True
        error_message = 'Invalid solute rejection(s).'
        return error, error_message
    
    if data_dict['solute1_rejection'] > 1.0 or (ternary and data_dict['solute2_rejection'] > 1.0) or data_dict['solute1_rejection'] < -0.5 or (ternary and data_dict['solute2_rejection'] < -0.5):
        error = True
        error_message = 'Rejection values have to be between -0.5 and 1.0'
        return error, error_message
    
    if (not isinstance(data_dict['solute1_solubility'], float)) or (ternary and not isinstance(data_dict['solute2_solubility'], float)):
        error = True
        error_message = 'Invalid solute solubilities.'
        return error, error_message
    
    if (binary and data_dict['solute1_solubility'] < data_dict['solute1_target_concentration']) or (binary and data_dict['solute1_solubility'] < data_dict['solute1_feed_concentration']):
        error = True
        error_message = 'Target or feed concentration is higher than solubility.'
        return error, error_message
    
    if not data_dict['solute1_solubility'] > 0 or (ternary and not data_dict['solute2_solubility'] > 0):
        error = True
        error_message = 'Solubilities have to be greater than zero.'
        return error, error_message
    
    if (not isinstance(data_dict['external_heat_integration'], float)):
        error = True
        error_message = 'Invalid external heat integration value.'
        return error, error_message
    
    if data_dict['external_heat_integration'] < 0 or data_dict['external_heat_integration'] > 1:
        error = True
        error_message = 'External heat integration has to be between 0 and 1.'
        return error, error_message
    
    if (not isinstance(data_dict['permeance'], float)):
        error = True
        error_message = 'Invalid permeance.'
        return error, error_message
    
    if not data_dict['permeance'] > 0:
        error = True
        error_message = 'Permeance has to be greater than zero.'
        return error, error_message

    if (not isinstance(data_dict['membrane_cost'], float)):
        error = True
        error_message = 'Invalid membrance cost.'
        return error, error_message
    
    if data_dict['membrane_cost'] < 0:
        error = True
        error_message = 'Membrane cost cannot be negative.'
        return error, error_message

    if (not isinstance(data_dict['membrane_lifetime'], float)):
        error = True
        error_message = 'Invalid membrane lifetime.'
        return error, error_message
    
    if not data_dict['membrane_lifetime'] > 0:
        error = True
        error_message = 'Membrane lifetime has to be positive.'
        return error, error_message
    
    if data_dict['country'].lower() not in {s.lower() for s in set(pd.read_csv('data/technoeconomic_data.csv')['country'])}:
        error = True
        error_message = 'Data not available for the selected country.'
        return error, error_message
    
    data_dict['country'] = correct_case(data_dict['country'], set(pd.read_csv('data/technoeconomic_data.csv')['country']))

    return error, error_message
