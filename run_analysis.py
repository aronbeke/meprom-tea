import pandas as pd
import warnings
import model.auxiliary as auxiliary
import model.energetics

if __name__ == "__main__":

    input_path = input('Input file path or name: ')
    if not input_path.endswith(".txt"):
        input_path += ".txt"

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    try:
        data = auxiliary.read_input(input_path)
    except FileNotFoundError:
        print('File not found. All input data has to be provided in an input .txt file.')
        exit()

    is_error, error_message = auxiliary.check_input_data(data)
    if is_error:
        print(error_message)
        exit()
    
    print('Calculating...\n')

    data['permeance'] = (1/3600)*1e-8*data['permeance']
    data['membrane_lifetime'] = 8766*data['membrane_lifetime']

    if data['separation_type'] == 'solute_concentration' or data['separation_type'] == 'solvent_recovery':
        best_configuration, lowest_energy, conventional_energy, energy_reduction, co2eq_reduction, cost_reduction, threshold_membrane_price = model.energetics.run_binary_separation_calculation(data)
    elif data['separation_type'] == 'solute_separation' or data['separation_type'] == 'impurity_removal':
        best_configuration, lowest_energy, conventional_energy, energy_reduction, co2eq_reduction, cost_reduction, threshold_membrane_price = model.energetics.run_ternary_separation_calculation(data)
    else:
        print('Unrecognized separation type.')
        exit()

    if best_configuration == 'nanofiltration-coupled':
        best_configuration = 'ternary nanofiltration - binary nanofiltration - evaporation'
    if best_configuration == 'extraction-coupled':
        best_configuration = 'extraction - binary nanofiltration - evaporation'

    print('Best technology: '+best_configuration)
    print('Specific energy demand (J/kg): '+str(round(lowest_energy)))
    if data['separation_type'] in ('solute_concentration','solvent_recovery'):
        print('Evaporation specific energy demand (J/kg): '+str(round(conventional_energy)))
    else:
        print('Extraction-evaporation specific energy demand (J/kg): '+str(round(conventional_energy)))
    print()
    print('Specific energy demand reduction compared to conventional technologies (extraction, evaporation): '+str(round(100*energy_reduction,2))+'%')
    print('CO2 equivalent emissions (global warming potential) reduction: '+str(round(100*co2eq_reduction,2))+'%')
    print('Operating cost reduction: '+str(round(100*cost_reduction,2))+'%')
    print('Threshold membrane price (USD/m2/year): '+str(round(threshold_membrane_price,2)))