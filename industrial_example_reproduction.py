import model.energetics

'''
Apixaban - Nanofiltration steps in industrial example
Costs and emissions calculated for the USA
'''
cost_heat = 0.053 # USD/kWh
cost_elec = 0.146 # USD/kWh
co2_heat = 0.457 # co2eq/kWh
co2_elec = 0.438 # co2eq/kWh

membrane_lifetime = 8766 # hours = 1 year
membrane_emissions = 5.357 # kg CO2 /m2

pm = {
        'A': 40.0,  # m2,# module area
        'p0': 2e6, # Pa  # feed pressure
        'l_module': 0.97,  # m # module length
        'l_mesh': 0.00327,  # m # spacer mesh size
        'df': 0.00033,  # m # filament thickness
        'theta': 1.83259571,  # rad # spacer angle
        'n_env': 28, # number of envelopes
        'b_env': 0.85,  # m # envelope width
        'h': 0.0007,  # m # spacer height 
        'T': 303.0,  # K # temperature
        'pump_eff': 0.85,  # pump efficiency
        'stage_cut': 0.75, # approx. stage cut on a separation stage
        'evap_eff': 2.6, # multiple-effect evaporator efficiency
        'n_elements': 3, # elements/modules per stage
        'solute_density': 1300, # kg/m3 assumed
        'nn': 10, # number of simulation nodes per module
        'P': 3.42e-12, #m3/m2sPa
        'R': [0.98],
        'solvent': 'Methanol',
        'L': [],
        'M': [0.460], #kg/mol
        'nu': [0.460/1300], #m3/mol
        'solubility': [500], # mol /m3
        'D': [1e-9], #m2/s
        'ns': 1,
        'n_elements': 1,
        'evap_eff': 2.6,
        'solvent_recovery': False,
        'heat_integration_eff': 0.75
}

pm['L'].append(model.energetics.solute_permeance_from_rejection(pm['R'][0],pm['nu'][0],pm['P'],pm['p0']))

c0 = [3.590905818] # kg/m3
c0[0] = c0[0] / pm['M'][0]
ctarget = 0.95*7.18 / pm['M'][0]  #mol/m3

M_mix = 0.026 # kg/mol
rho_mix = 844 # kg/m3
dH_mix = 37.55 # kJ/mol
heat_int = 0.75

print('Apixaban separation step (industrial example)\n')
print('Calculations assume USA as location\n')
print('Feed concentration (mol/m3): '+str(round(c0[0],2)))
print('Target concentration (mol/m3): '+str(ctarget))
print('Assumed external heat integration: 75%\n')

print('NANOFILTRATION')
nf_e, recovery, nf_area, n_stages, stages = model.energetics.targeted_binary_retentate_nf_cascade(c0[0],ctarget,pm,index=0)
print('Specific energy demand (J/kg): '+str(nf_e))
apix_nf_e = nf_e / 3600000
print('Specific energy demand (kWh/kg): '+str(apix_nf_e))
print('Membrane area required (m2): '+str(nf_area))
apix_production_rate = 4.41 # kg/h
print('Production rate (from simulations) (kg/h): '+str(apix_production_rate))
print()

print('EVAPORATION')
evap_e = model.energetics.evaporation_energy_parametric(c0[0],ctarget,heat_int,2.6,rho_mix,M_mix,dH_mix*1000,pm['M'][0])
print('Specific energy demand (J/kg): '+str(evap_e))
apix_evap_e = evap_e / 3600000
print('Specific energy demand (kWh/kg): '+str(apix_evap_e))

apix_e_reduction = 100 * (apix_evap_e - apix_nf_e)/apix_evap_e
print('\nEnergy reduction (%) : '+str(round(apix_e_reduction,2)))

_, apix_co2eq_reduction = model.energetics.binary_cost_and_co2_calculation(evap_e,0,nf_e,nf_area,cost_heat,cost_elec,500,co2_heat,co2_elec,membrane_emissions,membrane_lifetime,apix_production_rate)
apix_threshold = model.energetics.threshold_membrane_price_calculation(membrane_lifetime,apix_production_rate,cost_heat,cost_elec,evap_e,0,nf_e,nf_area)
print('CO2eq emissions reduction (%) : '+str(round(100*apix_co2eq_reduction,2)))
print('Membrane price threshold (USD/m2/year): '+str(round(apix_threshold,2)))

##################################
'''
Metoprolol - Industrial example
'''

pm = {
        'A': 40.0,  # m2,# module area
        'p0': 2e6, # Pa  # feed pressure
        'l_module': 0.97,  # m # module length
        'l_mesh': 0.00327,  # m # spacer mesh size
        'df': 0.00033,  # m # filament thickness
        'theta': 1.83259571,  # rad # spacer angle
        'n_env': 28, # number of envelopes
        'b_env': 0.85,  # m # envelope width
        'h': 0.0007,  # m # spacer height 
        'T': 303.0,  # K # temperature
        'pump_eff': 0.85,  # pump efficiency
        'stage_cut': 0.75, # approx. stage cut on a separation stage
        'evap_eff': 2.6, # multiple-effect evaporator efficiency
        'n_elements': 3, # elements/modules per stage
        'solute_density': 1300, # kg/m3 assumed
        'nn': 10, # number of simulation nodes per module
        'P': 3.42e-12, #m3/m2sPa
        'R': [0.991, 0.279],
        'solvent': 'Methanol',
        # 'L': [0.003151333594567373,0.0066374923828726225],#m3/m2sPa
        'L': [],
        'M': [0.2674,0.05911], #kg/mol
        'nu': [0.2674/1300,0.05911/1300], #m3/mol
        'solubility': [500, 500], # mol /m3
        'D': [1e-9,1e-9], #m2/s
        'ns': 2,
        'n_elements': 1,
        'evap_eff': 2.6,
        'solvent_recovery': False,
        'heat_integration_eff': 0
}

pm['L'].append(model.energetics.solute_permeance_from_rejection(pm['R'][0],pm['nu'][0],pm['P'],pm['p0']))
pm['L'].append(model.energetics.solute_permeance_from_rejection(pm['R'][1],pm['nu'][1],pm['P'],pm['p0']))

c0 = [13.1,92.5] # mol/m3
ctargetratio = 10

print('######################\n')
print('Metoprolol separation step (industrial example scenario)\n')
print('Feed concentrations (mol/m3): '+str(c0))
print('Target concentration ratio: '+str(ctargetratio))
print()

print('NANOFILTRATION')
tnf_e, recovery, tnf_area, n_stages, c_final, stages = model.energetics.targeted_ternary_retentate_nf_cascade(c0,ctargetratio,pm)
print('Specific energy demand (J/kg): '+str(tnf_e))
meto_nf_e = tnf_e / 3600000
print('Specific energy demand (kWh/kg): '+str(meto_nf_e))
print('Membrane area required (m2): '+str(tnf_area))
meto_production_rate = 4.00 # kg/h
print('Production rate (from simulations) (kg/h): '+str(meto_production_rate))
print()

print('EVAPORATION')
print('We assume 50% yield of API (approx. 17 kg) after extraction step. 925 mol isopropyl amine, 37 kg water, and 38.2 kg toluene are evaporated.')
amine_evap_e = 925*30000
water_evap_e = 37*40700/0.018
toluene_evap_e = 38.2*33200/0.09214
print('Energy demand of evaporation of 37 kg of water (J): '+str(water_evap_e))
print('Energy demand of evaporation of 925 mol amine (J): '+str(amine_evap_e))
print('Energy demand of evaporation of 38.2 kg of toluene (J): '+str(toluene_evap_e))
evap_e = (water_evap_e + toluene_evap_e + amine_evap_e) / 17
print('Total evaporation energy demand (J/kg): '+str(round(evap_e)))
meto_evap_e = evap_e / 3600000
print('Specific energy demand (kWh/kg): '+str(meto_evap_e))

meto_e_reduction = 100 * (meto_evap_e - meto_nf_e)/meto_evap_e
print('\nEnergy reduction (%): '+str(round(meto_e_reduction,2)))

_, meto_co2eq_reduction = model.energetics.ternary_cost_and_co2_calculation('nanofiltration',tnf_area,0,evap_e,0,tnf_e,cost_heat,cost_elec,500,co2_heat,co2_elec,membrane_emissions,membrane_lifetime,meto_production_rate)
meto_threshold = model.energetics.threshold_membrane_price_calculation(membrane_lifetime,meto_production_rate,cost_heat,cost_elec,evap_e,0,tnf_e,tnf_area)
print('CO2eq emissions reduction (%): '+str(round(100*meto_co2eq_reduction,2)))
print('Membrane price threshold (USD/m2/year): '+str(round(meto_threshold,2)))