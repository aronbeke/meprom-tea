import numpy as np
import pandas as pd
import math
from scipy.optimize import fsolve
import rdkit.Chem
import rdkit.Chem.Crippen
import rdkit.Chem.AllChem
import rdkit.Chem.Descriptors

'''
All concentrations are molar concentrations (mol/m3). SI units if not indicated otherwise.
'''

################# Parameter initialization #################

def initiate_separation_parameters(separation_type,solvent,permeance,heat_integration_efficiency,samples,empty=False):
    '''
    Initiates parameters.
    Returns either partial parameter combinations or all parameters for samples taken from Kernel Density Estimate models.
    Sample order: [['corrected_log10_solute_permeance','logS_298_from_aq [log10(mol/L)]','solute_molar_mass','logPhi_water','logPhi_heptane','solute_diffusivity_from_smiles']]
    Samples is a list, one or two numpy arrays.
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
        'R': [], # rejection
        'L': [], #m3/m2sPa # solute permeance
        'M': [], #kg/mol # solute molar mass
        'logPhi_water': [], # solute logP
        'logPhi_heptane': [], # solute logPhi (solvent/water)
        'nu': [], #m3/mol # solute molar volume 
        'solubility': [], # mol /m3
        'D': [], #m2/s # solute diffusivity
    }

    if separation_type == 'solvent_recovery':
        pm['solvent_recovery'] = True
    else:
        pm['solvent_recovery'] = False
    
    pm['P'] = permeance #m3/m2sPa
    pm['solvent'] = solvent
    pm['heat_integration_eff'] = heat_integration_efficiency

    if empty:
        return pm

    sample_1 = samples[0].flatten().tolist()
    L_1 = np.power(10,sample_1[0]) + 1e-10
    pm['L'].append(L_1)
    pm['solubility'].append(np.power(10,sample_1[1])*1000)
    pm['M'].append(sample_1[2])
    pm['logPhi_water'].append(sample_1[3])
    pm['logPhi_heptane'].append(sample_1[4])
    pm['D'].append(sample_1[5])
    molar_volume1 = sample_1[2] / pm['solute_density']
    pm['nu'].append(molar_volume1)
    pm['R'].append(rejection_from_solute_permeance(L_1, molar_volume1, permeance, pm['p0']))

    if separation_type == 'solute_concentration' or separation_type == 'solvent_recovery':
        pm['ns'] = 1
    else:
        pm['ns'] = 2
        sample_2 = samples[1].flatten().tolist()
        L_2 = np.power(10,sample_2[0]) + 1e-10
        pm['L'].append(L_2)
        pm['solubility'].append(np.power(10,sample_2[1])*1000)
        pm['M'].append(sample_2[2])
        pm['logPhi_water'].append(sample_2[3])
        pm['logPhi_heptane'].append(sample_2[4])
        pm['D'].append(sample_2[5])
        molar_volume2 = sample_2[2] / pm['solute_density']
        pm['nu'].append(molar_volume2)
        pm['R'].append(rejection_from_solute_permeance(L_2, molar_volume2, permeance, pm['p0']))

    return pm

def solvent_property(property,solvent):
    property_df = pd.read_csv('data/solvent_properties.csv')
    return property_df[property_df['solvent'] == solvent][property].iloc[0]

def solvent_from_smiles(solvent):
    solvent_smiles = {
        'CCO': 'Ethanol',
        'CCCCCCC': 'Heptane',
        'O': 'Water',
        'CC(OCC)=O': 'Ethyl acetate',
        'CO': 'Methanol',
        'CCCCCC': 'Hexane',
        'N#CC': "Acetonitrile",
        'CC(C)=O': "Acetone",
        'O=CN(C)C': 'Dimethylformamide',
        'ClCCl': 'Dichloromethane',
        'CC1CCCO1': '2-Methyltetrahydrofuran',
        'CC1=CC=CC=C1': 'Toluene',
        'CC(O)C': 'Isopropanol',
        'CC(N(C)C)=O': 'Dimethylacetamide',
        'CC(CC)=O': 'Methyl ethyl ketone',
        'CC(C)(C)OC': 'Methyl tert-butyl ether',
        'CC#N': 'Acetonitrile',
        'C1CCCO1': 'Tetrahydrofuran',
        'C1CCCCC1': 'Cyclohexane'
    }
    return solvent_smiles[solvent]

################# Molecular calculations #################

def logp_from_smiles(smiles):
    new_mol=rdkit.Chem.MolFromSmiles(smiles)
    val = rdkit.Chem.Crippen.MolLogP(new_mol)
    return val

def calculate_spherical_radius_from_smiles(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Compute 3D coordinates
    mol = rdkit.Chem.AddHs(mol)
    rdkit.Chem.AllChem.EmbedMolecule(mol, rdkit.Chem.AllChem.ETKDG())
    rdkit.Chem.AllChem.UFFOptimizeMolecule(mol)

    volume = rdkit.Chem.AllChem.ComputeMolVolume(mol)

    radius = ((3 * volume) / (4 * np.pi))**(1/3)

    # returns radius in Angstrom
    return radius

def molar_mass_from_smiles(smiles):
    # return molar mass in kg/mol
    return rdkit.Chem.Descriptors.ExactMolWt(rdkit.Chem.MolFromSmiles(smiles)) / 1000

def diffusivity_from_smiles(smiles, viscosity, T=298):
    # Stokes-Einstein–Sutherland equation
    # returns m2/s if viscosity in Pa s

    kB = 1.380649e-23 # J⋅K−1 

    try:
        r = calculate_spherical_radius_from_smiles(smiles) * 1e-10 # m

        D = kB * T / (6 * np.pi * viscosity * r)
    except:
        print('Diffusivity exception')
        D = 1e-9 

    return D
    
def solute_permeance_from_rejection(rejection, molar_volume, solvent_permeance, dp):
    # neglecting osmotic pressure
    # SI units
    Rg = 8.314
    T = 298
    L = (solvent_permeance * dp * (1-rejection)) / (1 + np.exp((-molar_volume/(Rg*T))*dp) * (rejection - 1))
    return L

def rejection_from_solute_permeance(solute_permeance, molar_volume, solvent_permeance, dp):
    Rg = 8.314
    T = 298
    rejection = (solvent_permeance * dp + solute_permeance*(np.exp((-molar_volume/(Rg*T))*dp) - 1)) / (solvent_permeance * dp + solute_permeance*(np.exp((-molar_volume/(Rg*T))*dp)))
    return rejection

################# Nanofiltration mechanistic model #################

def spiral_wound_mass_transfer(F0,D_list,h,rho,eta,l_mesh,df,theta,n_env,b_env):
    '''
    Calculates mass transfer coefficients
    '''

    V_sp = 0.5 * np.pi * (df**2) * l_mesh
    V_tot = (l_mesh**2) * h * np.sin(theta)
    epsilon = 1 - (V_sp/V_tot)
    S_vsp = 4 / df
    dh = (4*epsilon) / (2*(1/h) + (1-epsilon)*S_vsp)
    v = F0 / (b_env*h*epsilon*n_env)
    Re = (rho*v*dh)/eta

    k_list = []
    for D in D_list:
        Sc = eta/(rho*D)
        Sh = 0.065*(Sc ** 0.25)*(Re ** 0.875)
        k = (D*Sh)/dh
        k_list.append(k)

    return k_list


def csd_ternary(p,args):
    '''
    Simulating one node with classical solution-diffusion
    '''
    F0,c0_list,A,P,L_list,p0,pp,nu_list,k_list,ns = args

    Rg = 8.314
    T = 298.15
    
    Fr = p[0]
    Fp = p[1]
    cr_list = p[2:(2+ns)]
    cm_list = p[(2+ns):(2+2*ns)]
    cp_list = p[(2+2*ns):(2+3*ns)]

    cr, cm, cp, c0 = np.transpose(np.array(cr_list)), np.transpose(np.array(cm_list)), np.transpose(np.array(cp_list)), np.transpose(np.array(c0_list))
    L, nu, k = np.transpose(np.array(L_list)), np.transpose(np.array(nu_list)), np.transpose(np.array(k_list))

    netdp = (p0 - pp - Rg*T*(np.sum(cm) - np.sum(cp)))

    #####
    eq_flux = L * A * (cm - cp*np.exp((-(1/(Rg*T))*nu)*netdp)) - Fp* cp

    conc_pol = (cm - cp) - np.exp(np.divide((Fp/A),k))*(cr - cp)

    eq_solvent_flow = P * A * netdp - Fp

    eq_omb = Fr + Fp - F0

    eq_mb = F0*c0 - Fr*cr - Fp*cp
    ######

    eq = []
    eq.extend(eq_flux.ravel().tolist())
    eq.extend(conc_pol.ravel().tolist())
    eq.append(float(eq_solvent_flow))
    eq.append(float(eq_omb))
    eq.extend(eq_mb.ravel().tolist())

    return eq


def sd_mesh_module(parameters, constants):
    '''
    Simulating one separation module
    '''
    F0, c0, A, T, nn, ns, p0, pp = parameters
    L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env = constants

    k_list = spiral_wound_mass_transfer(F0,D_list,h,rho,eta,l_mesh,df,theta,n_env,b_env)

    A_bin = A / nn

    F0_bin = F0
    c0_bin = c0

    Cp_matrix = np.zeros((ns,nn))
    Fp_vector = np.zeros((nn,1))

    for i in range(nn):
        Fp_init = P*A_bin*(p0-pp)
        Fr_init = F0_bin - Fp_init
        cr_init = (np.array(c0_bin)).tolist()
        cm_init = (np.array(c0_bin)*1.1).tolist()
        cp_init = (np.array(c0_bin)*0.9).tolist()
        init = [Fr_init,Fp_init]
        init.extend(cr_init)
        init.extend(cm_init)
        init.extend(cp_init)

        args = [F0_bin,c0_bin,A_bin,P,L_list,p0,pp,nu_list,k_list,ns]
        sol = fsolve(csd_ternary, init, args=args)
        
        Fr_bin = sol[0]
        Fp_bin = sol[1]
        cr_bin = sol[2:(2+ns)]
        cm_bin = sol[(2+ns):(2+2*ns)]
        cp_bin = sol[(2+2*ns):(2+3*ns)]
        cp = np.array(cp_bin)

        Cp_matrix[:,i] = cp.flatten()
        Fp_vector[i,0] = Fp_bin
        F0_bin = Fr_bin
        c0_bin = cr_bin
    
    Fr = Fr_bin
    cr_final = cr_bin
    Fp = np.sum(Fp_vector)
    cp = (Cp_matrix @ Fp_vector) / Fp
    cp_final = cp_bin

    els = {}
    els['Fr'] = Fr
    els['cr'] = cr_final
    els['F0'] = F0
    els['c0'] = c0
    els['p0'] = p0
    els['Fp'] = Fp
    els['cp'] = cp_final
    els['pp'] = pp
    return els


def sd_stage_sim(n_elements,parameters, constants):
    '''
    Simulating one separation stage consisting of n_elements modules
    '''
    F0, c_feed, A, T, nn, ns, p0, pp = parameters

    solutions = []
    stage_solutions = {}
    
    parameters = [F0, c_feed, A, T, nn, ns, p0, pp]
    els = sd_mesh_module(parameters, constants)
    solutions.append(els)

    for i in range(n_elements-1):
        parameters = [solutions[-1]['Fr'], solutions[-1]['cr'], A, T, nn, ns, p0, pp]
        els = sd_mesh_module(parameters, constants)
        solutions.append(els)

    Fp_final = np.sum(i['Fp'] for i in solutions)
    cp_final = []
    for i in range(len(c_feed)):
        cp_final.append(np.sum(j['Fp']*j['cp'][i] for j in solutions)/Fp_final)

    stage_solutions['Fr'] = solutions[-1]['Fr']
    stage_solutions['cr'] = solutions[-1]['cr']
    stage_solutions['F0'] = F0
    stage_solutions['c0'] = c_feed
    stage_solutions['p0'] = p0
    stage_solutions['Fp'] = Fp_final
    stage_solutions['cp'] = cp_final
    stage_solutions['pp'] = pp
    stage_solutions['elements'] = solutions
    return stage_solutions


################# Nanofiltration calculations #################


def check_concentration(c_actual, c_target, pm):
    '''
    Concentration target check
    '''
    if not pm['solvent_recovery']:
        return c_actual < c_target
    else:
        return c_actual > c_target


def targeted_binary_retentate_nf_cascade(c0,ctarget,pm,index=0):
    '''
    Considers only first solute of list
    Specific energy in J/kg
    '''

    ns = 1
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = [pm['L'][index]], [pm['nu'][index]], [pm['solubility'][index]], [pm['D'][index]], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    rho = solvent_property('density',pm['solvent'])
    eta = solvent_property('viscosity',pm['solvent'])
    constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env]
    stages = {}

    if c0 == 0:
        return float('inf'), 0, float('inf'), float('inf'), stages

    if (ctarget <= c0 and not pm['solvent_recovery']) or (ctarget >= c0 and pm['solvent_recovery']):
        return 0, 1, 0, 0, stages

    # calculate approx. feed flow
    p0 = pm['p0'] # Pa
    pp = 0
    
    F0_ref = P*A*n_elements*(p0-pp) / stage_cut
    F0 = F0_ref
    c0_actual = c0

    n_stages = 0
    parallel = []

    try:
        while check_concentration(c0_actual, ctarget, pm) and n_stages <= 19:
            parameters = [F0, [c0_actual], A, T, nn, ns, p0, pp]
            mod_results = sd_stage_sim(n_elements,parameters, constants)
            stages[n_stages] = mod_results
            c0_actual = mod_results['cr'][0]
            Fr_actual = mod_results['Fr']

            if check_concentration(c0_actual, ctarget, pm):
                parallel_factor = round(F0_ref/Fr_actual)
            else:
                parallel_factor = 1
            parallel.append(parallel_factor)

            F0 = parallel_factor*Fr_actual

            n_stages += 1
    except RuntimeWarning:
        return float('inf'), 0, float('inf'), float('inf'), stages
    
    if n_stages == 20 and check_concentration(c0_actual, ctarget, pm):
        return float('inf'), 0, float('inf'), float('inf'), stages
    
    for i in range(n_stages):
        stages[i]['no_in_parallel'] = np.prod(parallel[i:])
    
    power = (1/pump_eff) * np.prod(parallel) * p0 * F0_ref
    if not pm['solvent_recovery']:
        time = 1/(stages[n_stages-1]['Fr']*stages[n_stages-1]['cr'][0]) # for 1 mol
        time_w = time / pm['M'][0]
        specific_energy_demand = power*time_w
        recovery = (stages[n_stages-1]['Fr'] * stages[n_stages-1]['cr'][0]) / (F0_ref * c0 * np.prod(parallel))
        #print(str(stages[n_stages-1]['cr'][0]*pm['M'][0]*stages[n_stages-1]['Fr']*3600),'kg/h')
    else:
        time = 1/(stages[n_stages-1]['Fr']) # for 1 m3
        specific_energy_demand = power*time
        recovery = (stages[n_stages-1]['Fr']) / (F0_ref * np.prod(parallel))

    total_area = A * n_elements* np.sum(stages[i]['no_in_parallel'] for i in range(n_stages))

    return specific_energy_demand, recovery, total_area, n_stages, stages


def targeted_binary_permeate_nf_cascade(c0,ctarget,pm,index=0):
    '''
    Only one solute
    '''

    ns = 1
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = [pm['L'][index]], [pm['nu'][index]], [pm['solubility'][index]], [pm['D'][index]], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    rho = solvent_property('density',pm['solvent'])
    eta = solvent_property('viscosity',pm['solvent'])
    constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env]

    stages = {}

    if c0 == 0:
        return float('inf'), 0, float('inf'), float('inf'), stages

    if (ctarget <= c0 and not pm['solvent_recovery']) or (ctarget >= c0 and pm['solvent_recovery']):
        return 0, 1, 0, 0, stages

    # calculate approx. feed flow / pressure
    p0 = pm['p0']
    pp = 0
    
    F0_ref = P*A*n_elements*(p0-pp) / stage_cut
    F0 = F0_ref
    c0_actual = c0

    n_stages = 0
    parallel = []

    try:
        while check_concentration(c0_actual, ctarget, pm) and n_stages <= 19:
            parameters = [F0, [c0_actual], A, T, nn, ns, p0, pp]
            mod_results = sd_stage_sim(n_elements,parameters, constants)
            stages[n_stages] = mod_results
            c0_actual = mod_results['cp'][0]
            Fp_actual = mod_results['Fp']

            if check_concentration(c0_actual, ctarget, pm):
                parallel_factor = round(F0_ref/Fp_actual)
            else:
                parallel_factor = 1
            parallel.append(parallel_factor)

            F0 = parallel_factor*Fp_actual

            n_stages += 1
    except RuntimeWarning:
        return float('inf'), 0, float('inf'), float('inf'), stages

    if n_stages == 20 and check_concentration(c0_actual, ctarget, pm):
        return float('inf'), 0, float('inf'), float('inf'), stages

    for i in range(n_stages):
        stages[i]['no_in_parallel'] = np.prod(parallel[i:])
    
    power = (1/pump_eff) * np.sum(stages[i]['no_in_parallel'] * stages[i]['F0'] for i in range(n_stages)) * p0
    if not pm['solvent_recovery']:
        time = 1/(stages[n_stages-1]['Fp']*stages[n_stages-1]['cp'][0]) # for 1 mol
        time_w = time / pm['M'][0]
        specific_energy_demand = power*time_w
        recovery = (stages[n_stages-1]['Fp'] * stages[n_stages-1]['cp'][0]) / (F0_ref * c0 * np.prod(parallel))
    else:
        time = 1/(stages[n_stages-1]['Fp']) # for 1 m3
        specific_energy_demand = power*time
        recovery = (stages[n_stages-1]['Fp']) / (F0_ref * np.prod(parallel))

    total_area = A * n_elements* np.sum(stages[i]['no_in_parallel'] for i in range(n_stages))

    return specific_energy_demand, recovery, total_area, n_stages, stages


def targeted_ternary_retentate_nf_cascade(cfeed,cratio_target,pm):
    '''
    Separation of two solutes.
    Model algorithm keeps input flow rate of each stage constant.
    '''

    ns = 2
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = pm['L'], pm['nu'], pm['solubility'], pm['D'], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    rho = solvent_property('density',pm['solvent'])
    eta = solvent_property('viscosity',pm['solvent'])
    constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env]

    # Determine which solute will be enriched
    stages = {}
    if L_list[0] < L_list[1]:
        def c_ratio(c):
            return c[0]/c[1]
    elif L_list[0] > L_list[1]:
        def c_ratio(c):
            return c[1]/c[0]
        
    # Catch if separation is impossible
    if L_list[0] == L_list[1] or pm['R'][0] == pm['R'][1]:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages

    # Catch if separation is unnecessary
    if cratio_target < c_ratio(cfeed):
        if L_list[0] < L_list[1]:
            c_final = cfeed[0]
        elif L_list[0] > L_list[1]:
            c_final = cfeed[1]
        return 0, 1, 0, 0, c_final, stages

    # Calculate stage feed flow based on approximate stage cut
    p0 = pm['p0']
    pp = 0
    
    F0_ref = P*A*n_elements*(p0-pp) / stage_cut
    F0 = F0_ref

    c0 = cfeed.copy()
    c0_actual = c0.copy()

    # simulating process
    n_stages = 0

    conc_lost = False

    try:
        while c_ratio(c0_actual) < cratio_target and n_stages <= 19:
            # print()
            # print('sim params: ',F0,c0_actual)
            parameters = [F0, c0_actual, A, T, nn, ns, p0, pp]  
            mod_results = sd_stage_sim(n_elements,parameters, constants)

            #Check if we have surpassed solubility limits
            solution_factors = [mod_results['cr'][0]/solubility_list[0], mod_results['cr'][1]/solubility_list[1]]

            if (solution_factors[0] >= 0.95 or solution_factors[1] >= 0.95) and n_stages == 0:
                c0[0] = c0[0] * 0.5
                c0[1] = c0[1] * 0.5
                c0_actual = c0.copy()
            else:
                stages[n_stages] = mod_results
                if c_ratio(mod_results['cr']) < cratio_target:
                    c0_actual[0] = (mod_results['cr'][0] * mod_results['Fr'])/F0
                    c0_actual[1] = (mod_results['cr'][1] * mod_results['Fr'])/F0
                    mod_results['Fd'] = mod_results['Fp'] # dilution rate corresponds to the lost permeate
                else:
                    c0_actual[0] = mod_results['cr'][0]
                    c0_actual[1] = mod_results['cr'][1]
                    mod_results['Fd'] = 0
                # print('stage success','Fr',Fr_actual,'c_ratio',c_ratio(c0_actual),'parallels',parallel,'c',c0_actual)
                n_stages += 1
            
            if c0_actual[0] < c0[0] * 0.001 or c0_actual[1] < c0[1] * 0.001:
                conc_lost = True
                break

    except RuntimeWarning:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages
    
    if (n_stages == 20 and c_ratio(c0_actual) < cratio_target) or conc_lost:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages
    
    # Calculating total area, power
    total_area = A * n_elements * n_stages
    dilution_power = (1/pump_eff) * np.sum(stages[i]['Fd'] for i in range(n_stages)) * p0
    power = (1/pump_eff) * p0 * F0 + dilution_power

    # Calculating recovery, necessary time, final concentration
    if L_list[0] < L_list[1]:
        recovery = (stages[n_stages-1]['Fr'] * stages[n_stages-1]['cr'][0]) / (F0_ref * c0[0])
        time = 1/(stages[n_stages-1]['Fr']*stages[n_stages-1]['cr'][0])
        time_w = time / pm['M'][0]
        c_final = stages[n_stages-1]['cr'][0]
        #print(str(c_final*pm['M'][0]),'kg/m3')
    else:
        recovery = (stages[n_stages-1]['Fr'] * stages[n_stages-1]['cr'][1]) / (F0_ref * c0[1])
        time = 1/(stages[n_stages-1]['Fr']*stages[n_stages-1]['cr'][1])
        time_w = time / pm['M'][1]
        c_final = stages[n_stages-1]['cr'][1]
        #print(str(c_final*pm['M'][1]),'kg/m3')
    
    specific_energy_demand = power*time_w

    #print(str(stages[n_stages-1]['Fr']*3600),'m3/h')
    return specific_energy_demand, recovery, total_area, n_stages, c_final, stages


def targeted_ternary_permeate_nf_cascade(cfeed, cratio_target, pm):
    '''
    Two solutes, permeate nanofiltration cascade.
    '''

    ns = 2
    stage_cut, A, T, nn, n_elements, pump_eff = pm['stage_cut'], pm['A'], pm['T'], pm['nn'], pm['n_elements'], pm['pump_eff']
    L_list, nu_list, solubility_list, D_list, P = pm['L'], pm['nu'], pm['solubility'], pm['D'], pm['P']
    h, l_module, l_mesh, df,theta,n_env,b_env = pm['h'], pm['l_module'], pm['l_mesh'], pm['df'], pm['theta'], pm['n_env'], pm['b_env']
    rho = solvent_property('density',pm['solvent'])
    eta = solvent_property('viscosity',pm['solvent'])
    constants = [L_list, nu_list, solubility_list, D_list, P, h, rho, l_module, eta, l_mesh, df,theta,n_env,b_env]

    stages = {}
    if L_list[0] < L_list[1]:
        def c_ratio(c):
            return c[1]/c[0]
    elif L_list[0] > L_list[1]:
        def c_ratio(c):
            return c[0]/c[1]
    
    if L_list[0] == L_list[1] or pm['R'][0] == pm['R'][1]:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages

    if cratio_target < c_ratio(cfeed):
        if L_list[0] < L_list[1]:
            c_final = cfeed[1]
        elif L_list[0] > L_list[1]:
            c_final = cfeed[0]
        return 0, 1, 0, 0, c_final, stages

    # calculate approx. feed flow / pressure
    p0 = pm['p0']
    pp = 0
    
    F0_ref = P*A*n_elements*(p0-pp) / stage_cut
    F0 = F0_ref

    c0 = cfeed.copy()
    c0_actual = c0.copy()

    n_stages = 0
    conc_lost = False
    # warnings.simplefilter('error', RuntimeWarning)
    # try:
    while c_ratio(c0_actual) < cratio_target and n_stages <= 19:
        parameters = [F0, c0_actual, A, T, nn, ns, p0, pp]
        mod_results = sd_stage_sim(n_elements,parameters, constants)

        solution_factors = [mod_results['cp'][0]/solubility_list[0], mod_results['cp'][1]/solubility_list[1]]

        if (solution_factors[0] >= 0.95 or solution_factors[1] >= 0.95) and n_stages == 0:
            c0[0] = c0[0] * 0.5
            c0[1] = c0[1] * 0.5
            c0_actual = c0.copy()
        else:
            stages[n_stages] = mod_results
            if c_ratio(mod_results['cp']) < cratio_target:
                c0_actual[0] = (mod_results['cp'][0] * mod_results['Fp'])/F0
                c0_actual[1] = (mod_results['cp'][1] * mod_results['Fp'])/F0
                mod_results['Fd'] = mod_results['Fr'] # dilution rate corresponds to the lost retentate
            else:
                c0_actual[0] = mod_results['cp'][0]
                c0_actual[1] = mod_results['cp'][1]
                mod_results['Fd'] = 0
            # print('stage success','Fr',Fr_actual,'c_ratio',c_ratio(c0_actual),'parallels',parallel,'c',c0_actual)
            n_stages += 1
            # print(c_ratio(c0_actual))

    if (n_stages == 20 and c_ratio(c0_actual) < cratio_target) or conc_lost:
        return float('inf'), 0, float('inf'), float('inf'), 0, stages
    
    dilution_power = (1/pump_eff) * np.sum(stages[i]['Fd'] for i in range(n_stages)) * p0
    power = (1/pump_eff) * n_stages * F0 * p0 + dilution_power
    total_area = A * n_elements * n_stages

    if L_list[0] < L_list[1]:
        recovery = (stages[n_stages-1]['Fp'] * stages[n_stages-1]['cp'][1]) / (F0_ref * c0[1])
        time = 1/(stages[n_stages-1]['Fp']*stages[n_stages-1]['cp'][1])
        time_w = time / pm['M'][0]
        c_final = stages[n_stages-1]['cp'][1]
    else:
        recovery = (stages[n_stages-1]['Fp'] * stages[n_stages-1]['cp'][0]) / (F0_ref * c0[0])
        time = 1/(stages[n_stages-1]['Fp']*stages[n_stages-1]['cp'][0])
        time_w = time / pm['M'][0]
        c_final = stages[n_stages-1]['cp'][0]
    specific_energy_demand = power*time_w

    return specific_energy_demand, recovery, total_area, n_stages, c_final, stages


################# Evaporation calculations #################


def evaporation_energy(c0,ctarget,pm,index=0):
    '''
    Only considers first solute
    c: mol/m3
    '''
    if (ctarget <= c0 and not pm['solvent_recovery']) or (ctarget >= c0 and pm['solvent_recovery']):
        return 0
    
    if c0 == 0 and pm['solvent_recovery'] == False:
        return float('inf')
    elif c0 == 0 and pm['solvent_recovery']:
        return 0
    
    rho = solvent_property('density',pm['solvent'])
    M = solvent_property('molar_mass',pm['solvent'])
    dH = solvent_property('heat_of_evaporation',pm['solvent'])

    try:
        if pm['solvent_recovery'] == False:
            specific_energy_demand = (1-pm['heat_integration_eff'])*(1/pm['evap_eff'])*(rho/(M))*((1/(c0*(pm['M'][index])))-(1/(ctarget*(pm['M'][index]))))*(dH)
        elif pm['solvent_recovery']:
            specific_energy_demand = (1-pm['heat_integration_eff'])*(1/pm['evap_eff'])*(rho/(M))*(dH)
        return specific_energy_demand
    except RuntimeWarning:
        return float('inf')
    
def evaporation_energy_parametric(c0,ctarget,heat_int,evap_eff,rho,solvent_M,dH,solute_M):
    try:
        specific_energy_demand = (1-heat_int)*(1/evap_eff)*(rho/(solvent_M))*((1/(c0*(solute_M)))-(1/(ctarget*(solute_M))))*(dH)
        return specific_energy_demand
    except RuntimeWarning:
        return float('inf')


def evaporation_energy_multiple(c0,ctarget,pm,index=0):
    '''
    Multiple target concentrations
    Only considers first solute
    c: mol/m3
    '''

    rho = solvent_property('density',pm['solvent'])
    M = solvent_property('molar_mass',pm['solvent'])
    dH = solvent_property('heat_of_evaporation',pm['solvent'])

    conc = np.linspace(c0,ctarget,pm['c_resolution'])
    try:
        specific_energy_demand = (1-pm['heat_integration_eff'])*(1/pm['evap_eff'])*(rho/(M))*((1/(c0*(pm['M'][index])))-np.divide(1,conc*(pm['M'][index])))*(dH)
        return specific_energy_demand
    except RuntimeWarning:
        return float('inf')


################# Hybrid (coupled) technology calculations #################


def coupled_binary_energy(c0,ctarget,pm,c_resolution=10,index=0):
    '''
    Calculates optimal configuration of continuous NF and evaporation in series. The number of NF stages is calculated from the overall c0 and ctarget.
    Index argument is useful in ternary parameter situations
    '''

    if c0 == 0:
        return float('inf'), 0, float('inf'), 0, float('inf'), float('inf'), float('inf')

    if ctarget <= c0:
        return 0, 1, 0, 0, 0, 0, 0

    conc = np.linspace(c0,ctarget,c_resolution)
    c_shift = c0
    E_list = []

    optimal_E = evaporation_energy(c0,ctarget,pm,index=index)
    c_shift = c0

    optimal_E_cnf = 0
    optimal_E_eva = optimal_E
    optimal_area = 0
    optimal_n_stages = 0

    optimal_recovery = 1

    for c in conc:
        if pm['R'][index] > 0:
            nf_specific_energy, nf_recovery, total_area, n_stages, elements = targeted_binary_retentate_nf_cascade(c0,c,pm,index=index)
            evap_specific_energy = evaporation_energy(c,ctarget,pm,index=index)
            total_energy = nf_specific_energy + evap_specific_energy
        else:
            nf_specific_energy, nf_recovery, total_area, n_stages, elements = targeted_binary_permeate_nf_cascade(c0,c,pm,index=index)
            evap_specific_energy = evaporation_energy(c,ctarget,pm,index=index)
            total_energy = nf_specific_energy + evap_specific_energy

        E_list.append(total_energy)

        if total_energy < optimal_E:
            optimal_E = total_energy
            c_shift = c
            optimal_E_cnf = nf_specific_energy
            optimal_E_eva = evap_specific_energy
            optimal_recovery = nf_recovery
            optimal_area = total_area
            optimal_n_stages = n_stages

    return optimal_E, optimal_recovery, optimal_area, c_shift, optimal_E_cnf, optimal_E_eva, optimal_n_stages


def coupled_binary_energy_after_ternary(E_ternary,c1,c_target,solute_idx,pm,c_resolution=10):
    '''
    Calculates optimal configuration of continuous NF and evaporation in series after a ternary separation. The number of NF stages is calculated from the overall c0 and ctarget.
    solute_idx: solute to examine from parameters in pm
    '''

    if c1 == 0:
        return float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'), float('inf'), 0

    if c_target <= c1:
        return 0, E_ternary, 0, 0, 1, 0, 0, 0

    conc = np.linspace(c1,c_target,c_resolution)
    c_shift = c1
    E_list = []
    recovery = 1
    n_stages = 0
    area = 0

    # Initialization
    if c_target < c1 or pm['R'][solute_idx] == 0:
        pass
    else:
        E_eva_min = evaporation_energy(c1,c_target,pm,index=solute_idx)
        E_nf_min = 0
        recovery = 1
        c_shift = c1
        n_stages = 0
        area = 0

    if c_target < c1:
        E_nf_min = 0
        E_eva_min = 0
        recovery = 1
        c_shift = c1
        n_stages = 0
        area = 0
    elif pm['R'][solute_idx] == 0:
        E_nf_min = 0
        E_eva_min = evaporation_energy(c1,c_target,pm,index=solute_idx)
        recovery = 1
        c_shift = c1
        n_stages = 0
        area = 0
    else:
        E_min = E_ternary/recovery + E_eva_min
        for c in conc:
            if pm['R'][solute_idx] > 0:
                E_nf, recovery_nf, total_area_nf, n_stages_nf, _ = targeted_binary_retentate_nf_cascade(c1,c,pm,index=solute_idx)
                E_eva = evaporation_energy(c,c_target,pm,index=solute_idx)
                if recovery_nf == 0:
                    E = float('inf')
                else:
                    E = E_ternary/recovery_nf + E_nf + E_eva
            else:
                E_nf, recovery_nf, total_area_nf, n_stages_nf, _ = targeted_binary_permeate_nf_cascade(c1,c,pm,index=solute_idx)
                E_eva = evaporation_energy(c,c_target,pm,index=solute_idx)
                if recovery_nf == 0:
                    E = float('inf')
                else:
                    E = E_ternary/recovery_nf + E_nf + E_eva
            E_list.append(E) 
            if E < E_min:
                E_nf_min = E_nf
                E_eva_min = E_eva
                recovery = recovery_nf
                c_shift = c
                E_min = E
                n_stages = n_stages_nf
                area = total_area_nf
    
    E_binary_min = E_nf_min + E_eva_min
    corrected_ternary_E = E_ternary/recovery
    nanofiltration_E = E_nf_min
    evaporation_E = E_eva_min

    return E_binary_min, corrected_ternary_E, nanofiltration_E, evaporation_E, recovery, n_stages, area, c_shift


################# Extraction calculations #################

def extraction(case,c0,c_ratio_t,KA,KB):
    '''
    K is defined as:
    K = c(extractor)/c(original)

    A-standard: A prefers original solution better
    B-standard: B prefers original solution better
    '''
    cA_0 = c0[0]
    cB_0 = c0[1]
    c_ratio_0 = cA_0 / cB_0

    # Determine target ratios
    if case == 'A-standard':
        c_ratio_t_1 = c_ratio_t
        c_ratio_t_2 = 1/c_ratio_t
    elif case == 'B-standard':
        c_ratio_t_1 = 1/c_ratio_t
        c_ratio_t_2 = c_ratio_t   

    extraction_constant = (1+KB)/(1+KA)
    n = np.ceil(np.log(c_ratio_t_1/c_ratio_0)/np.log(extraction_constant))
    try:
        cA_1_final = cA_0/np.power((1+KA),n)
        cB_1_final = cB_0/np.power((1+KB),n)
    except:
        cA_1_final = 0
        cB_1_final = 0

    m = np.ceil(np.log(c_ratio_t_2/c_ratio_0)/np.log((KA/KB)*extraction_constant))
    try:
        cA_2_final = np.power((KA),m)*cA_0/np.power((1+KA),m)
        cB_2_final = np.power((KB),m)*cB_0/np.power((1+KB),m)
    except:
        cA_2_final = 0
        cB_2_final = 0        

    if case == 'A-standard':
        if cB_1_final != 0:
            c_ratio_1_final = cA_1_final/cB_1_final
        elif cA_1_final !=0 and cB_1_final ==0:
            c_ratio_1_final = c_ratio_t
        else:
            c_ratio_1_final = 0
        if cA_2_final != 0:
            c_ratio_2_final = cB_2_final/cA_2_final
        elif cB_2_final !=0 and cA_2_final ==0:
            c_ratio_2_final = c_ratio_t
        else:
            c_ratio_2_final = 0
        recovery_A = cA_1_final/cA_0
        recovery_B = cB_2_final/cA_0

    elif case == 'B-standard':
        if cA_1_final != 0:
            c_ratio_1_final = cB_1_final/cA_1_final
        elif cB_1_final != 0 and cA_1_final ==0:
            c_ratio_1_final = c_ratio_t
        else:
            c_ratio_1_final = 0
        if cB_2_final != 0:
            c_ratio_2_final = cA_2_final/cB_2_final
        elif cA_2_final != 0 and cB_2_final ==0:
            c_ratio_2_final = c_ratio_t
        else:
            c_ratio_2_final = 0
        recovery_A = cA_2_final/cA_0
        recovery_B = cB_1_final/cA_0

    return cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B


def counter_current_extraction(case,c0,c_ratio_t,KA,KB,just_impurity=False):
    '''
    Continuous counter current extraction based on Perry's handbook
    Looks like it doesnt work when both solutes prefer the original solvent -> there is an inherent limit
    Therefore if both solutes prefer the same solvent it will ruin one of the extraction branches
    1. n stages of counter current extraction
    2. m stages where the final extractor solution stream is the feed and original solvent the extractor

    If extraction is for purity removal only then solute on index 0 (solute A) is considered the main / target solute
    Ratio of flow rates is 1.
    '''

    cA_0 = c0[0]
    cB_0 = c0[1]

    cA_n = cA_0
    cB_n = cB_0
    cA_m = cA_0
    cB_m = cB_0
    n = 0
    m = 0
    c_ratio_n = cA_0/cB_0
    c_ratio_m = cB_0/cA_0
    rec_A = 1
    rec_B = 1

    if just_impurity and case=='A-standard':
        while cA_n/cB_n < c_ratio_t:
            n += 1
            cA_n = cA_0*(KA-1)/(np.power(KA,(n+1))-1)
            cB_n = cB_0*(KB-1)/(np.power(KB,(n+1))-1)
        c_ratio_n = cA_n/cB_n
        rec_A = cA_n/cA_0
    elif just_impurity and case=='B-standard':
        cA_ext = KA*cA_0/(1+KA)  # conc. of A in the extracting solvent after 1 step
        cB_ext = KB*cB_0/(1+KB)
        cA_m = cA_ext
        cB_m = cB_ext
        while cA_m/cB_m < c_ratio_t:
            m += 1
            cA_m = cA_ext*(1/KA-1)/(np.power(1/KA,(m+1))-1)
            cB_m = cB_ext*(1/KB-1)/(np.power(1/KB,(m+1))-1)
        c_ratio_m = cA_m/cB_m
        rec_A = cA_m/cA_0

    if just_impurity == False and case == 'A-standard':
        while cA_n/cB_n < c_ratio_t:
            n += 1
            cA_n = cA_0*(KA-1)/(np.power(KA,(n+1))-1)
            cB_n = cB_0*(KB-1)/(np.power(KB,(n+1))-1)
            if cB_n == 0 and cA_n != 0:
                c_ratio_n == float('inf')
                break
            elif cB_n == 0 and cA_n == 0:
                c_ratio_n = 0
            else:
                c_ratio_n = cA_n/cB_n

        cA_ext = cA_0 - cA_n
        cB_ext = cB_0 - cB_n

        cA_m = cA_ext
        cB_m = cB_ext

        if cA_m != 0:
            while cB_m/cA_m < c_ratio_t:
                m += 1
                cA_m = cA_ext*(1/KA-1)/(np.power(1/KA,(m+1))-1)
                cB_m = cB_ext*(1/KB-1)/(np.power(1/KB,(m+1))-1)
                if cA_m == 0 and cB_m != 0:
                    c_ratio_m == float('inf')
                    break
                elif cA_m == 0 and cB_m == 0:
                    c_ratio_m = 0
                else:
                    c_ratio_m = cB_m/cA_m
        else:
            c_ratio_m == float('inf')

        rec_A = cA_n/cA_0
        rec_B = cB_m/cB_0

    elif just_impurity == False and case == 'B-standard':
        while cB_n/cA_n < c_ratio_t:
            n += 1
            cA_n = cA_0*(KA-1)/(np.power(KA,(n+1))-1)
            cB_n = cB_0*(KB-1)/(np.power(KB,(n+1))-1)
            if cA_n == 0 and cB_n != 0:
                c_ratio_n == float('inf')
                break
            elif cB_n == 0 and cA_n == 0:
                c_ratio_n = 0
            else:
                c_ratio_n = cB_n/cA_n

        cA_ext = cA_0 - cA_n
        cB_ext = cB_0 - cB_n

        cA_m = cA_ext
        cB_m = cB_ext
        
        if cB_m != 0:
            while cA_m/cB_m < c_ratio_t:
                m += 1
                cA_m = cA_ext*(1/KA-1)/(np.power(1/KA,(m+1))-1)
                cB_m = cB_ext*(1/KB-1)/(np.power(1/KB,(m+1))-1)
                if cB_m == 0 and cA_m != 0:
                    c_ratio_m == float('inf')
                    break
                elif cB_m == 0 and cA_m == 0:
                    c_ratio_m = 0
                else:
                    c_ratio_m = cA_m/cB_m
        else:
            c_ratio_m == float('inf')

        rec_A = cA_m/cA_0
        rec_B = cB_n/cB_0

    c_ratio_1_final = c_ratio_n
    c_ratio_2_final = c_ratio_m
    cA_1_final = cA_n
    cB_1_final = cB_n
    cA_2_final = cA_m
    cB_2_final = cB_m
    recovery_A = rec_A
    recovery_B = rec_B

    return cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B


def extraction_wrapper(c0,c_ratio_t,A_logPhi,B_logPhi,just_impurity=False):
    '''
    Wrapper function that chooses the appropriate extraction process (multi-stage batch or continuous counter-current)
    For impurities main solute is in index 0

    Phi = c(extractor)/c(original)
    '''

    # Choose case
    if A_logPhi == B_logPhi:
        print('Extraction error: equal logPhi')
        return 'Null-standard', 0, 0, 0, 0, 0, 0, 0, 0

    if A_logPhi < B_logPhi:
        case = 'A-standard'
    elif B_logPhi < A_logPhi:
        case = 'B-standard'
    
    KA = np.power(10,A_logPhi)
    KB = np.power(10,B_logPhi)

    try:
        if A_logPhi * B_logPhi < 0:
            cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = counter_current_extraction(case,c0,c_ratio_t,KA,KB,just_impurity=just_impurity)
        elif just_impurity and A_logPhi*B_logPhi > 0 and KA > 1 and case == 'A-standard':
            cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = counter_current_extraction(case,c0,c_ratio_t,KA,KB,just_impurity=just_impurity)
        elif just_impurity and A_logPhi*B_logPhi > 0 and KB < 1 and case == 'B-standard':
            cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = counter_current_extraction(case,c0,c_ratio_t,KA,KB,just_impurity=just_impurity)
        else:
            cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = extraction(case,c0,c_ratio_t,KA,KB)
    except RuntimeWarning:
        cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B = 0, 0, 1, 0, 0, 1, 0, 0
    
    if math.isnan(cA_1_final):
        cA_1_final = 0
    if math.isnan(cB_1_final):
        cB_1_final = 0
    if math.isnan(c_ratio_1_final):
        c_ratio_1_final = 1
    if math.isnan(cA_2_final):
        cA_2_final = 0
    if math.isnan(cB_2_final):
        cB_2_final = 0
    if math.isnan(c_ratio_2_final):
        c_ratio_2_final = 1
    if math.isnan(recovery_A):
        recovery_A = 0
    if math.isnan(recovery_B):
        recovery_B = 0

    return case, cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, recovery_A, recovery_B


################# Main #################

def solute_separation_energy(cfeed,c_ratio_t,pm,target_is_max=True):
    '''
    Index 0: solute A
    Index 1: solute B
    '''
    extractor = solvent_property('extractor',pm['solvent'])

    if extractor == 'Heptane':
        A_logPhi = pm['logPhi_heptane'][0]
        B_logPhi = pm['logPhi_heptane'][1]
    elif extractor == 'Water':
        A_logPhi = pm['logPhi_water'][0]
        B_logPhi = pm['logPhi_water'][1]
    
    # Extraction for ternary separation
    case, cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, ext_recovery_A, ext_recovery_B = extraction_wrapper(cfeed,c_ratio_t,A_logPhi,B_logPhi,just_impurity=False)
    
    # Nanofiltration for ternary separation
    tnf_specific_energy_ret, tnf_recovery_ret, tnf_area_ret, tnf_n_stages_ret, tnf_c_final_ret, tnf_stages_ret = targeted_ternary_retentate_nf_cascade(cfeed,c_ratio_t,pm)
    tnf_specific_energy_per, tnf_recovery_per, tnf_area_per, tnf_n_stages_per, tnf_c_final_per, tnf_stages_per = targeted_ternary_permeate_nf_cascade(cfeed,c_ratio_t,pm)

    # Linking solutes and streams
    if case == 'A-standard':
        cA_from_ext = cA_1_final
        cB_from_ext = cB_2_final
    elif case == 'B-standard':
        cA_from_ext = cA_2_final
        cB_from_ext = cB_1_final
    
    if pm['R'][0] > pm['R'][1]:
        cA_from_tnf = tnf_c_final_ret
        cB_from_tnf = tnf_c_final_per
        nA_tnf = tnf_n_stages_ret
        nB_tnf = tnf_n_stages_per
        areaA_tnf = tnf_area_ret
        areaB_tnf = tnf_area_per
        E_ternaryA_tnf = tnf_specific_energy_ret
        E_ternaryB_tnf = tnf_specific_energy_per
        recoveryA_tnf = tnf_recovery_ret
        recoveryB_tnf = tnf_recovery_per
    elif pm['R'][0] < pm['R'][1]:
        cA_from_tnf = tnf_c_final_per
        cB_from_tnf = tnf_c_final_ret
        nA_tnf = tnf_n_stages_per
        nB_tnf = tnf_n_stages_ret
        areaA_tnf = tnf_area_per
        areaB_tnf = tnf_area_ret
        E_ternaryA_tnf = tnf_specific_energy_per
        E_ternaryB_tnf = tnf_specific_energy_ret
        recoveryA_tnf = tnf_recovery_per
        recoveryB_tnf = tnf_recovery_ret

    # Binary concentration target
    if target_is_max:
        c_target_A = min(max(cA_from_ext,cB_from_ext,cA_from_tnf,cB_from_tnf),pm['solubility'][0])
        c_target_B = min(c_target_A,pm['solubility'][1])
    else:
        c_target_A = cfeed[0]
        c_target_B = cfeed[1]

    # Concentration processes after extraction
    specific_energy_evap_after_ext = evaporation_energy(cA_from_ext,c_target_A,pm,index=0) + evaporation_energy(cB_from_ext,c_target_B,pm,index=1)

    specific_energy_A_cpld_after_ext, recovery_A_cpld_after_ext, area_A_cpld_after_ext, c_shift_A_cpld_after_ext, nf_specific_energy_A_cpld_after_ext, eva_specific_energy_A_cpld_after_ext, n_stages_A_cpld_after_ext = coupled_binary_energy(cA_from_ext,c_target_A,pm,index=0)
    specific_energy_B_cpld_after_ext, recovery_B_cpld_after_ext, area_B_cpld_after_ext, c_shift_B_cpld_after_ext, nf_specific_energy_B_cpld_after_ext, eva_specific_energy_B_cpld_after_ext, n_stages_B_cpld_after_ext = coupled_binary_energy(cB_from_ext,c_target_B,pm,index=1)

    if pm['R'][0] > 0:
        specific_energy_A_nf_after_ext, recovery_A_nf_after_ext, area_A_nf_after_ext, n_stages_A_nf_after_ext, _ = targeted_binary_retentate_nf_cascade(cA_from_ext,c_target_A,pm,index=0)
    else:
        specific_energy_A_nf_after_ext, recovery_A_nf_after_ext, area_A_nf_after_ext, n_stages_A_nf_after_ext, _ = targeted_binary_permeate_nf_cascade(cA_from_ext,c_target_A,pm,index=0)
    if pm['R'][1] > 0:
        specific_energy_B_nf_after_ext, recovery_B_nf_after_ext, area_B_nf_after_ext, n_stages_B_nf_after_ext, _ = targeted_binary_retentate_nf_cascade(cB_from_ext,c_target_B,pm,index=1)
    else:
        specific_energy_B_nf_after_ext, recovery_B_nf_after_ext, area_B_nf_after_ext, n_stages_B_nf_after_ext, _ = targeted_binary_permeate_nf_cascade(cB_from_ext,c_target_B,pm,index=1)
    
    # Concentration processes after ternary nanofiltration

    specific_energy_A_eva_after_nf = evaporation_energy(cA_from_tnf,c_target_A,pm,index=0)
    specific_energy_B_eva_after_nf = evaporation_energy(cB_from_tnf,c_target_B,pm,index=1)

    specific_energy_A_cpld_after_nf, _, nf_specific_energy_A_cpld_after_nf, eva_specific_energy_A_cpld_after_nf, recovery_A_cpld_after_nf, n_stages_A_cpld_after_nf, area_A_cpld_after_nf, c_shift_A = coupled_binary_energy_after_ternary(E_ternaryA_tnf,cA_from_tnf,c_target_A,0,pm)
    specific_energy_B_cpld_after_nf, _, nf_specific_energy_B_cpld_after_nf, eva_specific_energy_B_cpld_after_nf, recovery_B_cpld_after_nf, n_stages_B_cpld_after_nf, area_B_cpld_after_nf, c_shift_B = coupled_binary_energy_after_ternary(E_ternaryB_tnf,cB_from_tnf,c_target_B,1,pm)

    try:
        if pm['R'][0] > 0:
            specific_energy_A_nf_after_nf, recovery_A_nf_after_nf, area_A_nf_after_nf, n_stages_A_nf_after_nf, _ = targeted_binary_retentate_nf_cascade(cA_from_tnf,c_target_A,pm,index=0)
        else:
            specific_energy_A_nf_after_nf, recovery_A_nf_after_nf, area_A_nf_after_nf, n_stages_A_nf_after_nf, _ = targeted_binary_permeate_nf_cascade(cA_from_tnf,c_target_A,pm,index=0)
    except:
        specific_energy_A_nf_after_nf = float('inf')
        recovery_A_nf_after_nf = 0
        area_A_nf_after_nf = float('inf')
        n_stages_A_nf_after_nf = float('inf')
    
    try:
        if pm['R'][1] > 0:
            specific_energy_B_nf_after_nf, recovery_B_nf_after_nf, area_B_nf_after_nf, n_stages_B_nf_after_nf, _ = targeted_binary_retentate_nf_cascade(cB_from_tnf,c_target_B,pm,index=1)
        else:
            specific_energy_B_nf_after_nf, recovery_B_nf_after_nf, area_B_nf_after_nf, n_stages_B_nf_after_nf, _ = targeted_binary_permeate_nf_cascade(cB_from_tnf,c_target_B,pm,index=1)
    except:
        specific_energy_B_nf_after_nf = float('inf')
        recovery_B_nf_after_nf = 0
        area_B_nf_after_nf = float('inf')
        n_stages_B_nf_after_nf = float('inf')

    # Summarizing energies

    specific_energies = {}

    specific_energies['ext-eva'] = specific_energy_evap_after_ext

    specific_energies['ext-nf']  = specific_energy_A_nf_after_ext + specific_energy_B_nf_after_ext

    specific_energies['ext-cpld']  = specific_energy_A_cpld_after_ext + specific_energy_B_cpld_after_ext
    specific_energies['ext-cpld (nf)']  = nf_specific_energy_A_cpld_after_ext + nf_specific_energy_B_cpld_after_ext
    specific_energies['ext-cpld (eva)'] = eva_specific_energy_A_cpld_after_ext + eva_specific_energy_B_cpld_after_ext

    specific_energies['nf_ternary'] = E_ternaryA_tnf + E_ternaryB_tnf
    specific_energies['nf-eva'] = E_ternaryA_tnf + E_ternaryB_tnf + specific_energy_A_eva_after_nf + specific_energy_B_eva_after_nf

    if recovery_A_nf_after_nf == 0 or recovery_B_nf_after_nf == 0:
        specific_energies['nf-nf'] = float('inf')
        specific_energies['nf-nf (ternary)'] = float('inf')
    else:
        specific_energies['nf-nf'] = E_ternaryA_tnf/recovery_A_nf_after_nf + E_ternaryB_tnf/recovery_B_nf_after_nf + specific_energy_A_nf_after_nf + specific_energy_B_nf_after_nf
        specific_energies['nf-nf (ternary)'] = E_ternaryA_tnf/recovery_A_nf_after_nf + E_ternaryB_tnf/recovery_B_nf_after_nf

    if recovery_A_cpld_after_nf == 0 or recovery_B_cpld_after_nf == 0:
        specific_energies['nf-cpld'] = float('inf')
        specific_energies['nf-cpld (ternary)'] = float('inf')
        specific_energies['nf-cpld (nf)'] = float('inf')
        specific_energies['nf-cpld (eva)'] = eva_specific_energy_A_cpld_after_nf + eva_specific_energy_B_cpld_after_nf
    else:
        specific_energies['nf-cpld'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + E_ternaryB_tnf/recovery_B_cpld_after_nf + specific_energy_A_cpld_after_nf + specific_energy_B_cpld_after_nf
        specific_energies['nf-cpld (ternary)'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + E_ternaryB_tnf/recovery_B_cpld_after_nf
        specific_energies['nf-cpld (nf)'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + nf_specific_energy_A_cpld_after_nf + E_ternaryB_tnf/recovery_B_cpld_after_nf + nf_specific_energy_B_cpld_after_nf
        specific_energies['nf-cpld (eva)'] = eva_specific_energy_A_cpld_after_nf + eva_specific_energy_B_cpld_after_nf

    # Summarizing average recoveries

    recoveries = {}

    recoveries['ext-eva'] = (ext_recovery_A + ext_recovery_B)/2
    recoveries['ext-nf'] = (ext_recovery_A*recovery_A_nf_after_ext + ext_recovery_B*recovery_B_nf_after_ext)/2
    recoveries['ext-cpld'] = (ext_recovery_A*recovery_A_cpld_after_ext + ext_recovery_B*recovery_B_cpld_after_ext)/2

    recoveries['nf-eva'] = (recoveryA_tnf + recoveryB_tnf)/2
    recoveries['nf-nf'] = (recoveryA_tnf*recovery_A_nf_after_nf + recoveryB_tnf*recovery_B_nf_after_nf)/2
    recoveries['nf-cpld'] = (recoveryA_tnf*recovery_A_cpld_after_nf + recoveryB_tnf*recovery_B_cpld_after_nf)/2

    # Summarizing stages

    no_of_stages = {}

    no_of_stages['ext-eva'] = 0
    no_of_stages['ext-nf'] = n_stages_A_nf_after_ext + n_stages_B_nf_after_ext
    no_of_stages['ext-cpld'] = n_stages_A_cpld_after_ext + n_stages_B_cpld_after_ext

    no_of_stages['nf-eva'] = nA_tnf + nB_tnf
    no_of_stages['nf-nf'] = nA_tnf + nB_tnf + n_stages_A_nf_after_nf + n_stages_B_nf_after_nf
    no_of_stages['nf-cpld'] = nA_tnf + nB_tnf + n_stages_A_cpld_after_nf + n_stages_B_cpld_after_nf

    # Areas

    areas = {}

    areas['ext-eva'] = 0
    areas['ext-nf'] = area_A_nf_after_ext + area_B_nf_after_ext
    areas['ext-cpld'] = area_A_cpld_after_ext + area_B_cpld_after_ext
    areas['nf-eva'] = areaA_tnf + areaB_tnf
    areas['nf-nf'] = areaA_tnf + areaB_tnf + area_A_nf_after_nf + area_B_nf_after_nf
    areas['nf-cpld'] = areaA_tnf + areaB_tnf + area_A_cpld_after_nf + area_B_cpld_after_nf

    for key in specific_energies:
        if specific_energies[key] == float('nan'):
            specific_energies[key] == float('inf')
    for key in recoveries:
        if recoveries[key] == float('nan'):
            recoveries[key] == 0
    for key in no_of_stages:
        if no_of_stages[key] == float('nan'):
            no_of_stages[key] == float('inf')
    for key in areas:
        if areas[key] == float('nan'):
            areas[key] == float('inf')

    return specific_energies, recoveries, no_of_stages, areas


def impurity_removal_energy(cfeed,c_ratio_t,pm,target_is_max=True):
    '''
    Index 0 / A: MAIN solute
    Index 1 / B: IMPURITY solute
    '''
    extractor = solvent_property('extractor',pm['solvent'])

    if extractor == 'Heptane':
        A_logPhi = pm['logPhi_heptane'][0]
        B_logPhi = pm['logPhi_heptane'][1]
    elif extractor == 'Water':
        A_logPhi = pm['logPhi_water'][0]
        B_logPhi = pm['logPhi_water'][1]
    
    # Extraction for ternary separation
    case, cA_1_final, cB_1_final, c_ratio_1_final, cA_2_final, cB_2_final, c_ratio_2_final, ext_recovery_A, ext_recovery_B = extraction_wrapper(cfeed,c_ratio_t,A_logPhi,B_logPhi,just_impurity=True)

    #print(cfeed)
    # Nanofiltration for ternary separation
    if pm['R'][0] > pm['R'][1]:
        tnf_specific_energy, tnf_recovery, tnf_area, tnf_n_stages, tnf_c_final, tnf_stages = targeted_ternary_retentate_nf_cascade(cfeed,c_ratio_t,pm)
    elif pm['R'][1] > pm['R'][0]:
        tnf_specific_energy, tnf_recovery, tnf_area, tnf_n_stages, tnf_c_final, tnf_stages = targeted_ternary_permeate_nf_cascade(cfeed,c_ratio_t,pm)

    # Linking solutes and streams
    if case == 'A-standard':
        cA_from_ext = cA_1_final
    elif case == 'B-standard':
        cA_from_ext = cA_2_final
    
    cA_from_tnf = tnf_c_final
    nA_tnf = tnf_n_stages
    areaA_tnf = tnf_area
    E_ternaryA_tnf = tnf_specific_energy
    recoveryA_tnf = tnf_recovery

    # Binary concentration target
    if target_is_max:
        c_target_A = min(max(cA_from_ext,cA_from_tnf),pm['solubility'][0])
    else:
        c_target_A = cfeed[0]
        

    # Concentration processes after extraction
    specific_energy_evap_after_ext = evaporation_energy(cA_from_ext,c_target_A,pm,index=0)

    specific_energy_A_cpld_after_ext, recovery_A_cpld_after_ext, area_A_cpld_after_ext, c_shift_A_cpld_after_ext, nf_specific_energy_A_cpld_after_ext, eva_specific_energy_A_cpld_after_ext, n_stages_A_cpld_after_ext = coupled_binary_energy(cA_from_ext,c_target_A,pm,index=0)

    if pm['R'][0] > 0:
        specific_energy_A_nf_after_ext, recovery_A_nf_after_ext, area_A_nf_after_ext, n_stages_A_nf_after_ext, _ = targeted_binary_retentate_nf_cascade(cA_from_ext,c_target_A,pm,index=0)
    else:
        specific_energy_A_nf_after_ext, recovery_A_nf_after_ext, area_A_nf_after_ext, n_stages_A_nf_after_ext, _ = targeted_binary_permeate_nf_cascade(cA_from_ext,c_target_A,pm,index=0)
    
    # Concentration processes after ternary nanofiltration

    specific_energy_A_eva_after_nf = evaporation_energy(cA_from_tnf,c_target_A,pm,index=0)

    specific_energy_A_cpld_after_nf, _, nf_specific_energy_A_cpld_after_nf, eva_specific_energy_A_cpld_after_nf, recovery_A_cpld_after_nf, n_stages_A_cpld_after_nf, area_A_cpld_after_nf, c_shift_A = coupled_binary_energy_after_ternary(E_ternaryA_tnf,cA_from_tnf,c_target_A,0,pm)

    try:
        if pm['R'][0] > 0:
            specific_energy_A_nf_after_nf, recovery_A_nf_after_nf, area_A_nf_after_nf, n_stages_A_nf_after_nf, _ = targeted_binary_retentate_nf_cascade(cA_from_tnf,c_target_A,pm,index=0)
        else:
            specific_energy_A_nf_after_nf, recovery_A_nf_after_nf, area_A_nf_after_nf, n_stages_A_nf_after_nf, _ = targeted_binary_permeate_nf_cascade(cA_from_tnf,c_target_A,pm,index=0)
    except:
        specific_energy_A_nf_after_nf = float('inf')
        recovery_A_nf_after_nf = 0
        area_A_nf_after_nf = float('inf')
        n_stages_A_nf_after_nf = float('inf')
    

    # Summarizing energies

    specific_energies = {}

    specific_energies['ext-eva'] = specific_energy_evap_after_ext

    specific_energies['ext-nf']  = specific_energy_A_nf_after_ext

    specific_energies['ext-cpld']  = specific_energy_A_cpld_after_ext
    specific_energies['ext-cpld (nf)']  = nf_specific_energy_A_cpld_after_ext
    specific_energies['ext-cpld (eva)'] = eva_specific_energy_A_cpld_after_ext

    specific_energies['nf_ternary'] = E_ternaryA_tnf
    specific_energies['nf-eva'] = E_ternaryA_tnf + specific_energy_A_eva_after_nf

    if recovery_A_nf_after_nf == 0:
        specific_energies['nf-nf'] = float('inf')
        specific_energies['nf-nf (ternary)'] = float('inf')
    else:
        specific_energies['nf-nf'] = E_ternaryA_tnf/recovery_A_nf_after_nf + specific_energy_A_nf_after_nf
        specific_energies['nf-nf (ternary)'] = E_ternaryA_tnf/recovery_A_nf_after_nf

    if recovery_A_cpld_after_nf == 0: 
        specific_energies['nf-cpld'] = float('inf')
        specific_energies['nf-cpld (ternary)'] = float('inf')
        specific_energies['nf-cpld (nf)'] = float('inf')
        specific_energies['nf-cpld (eva)'] = eva_specific_energy_A_cpld_after_nf 
    else: 
        specific_energies['nf-cpld'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + specific_energy_A_cpld_after_nf
        specific_energies['nf-cpld (ternary)'] = E_ternaryA_tnf/recovery_A_cpld_after_nf
        specific_energies['nf-cpld (nf)'] = E_ternaryA_tnf/recovery_A_cpld_after_nf + nf_specific_energy_A_cpld_after_nf
        specific_energies['nf-cpld (eva)'] = eva_specific_energy_A_cpld_after_nf 

    # Summarizing average recoveries

    recoveries = {}

    recoveries['ext-eva'] = ext_recovery_A
    recoveries['ext-nf'] = ext_recovery_A*recovery_A_nf_after_ext
    recoveries['ext-cpld'] = ext_recovery_A*recovery_A_cpld_after_ext

    recoveries['nf-eva'] = recoveryA_tnf
    recoveries['nf-nf'] = recoveryA_tnf*recovery_A_nf_after_nf
    recoveries['nf-cpld'] = recoveryA_tnf*recovery_A_cpld_after_nf

    # Summarizing stages

    no_of_stages = {}

    no_of_stages['ext-eva'] = 0
    no_of_stages['ext-nf'] = n_stages_A_nf_after_ext
    no_of_stages['ext-cpld'] = n_stages_A_cpld_after_ext

    no_of_stages['nf-eva'] = nA_tnf
    no_of_stages['nf-nf'] = nA_tnf + n_stages_A_nf_after_nf
    no_of_stages['nf-cpld'] = nA_tnf + n_stages_A_cpld_after_nf

    # Areas

    areas = {}

    areas['ext-eva'] = 0
    areas['ext-nf'] = area_A_nf_after_ext
    areas['ext-cpld'] = area_A_cpld_after_ext
    areas['nf-eva'] = areaA_tnf
    areas['nf-nf'] = areaA_tnf + area_A_nf_after_nf
    areas['nf-cpld'] = areaA_tnf + area_A_cpld_after_nf

    for key in specific_energies:
        if specific_energies[key] == float('nan'):
            specific_energies[key] == float('inf')
    for key in recoveries:
        if recoveries[key] == float('nan'):
            recoveries[key] == 0
    for key in no_of_stages:
        if no_of_stages[key] == float('nan'):
            no_of_stages[key] == float('inf')
    for key in areas:
        if areas[key] == float('nan'):
            areas[key] == float('inf')

    return specific_energies, recoveries, no_of_stages, areas



################# Technology selection #################

def energy_reduction_calculation(e_ref,e_novel):
    if e_ref == 0:
        return 0
    elif e_ref == float('inf') and e_novel != e_ref:
        return 1
    elif e_ref == float('inf') and e_novel == float('inf'):
        return 0
    else:
        return max(0,(e_ref-e_novel) / e_ref)

def is_equal(A,B):
    if A == float('inf') or B == float('inf'):
        return False
    elif B == 0 and A == 0:
        return True
    elif B == 0 and A != 0:
        return False
    
    if abs(A-B)/B < 0.001:
        return True
    else:
        return False

def choose_best_concentration_technology(sample_coupled_energy,sample_evap_energy,sample_nf_energy):
    best = 'none'
    if sample_coupled_energy == min(sample_coupled_energy,sample_evap_energy,sample_nf_energy) and sample_coupled_energy != sample_evap_energy and sample_coupled_energy != sample_nf_energy:
        best = 'coupled'
    elif sample_evap_energy == min(sample_coupled_energy,sample_evap_energy,sample_nf_energy):
        best = 'evaporation'
    elif sample_nf_energy == min(sample_coupled_energy,sample_evap_energy,sample_nf_energy):
        best = 'nanofiltration'
    return best

def choose_best_recovery_technology(sample_evap_energy,sample_nf_energy):
    best = 'none'
    if sample_evap_energy == min(sample_evap_energy,sample_nf_energy):
        best = 'evaporation'
    elif sample_nf_energy == min(sample_evap_energy,sample_nf_energy):
        best = 'nanofiltration'
    return best

def choose_best_ternary_technology(specific_energies):
    best_ternary = 'none'
    best_configuration = 'none'
    lowest_energy = specific_energies['nf-cpld']
    if specific_energies['nf-cpld'] == float('inf') and specific_energies['ext-cpld'] == float('inf'):
        best_ternary = 'none'
        best_configuration = 'none'
    elif specific_energies['nf-cpld'] == float('nan') and specific_energies['ext-cpld'] == float('nan'):
        best_ternary = 'none'
        best_configuration = 'none'       
    elif specific_energies['nf-cpld'] < specific_energies['ext-cpld']:
        best_ternary = 'nanofiltration'
        if is_equal(specific_energies['nf-cpld'],specific_energies['nf-cpld (ternary)']):
            best_configuration = 'nanofiltration-none'
        elif is_equal(specific_energies['nf-cpld'],specific_energies['nf-nf']):
            best_configuration = 'nanofiltration-nanofiltration'
        elif is_equal(specific_energies['nf-cpld'],specific_energies['nf-eva']):
            best_configuration = 'nanofiltration-evaporation'
        elif specific_energies['nf-cpld'] < specific_energies['nf-nf'] and specific_energies['nf-cpld'] < specific_energies['nf-eva']:
            best_configuration = 'nanofiltration-coupled'
        elif (np.isnan(specific_energies['nf-nf']) or np.isnan(specific_energies['nf-eva'])) and not np.isnan(specific_energies['nf-cpld']):
            best_configuration = 'nanofiltration-coupled'
        else:
            best_configuration = 'none'
    else:
        best_ternary = 'extraction'
        lowest_energy = specific_energies['ext-cpld']
        if is_equal(specific_energies['ext-cpld'],0.0):
            best_configuration = 'extraction-none'
        elif is_equal(specific_energies['ext-cpld'],specific_energies['ext-nf']):
            best_configuration = 'extraction-nanofiltration'
        elif is_equal(specific_energies['ext-cpld'],specific_energies['ext-eva']):
            best_configuration = 'extraction-evaporation'
        elif specific_energies['ext-cpld'] < specific_energies['ext-nf'] and specific_energies['ext-cpld'] < specific_energies['ext-eva']:
            best_configuration = 'extraction-coupled'
        elif (np.isnan(specific_energies['ext-nf']) or np.isnan(specific_energies['ext-eva'])) and not np.isnan(specific_energies['ext-cpld']):
            best_configuration = 'extraction-coupled'
        else:
            best_configuration = 'none'
    
    return best_ternary, best_configuration, lowest_energy


################# Sustainability calculations #################

def binary_cost_and_co2_calculation(energy_heat_ref,energy_heat_opt,energy_elec_opt,area_opt,cost_heat,cost_elec,cost_membrane,co2eq_heat,co2eq_elec,co2eq_membrane,membrane_lifetime,prod_rate):
    '''
    cost has to be in USD/kWh and USD/m2
    co2eq has to be in kg/kWh and kg/m2
    membrane lifetime in h
    production rate in kg/h

    kg product per membrane [kg/1]: lifetime*prod_rate
    membrane per kg prod [1/kg]: 1 / (lifetime*prod_rate)
    m2 membrane per kg prod [m2/kg]: total_area / (lifetime*prod_rate)

    array-compatible
    '''
    # CONVERTING TO JOULE REFERENCE
    cost_heat = cost_heat / 3600000
    cost_elec = cost_elec / 3600000
    co2eq_heat = co2eq_heat / 3600000
    co2eq_elec = co2eq_elec / 3600000

    # CALCULATIONS

    membrane_cost_per_kg_product = (cost_membrane * area_opt) / (membrane_lifetime * prod_rate)
    membrane_co2eq_per_kg_product = (co2eq_membrane * area_opt) / (membrane_lifetime * prod_rate)

    reference_co2 = co2eq_heat*energy_heat_ref
    novel_co2 = co2eq_heat*energy_heat_opt + co2eq_elec*energy_elec_opt + membrane_co2eq_per_kg_product
    co2eq_reduction = np.divide((reference_co2-novel_co2),reference_co2)

    reference_cost = cost_heat*energy_heat_ref
    novel_cost = cost_heat*energy_heat_opt + cost_elec*energy_elec_opt + membrane_cost_per_kg_product
    cost_reduction = np.divide((reference_cost-novel_cost),reference_cost)

    single = False
    if not isinstance(co2eq_reduction,np.ndarray):
        single = True
        co2eq_reduction = [co2eq_reduction]
        cost_reduction = [cost_reduction]
        novel_co2 = [novel_co2]
        novel_cost = [novel_cost]
        reference_co2 = [reference_co2]
        reference_cost = [reference_cost]

    i = 0
    for _ in co2eq_reduction:
        if co2eq_reduction[i] < 0:
            co2eq_reduction[i] = 0
        elif is_equal(novel_co2[i],0.0) and is_equal(reference_co2[i],0.0):
            co2eq_reduction[i] = 0
        elif np.isnan(co2eq_reduction[i]) and novel_co2[i] == float('inf'):
            co2eq_reduction[i] = 0
        elif np.isnan(co2eq_reduction[i]) and np.isnan(novel_co2[i]):
            co2eq_reduction[i] = 0
        elif np.isnan(co2eq_reduction[i]) and novel_co2[i] != float('inf') and not np.isnan(novel_co2[i]):
            co2eq_reduction[i] = 1
        if np.isnan(co2eq_reduction[i]) or co2eq_reduction[i] == float('inf'):
            co2eq_reduction[i] = 0
        i += 1

    i = 0
    for _ in cost_reduction:
        if cost_reduction[i] < 0:
            cost_reduction[i] = 0
        elif is_equal(novel_cost[i],0.0) and is_equal(reference_cost[i],0.0):
            cost_reduction[i] = 0
        elif np.isnan(cost_reduction[i]) and novel_cost[i] == float('inf'):
            cost_reduction[i] = 0
        elif np.isnan(cost_reduction[i]) and np.isnan(novel_cost[i]):
            cost_reduction[i] = 0
        elif np.isnan(cost_reduction[i]) and novel_cost[i] != float('inf') and not np.isnan(novel_cost[i]):
            cost_reduction[i] = 1
        
        if np.isnan(cost_reduction[i]) or cost_reduction[i] == float('inf'):
            cost_reduction[i] = 0
        i += 1
    
    if single:
        co2eq_reduction = co2eq_reduction[0]
        cost_reduction = cost_reduction[0]

    return cost_reduction, co2eq_reduction

def ternary_cost_and_co2_calculation(best_ternary,nf_area,ext_area,energy_heat_ref,energy_heat_opt,energy_elec_opt,cost_heat,cost_elec,cost_membrane,co2eq_heat,co2eq_elec,co2eq_membrane,membrane_lifetime,prod_rate):
    '''
    cost has to be in USD/kWh and USD/m2
    co2eq has to be in kg/kWh and kg/m2
    membrane lifetime in h
    production rate in kg/h

    kg product per membrane [kg/1]: lifetime*prod_rate
    membrane per kg prod [1/kg]: 1 / (lifetime*prod_rate)
    m2 membrane per kg prod [m2/kg]: total_area / (lifetime*prod_rate)

    array-compatible
    '''

    # CONVERTING TO JOULE REFERENCE
    cost_heat = cost_heat / 3600000
    cost_elec = cost_elec / 3600000
    co2eq_heat = co2eq_heat / 3600000
    co2eq_elec = co2eq_elec / 3600000

    # AREAS

    try:
        area_opt = np.zeros(len(best_ternary))
        for i in range(len(best_ternary)):
            if best_ternary.iloc[i] == 'nanofiltration':
                area_opt[i] = nf_area.iloc[i]
            else:
                area_opt[i] = ext_area.iloc[i]
    except AttributeError:
        area_opt = 0
        if best_ternary == 'nanofiltration':
            area_opt = nf_area
        else:
            area_opt = ext_area

    # CALCULATIONS

    membrane_cost_per_kg_product = (cost_membrane * area_opt) / (membrane_lifetime * prod_rate)
    membrane_co2eq_per_kg_product = (co2eq_membrane * area_opt) / (membrane_lifetime * prod_rate)

    reference_co2 = co2eq_heat*energy_heat_ref
    novel_co2 = co2eq_heat*energy_heat_opt + co2eq_elec*energy_elec_opt + membrane_co2eq_per_kg_product
    co2eq_reduction = np.divide((reference_co2-novel_co2),reference_co2)

    reference_cost = cost_heat*energy_heat_ref
    novel_cost = cost_heat*energy_heat_opt + cost_elec*energy_elec_opt + membrane_cost_per_kg_product
    cost_reduction = np.divide((reference_cost-novel_cost),reference_cost)

    single = False
    if not isinstance(co2eq_reduction,np.ndarray):
        single = True
        co2eq_reduction = [co2eq_reduction]
        cost_reduction = [cost_reduction]
        novel_co2 = [novel_co2]
        novel_cost = [novel_cost]
        reference_co2 = [reference_co2]
        reference_cost = [reference_cost]

    i = 0
    for _ in co2eq_reduction:
        if co2eq_reduction[i] < 0:
            co2eq_reduction[i] = 0
        elif is_equal(novel_co2[i],0.0) and is_equal(reference_co2[i],0.0):
            co2eq_reduction[i] = 0
        elif np.isnan(co2eq_reduction[i]) and novel_co2[i] == float('inf'):
            co2eq_reduction[i] = 0
        elif np.isnan(co2eq_reduction[i]) and np.isnan(novel_co2[i]):
            co2eq_reduction[i] = 0
        elif np.isnan(co2eq_reduction[i]) and novel_co2[i] != float('inf') and not np.isnan(novel_co2[i]):
            co2eq_reduction[i] = 1
        
        if np.isnan(co2eq_reduction[i]) or co2eq_reduction[i] == float('inf'):
            co2eq_reduction[i] = 0
        i += 1

    i = 0
    for _ in cost_reduction:
        if cost_reduction[i] < 0:
            cost_reduction[i] = 0
        elif is_equal(novel_cost[i],0.0) and is_equal(reference_cost[i],0.0):
            cost_reduction[i] = 0
        elif np.isnan(cost_reduction[i]) and novel_cost[i] == float('inf'):
            cost_reduction[i] = 0
        elif np.isnan(cost_reduction[i]) and np.isnan(novel_cost[i]):
            cost_reduction[i] = 0
        elif np.isnan(cost_reduction[i]) and novel_cost[i] != float('inf') and not np.isnan(novel_cost[i]):
            cost_reduction[i] = 1
        
        if np.isnan(cost_reduction[i]) or cost_reduction[i] == float('inf'):
            cost_reduction[i] = 0
        i += 1

    if single:
        co2eq_reduction = co2eq_reduction[0]
        cost_reduction = cost_reduction[0]

    return cost_reduction, co2eq_reduction

def threshold_membrane_price_calculation(membrane_lifetime,prod_rate,cost_heat,cost_elec,energy_heat_ref,energy_heat_opt,energy_elec_opt,nf_area):
    '''
    cost has to be in USD/kWh and USD/m2
    co2eq has to be in kg/kWh and kg/m2
    membrane lifetime in h
    production rate in kg/h
    '''

    # CONVERTING TO JOULE REFERENCE
    cost_heat = cost_heat / 3600000
    cost_elec = cost_elec / 3600000

    price_threshold = (membrane_lifetime*prod_rate)*(energy_heat_ref*cost_heat - energy_elec_opt*cost_elec - energy_heat_opt*cost_heat) / nf_area
    return price_threshold


################# Single calculations #################

def run_binary_separation_calculation(data):
    '''
    Calculations for a single binary separation instance.
    '''
    pm = initiate_separation_parameters(data['separation_type'],data['solvent'],data['permeance'],data['external_heat_integration'],samples=[],empty=True)
    co2eq_membrane = 5.357 # kg co2eq/m2
    production_rate = 10 # kg/h

    molar_volume1 = molar_mass_from_smiles(data['solute1_smiles']) / data['solute1_density']
    pm['L'].append(solute_permeance_from_rejection(data['solute1_rejection'], molar_volume1, data['permeance'], dp=2e6))
    pm['solubility'].append(data['solute1_solubility'])
    pm['M'].append(molar_mass_from_smiles(data['solute1_smiles']))
    pm['logPhi_water'].append(data['solute1_log_partition_water'])
    pm['logPhi_heptane'].append(data['solute1_log_partition_heptane'])
    pm['D'].append(diffusivity_from_smiles(data['solute1_smiles'], solvent_property('viscosity',data['solvent']), T=298))
    pm['nu'].append(molar_volume1)
    pm['R'].append(data['solute1_rejection'])
    pm['ns'] = 1

    evap_energy = evaporation_energy(data['solute1_feed_concentration'],data['solute1_target_concentration'],pm)

    if data['separation_type'] == 'solute_concentration':
        if pm['R'][0] >= 0:
            nf_energy, nf_recovery, nf_total_area, nf_no_stages, _ = targeted_binary_retentate_nf_cascade(data['solute1_feed_concentration'],data['solute1_target_concentration'],pm)
        else:
            nf_energy, nf_recovery, nf_total_area, nf_no_stages, _ = targeted_binary_permeate_nf_cascade(data['solute1_feed_concentration'],data['solute1_target_concentration'],pm)
        coupled_energy, coupled_recovery, coupled_total_area, coupled_shift_concentration, coupled_nf_energy, coupled_evap_energy, coupled_no_stages = coupled_binary_energy(data['solute1_feed_concentration'],data['solute1_target_concentration'],pm)

        best_technology = choose_best_concentration_technology(coupled_energy,evap_energy,nf_energy)
        if best_technology == 'nanofiltration':
            lowest_energy_demand = nf_energy
            energy_heat_opt = 0
            energy_elec_opt = nf_energy
            area_opt = nf_total_area
        elif best_technology == 'evaporation':
            lowest_energy_demand = evap_energy
            energy_heat_opt = evap_energy
            energy_elec_opt = 0
            area_opt = 0
        elif best_technology == 'coupled':
            lowest_energy_demand = coupled_energy
            energy_heat_opt = coupled_evap_energy
            energy_elec_opt = coupled_nf_energy
            area_opt = coupled_total_area
            best_technology = 'hybrid nanofiltration-evaporation, with technology shift at '+str(round(coupled_shift_concentration,2))+' mol/m3'

    elif data['separation_type'] == 'solvent_recovery':
        if pm['R'][0] >= 0:
            nf_energy, nf_recovery, nf_total_area, nf_no_stages, _ = targeted_binary_permeate_nf_cascade(data['solute1_feed_concentration'],data['solute1_target_concentration'],pm)
        else:
            nf_energy, nf_recovery, nf_total_area, nf_no_stages, _ = targeted_binary_retentate_nf_cascade(data['solute1_feed_concentration'],data['solute1_target_concentration'],pm)

        best_technology = choose_best_recovery_technology(evap_energy,nf_energy)
        if best_technology == 'nanofiltration':
            lowest_energy_demand = nf_energy
            energy_heat_opt = 0
            energy_elec_opt = nf_energy
            area_opt = nf_total_area
        elif best_technology == 'evaporation':
            lowest_energy_demand = evap_energy
            energy_heat_opt = evap_energy
            energy_elec_opt = 0
            area_opt = 0
    
    energy_reduction = energy_reduction_calculation(evap_energy,lowest_energy_demand)

    technodata_df = pd.read_csv('data/technoeconomic_data.csv')
    cost_heat = technodata_df.loc[technodata_df['country'] == data['country'], 's_h_usd_kwh'].values[0]
    cost_elec = technodata_df.loc[technodata_df['country'] == data['country'], 's_e_usd_kwh'].values[0]
    co2eq_heat = technodata_df.loc[technodata_df['country'] == data['country'], 'h_kg_co2_eq'].values[0]
    co2eq_elec = technodata_df.loc[technodata_df['country'] == data['country'], 'e_kg_co2_eq'].values[0]

    cost_reduction, co2eq_reduction = binary_cost_and_co2_calculation(evap_energy,energy_heat_opt,energy_elec_opt,area_opt,cost_heat,cost_elec,data['membrane_cost'],co2eq_heat,co2eq_elec,co2eq_membrane,data['membrane_lifetime'],production_rate)

    threshold_membrane_price = threshold_membrane_price_calculation(8766,production_rate,cost_heat,cost_elec,evap_energy,energy_heat_opt,energy_elec_opt,area_opt)

    return best_technology, lowest_energy_demand, evap_energy, energy_reduction, co2eq_reduction, cost_reduction, threshold_membrane_price


def run_ternary_separation_calculation(data):
    '''
    Calculations for a single ternary separation instance.
    '''
    pm = initiate_separation_parameters(data['separation_type'],data['solvent'],data['permeance'],data['external_heat_integration'],samples=[],empty=True)
    co2eq_membrane = 5.357 # kg co2eq/m2
    production_rate = 10 # kg/h

    molar_volume1 = molar_mass_from_smiles(data['solute1_smiles']) / data['solute1_density']
    pm['L'].append(solute_permeance_from_rejection(data['solute1_rejection'], molar_volume1, data['permeance'], dp=2e6))
    pm['solubility'].append(data['solute1_solubility'])
    pm['M'].append(molar_mass_from_smiles(data['solute1_smiles']))
    pm['logPhi_water'].append(data['solute1_log_partition_water'])
    pm['logPhi_heptane'].append(data['solute1_log_partition_heptane'])
    pm['D'].append(diffusivity_from_smiles(data['solute1_smiles'], solvent_property('viscosity',data['solvent']), T=298))
    pm['nu'].append(molar_volume1)
    pm['R'].append(data['solute1_rejection'])

    molar_volume2 = molar_mass_from_smiles(data['solute2_smiles']) / data['solute2_density']
    pm['L'].append(solute_permeance_from_rejection(data['solute2_rejection'], molar_volume2, data['permeance'], dp=2e6))
    pm['solubility'].append(data['solute2_solubility'])
    pm['M'].append(molar_mass_from_smiles(data['solute2_smiles']))
    pm['logPhi_water'].append(data['solute2_log_partition_water'])
    pm['logPhi_heptane'].append(data['solute2_log_partition_heptane'])
    pm['D'].append(diffusivity_from_smiles(data['solute2_smiles'], solvent_property('viscosity',data['solvent']), T=298))
    pm['nu'].append(molar_volume2)
    pm['R'].append(data['solute2_rejection'])
    pm['ns'] = 2

    c0 = [data['solute1_feed_concentration'], data['solute2_feed_concentration']]

    if data['separation_type'] == 'solute_separation':
        specific_energies, recoveries, no_of_stages, areas = solute_separation_energy(c0,data['solute_target_concentration_ratio'],pm,target_is_max=False)
    elif data['separation_type'] == 'impurity_removal':
        specific_energies, recoveries, no_of_stages, areas = impurity_removal_energy(c0,data['solute_target_concentration_ratio'],pm,target_is_max=False)

    best_ternary, best_configuration, lowest_energy = choose_best_ternary_technology(specific_energies)

    if best_ternary == 'nanofiltration':
        energy_heat_opt = specific_energies['nf-cpld (eva)']
        energy_elec_opt = specific_energies['nf-cpld'] - specific_energies['nf-cpld (eva)']
        area_opt = areas['nf-cpld']
        
    elif best_ternary == 'extraction':
        energy_heat_opt = specific_energies['ext-cpld (eva)']
        energy_elec_opt = specific_energies['ext-cpld (nf)']
        area_opt = areas['ext-cpld']

    else:
        energy_heat_opt = specific_energies['ext-eva']
        energy_elec_opt = 0
        area_opt = 0

    if np.isnan(energy_elec_opt):
        energy_elec_opt = float('inf')
    energy_heat_ref = specific_energies['ext-eva']

    energy_reduction = energy_reduction_calculation(specific_energies['ext-eva'],lowest_energy)

    technodata_df = pd.read_csv('data/technoeconomic_data.csv')
    cost_heat = technodata_df.loc[technodata_df['country'] == data['country'], 's_h_usd_kwh'].values[0]
    cost_elec = technodata_df.loc[technodata_df['country'] == data['country'], 's_e_usd_kwh'].values[0]
    co2eq_heat = technodata_df.loc[technodata_df['country'] == data['country'], 'h_kg_co2_eq'].values[0]
    co2eq_elec = technodata_df.loc[technodata_df['country'] == data['country'], 'e_kg_co2_eq'].values[0]

    cost_reduction, co2eq_reduction = ternary_cost_and_co2_calculation(best_ternary,areas['nf-cpld'],areas['ext-cpld'],energy_heat_ref,energy_heat_opt,energy_elec_opt,cost_heat,cost_elec,data['membrane_cost'],co2eq_heat,co2eq_elec,co2eq_membrane,data['membrane_lifetime'],production_rate)

    threshold_membrane_price = threshold_membrane_price_calculation(8766,production_rate,cost_heat,cost_elec,energy_heat_ref,energy_heat_opt,energy_elec_opt,area_opt)

    return best_configuration, lowest_energy, energy_heat_ref, energy_reduction, co2eq_reduction, cost_reduction, threshold_membrane_price