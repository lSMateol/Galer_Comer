import random
import math
import numpy as np

def create_initial_population(population_size, gene_definitions, user_inputs, constants):
    """
    Genera una población inicial para un algoritmo genético con genes multivaluados,
    considerando dependencias, valores de usuario y constantes.
    
    Args:
        population_size (int): El número de cromosomas en la población.
        gene_definitions (list): Una lista de definiciones para cada gen.
        user_inputs (dict): Diccionario con los valores fijos del usuario.
        constants (dict): Diccionario con las constantes del problema.
    
    Returns:
        Una lista de cromosomas únicos, donde cada cromosoma es una lista de valores
        elegidos basándose en las definiciones de los genes y las dependencias.
    """
    population = []
    
    # Mapeo de nombres de genes a sus índices para facilitar el acceso
    gene_map = {
        'g_TamLot': 0, 'g_TamCom': 1, 'g_TaZoCo': 2, 'g_AreVer': 3, 'g_TaCiPa': 4,
        'g_TaZoAu': 5, 'b_CanPri': 24, 'b_CanSec': 25, 'g_TamPar': 'constant',
        'g_TaUtPa': 'constant'
    }

    max_attempts = 1000
    attempts = 0

    while len(population) < population_size and attempts < max_attempts:
        chromosome = [None] * len(gene_definitions)

        # 1. Asignar los valores fijos del usuario
        for name, value in user_inputs.items():
            if name in gene_map:
                chromosome[gene_map[name]] = value
        
        # 2. Iterar sobre las definiciones de los genes para generar el resto
        for i, gene_def in enumerate(gene_definitions):
            # Saltar los genes que ya han sido definidos por el usuario
            if chromosome[i] is not None:
                continue

            # 3. Aplicar lógica de dependencias
            if i == gene_map['g_TamCom']:
                g_TamLot = chromosome[gene_map['g_TamLot']]
                start = int(g_TamLot * 0.60)
                end = int(g_TamLot * 0.75)
                increment = 50
                gene_def = (start, end, increment)

            elif i == gene_map['g_TaZoCo']:
                g_TamCom = chromosome[gene_map['g_TamCom']]
                start = int(g_TamCom * 0.15)
                end = int(g_TamCom * 0.25)
                increment = 50
                gene_def = (start, end, increment)

            elif i == gene_map['g_AreVer']:
                g_TamCom = chromosome[gene_map['g_TamCom']]
                start = int(g_TamCom * 0.10)
                end = int(g_TamCom * 0.20)
                increment = 50
                gene_def = (start, end, increment)

            elif i == gene_map['g_TaCiPa']:
                g_TamPar = constants['g_TamPar']
                start = int(g_TamPar * 0.30)
                end = int(g_TamPar * 0.40)
                increment = 50
                gene_def = (start, end, increment)

            elif i == gene_map['g_TaZoAu']:
                g_TaUtPa = constants['g_TaUtPa']
                start = int(g_TaUtPa * 0.70)
                end = int(g_TaUtPa * 0.80)
                increment = 20
                gene_def = (start, end, increment)
            
            # 4. Lógica de generación estándar para el gen actual
            if isinstance(gene_def, tuple) and len(gene_def) == 3:
                start, end, increment = gene_def
                
                if isinstance(start, int) and isinstance(end, int) and isinstance(increment, int):
                    possible_values = list(range(start, end + increment, increment))
                else:
                    num_steps = int(round((end - start) / increment)) + 1
                    possible_values = [start + i * increment for i in range(num_steps)]

                value = random.choice(possible_values)
                if isinstance(value, float):
                    chromosome[i] = round(value, 2)
                else:
                    chromosome[i] = value
            
            elif isinstance(gene_def, list):
                value = random.choice(gene_def)
                if isinstance(value, float):
                    chromosome[i] = round(value, 2)
                else:
                    chromosome[i] = value
            else:
                raise ValueError(f"Definición de gen inválida para el índice {i}.")
        
        # 5. Verificar la unicidad del cromosoma y agregarlo a la población
        if chromosome not in population:
            population.append(chromosome)
    
    return population

def mutate_chromosome(chromosome, gene_definitions, mutation_rate, sigma_factor, user_gene_indices=None):
    """
    Aplica mutación a un cromosoma, respetando los genes fijos y las dependencias.

    Args:
        chromosome (list): El cromosoma a mutar.
        gene_definitions (list): Las definiciones de los genes.
        mutation_rate (float): La probabilidad de que un gen individual mute.
        sigma_factor (float): Factor para la desviación estándar de la mutación gaussiana.
        user_gene_indices (list): Lista de índices de genes que son fijos y no deben mutar.

    Returns:
        list: El nuevo cromosoma mutado.
    """
    # Genes que no deben mutar: los valores fijos del usuario.
    # Los genes dependientes no se mutan, se recalcularán después.
    if user_gene_indices is None:
        user_gene_indices = [0, 24, 25]  # Solo g_TamLot, b_CanPri, b_CanSec

    mutated_chromosome = [None] * len(chromosome)

    # Paso 1: Mutar solo los genes independientes
    for i, gene_def in enumerate(gene_definitions):
        if i in user_gene_indices:
            mutated_chromosome[i] = chromosome[i]
            continue
        
        if random.random() < mutation_rate:
            if isinstance(gene_def, tuple) and len(gene_def) == 3:
                current_value = chromosome[i]
                start, end, _ = gene_def
                
                sigma = (end - start) * sigma_factor
                mutation_value = np.random.normal(0, sigma)
                new_value = current_value + mutation_value
                
                new_value = max(start, min(end, new_value))
                
                if isinstance(start, float):
                    mutated_chromosome[i] = round(new_value, 2)
                else:
                    mutated_chromosome[i] = int(round(new_value))
            elif isinstance(gene_def, list):
                mutated_chromosome[i] = random.choice(gene_def)
            else:
                raise ValueError("Definición de gen inválida.")
        else:
            mutated_chromosome[i] = chromosome[i]

    return mutated_chromosome

def recalculate_dependent_genes(chromosome, constants):
    """
    Recalcula los valores de los genes dependientes en un cromosoma.
    
    Args:
        chromosome (list): Un cromosoma con genes mutados.
        constants (dict): Diccionario con las constantes del problema.
    
    Returns:
        list: El cromosoma con los genes dependientes actualizados.
    """
    # Mapeo de nombres de genes a sus índices para facilitar el acceso
    gene_map = {
        'g_TamLot': 0, 'g_TamCom': 1, 'g_TaZoCo': 2, 'g_AreVer': 3, 'g_TaCiPa': 4,
        'g_TaZoAu': 5, 'b_CanPri': 24, 'b_CanSec': 25
    }

    # Recalcular g_TamCom, g_TaZoCo, g_AreVer
    g_TamLot = chromosome[gene_map['g_TamLot']]
    chromosome[gene_map['g_TamCom']] = int(g_TamLot * random.uniform(0.60, 0.75))
    
    g_TamCom = chromosome[gene_map['g_TamCom']]
    chromosome[gene_map['g_TaZoCo']] = int(g_TamCom * random.uniform(0.15, 0.25))
    chromosome[gene_map['g_AreVer']] = int(g_TamCom * random.uniform(0.10, 0.20))
    
    # Recalcular g_TaCiPa, g_TaZoAu
    g_TamPar = g_TamLot - g_TamCom # Asumiendo que g_TamPar es la diferencia
    chromosome[gene_map['g_TaCiPa']] = int(g_TamPar * random.uniform(0.30, 0.40))
    
    g_TaUtPa = g_TamPar - chromosome[gene_map['g_TaCiPa']] # Asumiendo que g_TaUtPa es la diferencia
    chromosome[gene_map['g_TaZoAu']] = int(g_TaUtPa * random.uniform(0.70, 0.80))
    
    return chromosome

# Definiciones de los genes basadas en el ejemplo del código proporcionado
# Se asume que los valores de las proporciones (l_PLTi25, l_PLTi20, l_PLTi16) son porcentajes,
# por lo que los valores enteros de gene_definitions (ej. 9, 19, 29) deben ser divididos por 100 en los cálculos.
gene_definitions = [
    (5000, 6000, 100), # 0: g_TamLot USUARIO
    (3000, 3750, 50),  # 1: g_TamCom depende 60%-75% de g_TamLot
    (450, 750, 50),    # 2: g_TaZoCo depende 15%-25% de g_TamCom
    (300, 600, 50),    # 3: g_AreVer depende 10%-20% de g_TamCom
    (600, 800, 50),    # 4: g_TaCiPa depende 30%-40% de g_TamPar
    (980, 1120, 20),   # 5: g_TaZoAu depende 70%-80% de g_TaUtPa
    (2300000, 2500000, 30000), # 6: g_CoCoLo
    (80000, 110000, 5000), # 7: g_VaArMe
    (5000, 6000, 100), # 8: g_DVALMe
    [9, 10, 11],         # 9: l_PLTI25 (proporción)
    [19,20,21,22],     # 10: l_PLTI20 (proporción)
    [29,30,31,32],  # 11: l_PLTI16 (proporción)
    (450000, 550000, 1), # 12: p_CoCoPa
    (400000, 550000, 1), # 13: z_CoZoCi
    (300000, 350000, 1), # 14: z_CoUrAv
    [9,10,11,12],      # 15: q_EspCl1
    [2, 3],             # 16: q_EspCl2
    [1, 2],             # 17: m_CaPeCo
    [8, 9, 10],          # 18: m_CaReMe
    (400, 450, 10),    # 19: s_CanBom
    (680, 800, 20),    # 20: s_MeCuAg
    (340, 420, 10),    # 21: s_MeCuGa
    [3, 4],             # 22: v_CaEmLi
    [12,13,14],        # 23: v_CaEmVi
    [0,1,2,3,4],       # 24: b_CanPri USUARIO
    [0,1,2,3,4],       # 25: b_CanSec USUARIO
    (0.80,0.99,0.01),      #26: e_CaTrPu
    (0.80,0.99,0.01),      #27: e_CaZoRe
    (0.70,0.99,0.01),      #28: e_CaEsPa
    [2, 3, 4],           # 29: w_CPEm25
    [2, 3],             # 30: w_CPEm20
    [1, 2],             # 31: w_CPEm16
    [1],               # 32: w_CPEm12
    [2, 3],             # 33: x_CPInPr
    (20, 40, 2),       # 34: x_CPInLg
    (15, 30, 3),       # 35: x_CPInMn
    (5, 10, 1),        # 36: x_CPInZv
    (50, 75, 3),       # 37: x_CPInEx
    (0.60,0.89,0.03),      #38: h_CaMeIn
    (0.60,0.89,0.03),      #39: h_CaPeCo
    (0.60,0.89,0.05),      #40: h_CaInSo
    (0.60,0.89,0.04),      #41: k_CaPABS
    (0.60,0.89,0.02),      #42: k_CaPeCo
    (0.60,0.89,0.01),      #43: k_CaPeSe
    (0.60,0.89,0.01),      #44: k_CaPeEn
]

# Diccionario para mapear los índices del cromosoma a los nombres de las variables
# y especificar si un valor debe ser tratado como porcentaje (dividido por 100)
GENE_INDEX_MAP = {
    0: {'name': 'g_TamLot', 'is_percentage': False},
    1: {'name': 'g_TamCom', 'is_percentage': False},
    2: {'name': 'g_TaZoCo', 'is_percentage': False},
    3: {'name': 'g_AreVer', 'is_percentage': False},
    4: {'name': 'g_TaCiPa', 'is_percentage': False},
    5: {'name': 'g_TaZoAu', 'is_percentage': False},
    6: {'name': 'g_CoCoLo', 'is_percentage': False},
    7: {'name': 'g_VaArMe', 'is_percentage': False},
    8: {'name': 'g_DVALMe', 'is_percentage': False},
    9: {'name': 'l_PLTi25', 'is_percentage': True},
    10: {'name': 'l_PLTi20', 'is_percentage': True},
    11: {'name': 'l_PLTi16', 'is_percentage': True},
    12: {'name': 'p_CoCoPa', 'is_percentage': False},
    13: {'name': 'z_CoZoCi', 'is_percentage': False},
    14: {'name': 'z_CoUrAV', 'is_percentage': False}, # Costo de Urbanismo y Áreas Verdes
    15: {'name': 'q_EspCl1', 'is_percentage': False},
    16: {'name': 'q_EspCl2', 'is_percentage': False},
    17: {'name': 'm_CaPeCo', 'is_percentage': False},
    18: {'name': 'm_CaReMe', 'is_percentage': False},
    19: {'name': 's_CanBom', 'is_percentage': False},
    20: {'name': 's_MeCuAg', 'is_percentage': False},
    21: {'name': 's_MeCuGa', 'is_percentage': False},
    22: {'name': 'v_CaEmLi', 'is_percentage': False},
    23: {'name': 'v_CaEmVi', 'is_percentage': False},
    24: {'name': 'b_CanPri', 'is_percentage': False},
    25: {'name': 'b_CanSec', 'is_percentage': False},
    26: {'name': 'w_CPEm25', 'is_percentage': False},
    27: {'name': 'w_CPEm20', 'is_percentage': False},
    28: {'name': 'w_CPEm16', 'is_percentage': False},
    29: {'name': 'w_CPEm12', 'is_percentage': False},
    30: {'name': 'x_CPInPr', 'is_percentage': False},
    31: {'name': 'x_CPInLg', 'is_percentage': False},
    32: {'name': 'x_CPInMn', 'is_percentage': False},
    33: {'name': 'x_CPInZv', 'is_percentage': False},
    34: {'name': 'x_CPInEx', 'is_percentage': False}
}

# Definición de las constantes que no son generadas por los genes en `gene_definitions`
# pero que son necesarias para los cálculos, tomadas de los valores de ejemplo de los fuentes.
CONSTANTS = {
    # Genes de Galería (Locus 1-3, no incluidos en gene_definitions)
    # 'g_CodGal': 1, #
    # 'g_CodCom': 1, #
    # 'g_CodLot': 2, #   

    # Constantes de dimensiones de zonas comunes administrativas
    'z_TaOfAd': 20,
    'z_TaBoAd': 20,
    'z_TaCaAd': 12,
    'z_TaBaCo': 68,

    # Constantes de consumo de servicio de parqueadero
    'q_Co1Cl1': 330498,
    'q_Co2Cl1': 118026,
    'q_Co1Cl2': 206514,
    'q_Co2Cl2': 68850,

    # Constantes de mantenimiento
    'm_SaPeCo': 1356423, # Salario unitario personal contratado
    'm_CUReMe': 250000, # Costo unitario reparaciones menores

    # Constantes de servicios públicos
    's_CoCoUB': 14.4, # Consumo unitario bombilla
    's_TarCME': 700,  # Tarifa consumo eléctrico por kilovatio
    's_TarCMA': 5168, # Tarifa consumo agua por metro cúbico
    's_TarCMG': 3000, # Tarifa consumo gas por metro cúbico
    's_CoToIM': 266910, # Costo total internet mensual

    # Constantes de salarios de personal
    'o_AdmGal': 4000000, # Salario mensual administrador
    'o_AsiGal': 2094729, # Salario mensual asistente

    # Constantes de servicios operativos
    'v_SaMeLi': 1356423, # Salario mensual empleados limpieza
    'v_SaMeVi': 4336930, # Salario mensual empleados vigilancia

    # Constantes de gastos administrativos
    'n_PaOfMe': 300000, # Papelería oficina mensual
    'n_SeAdMe': 5000000, # Póliza seguro administrativo mensual

    # Constantes de gastos legales
    't_ReMeMe': 208333, # Costo registro mercantil mensual

    # Constantes de licencias
    'c_LiFuMe': 500000, # Licencia funcionamiento mensual
    'c_LiAmMe': 5000000, # Licencia ambiental mensual

    # Constantes de empleo indirecto
    'x_TMEILg': 40, # Tope máximo logística
    'x_TMEIMn': 30, # Tope máximo mantenimiento
    'x_TMEIZv': 10, # Tope máximo zonas verdes
    'x_TMEIEx': 75, # Tope máximo externos

    # Constantes de calificación de ubicación estratégica
    'e_CaTrPu': 0.9256, # Calificación transporte público
    'e_CaZoRe': 0.8349, # Calificación zonas residenciales
    'e_CaEsPa': 0.7001, # Calificación escuelas, parques

    # Constantes de percepciones
    'h_CaMeIn': 0.8500, # Mejora de infraestructura
    'h_CaPeCo_comunidad': 0.6300, # Percepción de la comunidad (renombrada para evitar conflicto)
    'h_CaInSo': 0.7500, # Inclusión social
    'k_CaPABS': 0.9000, # Acceso a bienes y servicios
    'k_CaPeCo_comodidad': 0.8200, # Comodidad (renombrada para evitar conflicto)
    'k_CaPeSe': 0.7100, # Seguridad
    'k_CaPeEn': 0.8600, # Entorno

    # Constantes de proporción de locales complementarios
    'y_ProAlF': 0.4200, # Proporción para alimentos frescos
    'y_ProCoP': 0.1500, # Proporción para comidas preparadas
    'y_ProNAl': 0.3500, # Proporción para productos no alimentarios
    'y_ProSeC': 0.0800, # Proporción para servicios complementarios (este valor es fijo en el ejemplo, aunque la fórmula de y_CLoSeC sugiere que se calcula como el restante)
}

def calculate_gallery_metrics(chromosome, constants, gene_index_map):
    """
    Calcula todas las métricas de la galería comercial basándose en un cromosoma (genes)
    y un conjunto de constantes predefinidas.

    Args:
        chromosome (list): Una lista de valores que representa un cromosoma.
        constants (dict): Un diccionario de valores constantes no incluidos en el cromosoma.
        gene_index_map (dict): Un mapa de índices de cromosoma a nombres de variables y si son porcentajes.

    Returns:
        dict: Un diccionario con todas las métricas calculadas y sus valores.
    """
    metrics = {}
    metrics.update(constants)
    # -----------------------------------------------------------
    # 1. Asignar los valores de los genes del cromosoma a las métricas
    # -----------------------------------------------------------
    for index, gene_info in gene_index_map.items():
        gene_value = chromosome[index]
        if gene_info['is_percentage']:
            metrics[gene_info['name']] = gene_value / 100.0  # Convertir a decimal
        else:
            metrics[gene_info['name']] = gene_value

    # Función auxiliar para obtener valores con manejo de errores
    def get_metric(key, default=0):
        return metrics.get(key, default)    
    # Incorporar las constantes

    try:

        # -----------------------------------------------------------
        # 2. Genes de Galería (Cálculos iniciales) [12-14]
        # -----------------------------------------------------------
        # Locus 90: g_TaUtCo [13]
        metrics['g_TaUtCo'] = metrics['g_TamCom'] - (metrics['g_TaZoCo'] + metrics['g_AreVer'])
        # Locus 91: g_TaUtCo [12]
        metrics['g_TamPar'] = metrics['g_TamLot'] - metrics['g_TamCom'] 
        # Locus 92: g_TaUtPa [14]
        metrics['g_TaUtPa'] = metrics['g_TamPar'] - metrics['g_TaCiPa']
        # Locus 93: g_TaZoMo [14]
        metrics['g_TaZoMo'] = metrics['g_TaUtPa'] - metrics['g_TaZoAu']
        # Asegurar que los tamaños no sean negativos
        metrics['g_TaUtCo'] = max(0, metrics['g_TaUtCo'])
        metrics['g_TaZoMo'] = max(0, metrics['g_TaZoMo'])

        # -----------------------------------------------------------
        # 3. Inversión Inicial [48]
        # -----------------------------------------------------------

        # Construcción de Locales [15-19]
        # La proporción para locales tipo 12 es el restante para sumar 1 (o 100%)
        metrics['l_PLTi12'] = 1 - (metrics['l_PLTi25'] + metrics['l_PLTi20'] + metrics['l_PLTi16'])
        metrics['l_PLTi12'] = max(0, metrics['l_PLTi12'])  # Ensure not negative
        # Locus l_TLTi25, l_CLTi25, l_CULT25, l_CTLo25 [15]
        metrics['l_TLTi25'] = int(metrics['l_PLTi25'] * metrics['g_TaUtCo'])
        metrics['l_CLTi25'] = int(metrics['l_TLTi25'] / 25)
        metrics['l_CULT25'] = metrics['g_CoCoLo'] * 25
        metrics['l_CTLo25'] = metrics['l_CLTi25'] * metrics['l_CULT25']

        # Locus l_TLTi20, l_CLTi20, l_CULT20, l_CTLo20 [16]
        metrics['l_TLTi20'] = int(metrics['l_PLTi20'] * metrics['g_TaUtCo'])
        metrics['l_CLTi20'] = int(metrics['l_TLTi20'] / 20)
        metrics['l_CULT20'] = metrics['g_CoCoLo'] * 20
        metrics['l_CTLo20'] = metrics['l_CLTi20'] * metrics['l_CULT20']

        # Locus l_TLTi16, l_CLTi16, l_CULT16, l_CTLo16 [17]
        metrics['l_TLTi16'] = int(metrics['l_PLTi16'] * metrics['g_TaUtCo'])
        metrics['l_CLTi16'] = int(metrics['l_TLTi16'] / 16)
        metrics['l_CULT16'] = metrics['g_CoCoLo'] * 16
        metrics['l_CTLo16'] = metrics['l_CLTi16'] * metrics['l_CULT16']

        # Locus l_PLTi12, l_TLTi12, l_CLTi12, l_CULT12, l_CTLo12 [18]
    
        metrics['l_TLTi12'] = int(metrics['l_PLTi12'] * metrics['g_TaUtCo'])
        metrics['l_CLTi12'] = int(metrics['l_TLTi12'] / 12)
        metrics['l_CULT12'] = metrics['g_CoCoLo'] * 12
        metrics['l_CTLo12'] = metrics['l_CLTi12'] * metrics['l_CULT12']

        # Locus 140: l_CTCLo (Costo Total Construcción Locales) [19]
        metrics['l_CTCLo'] = metrics['l_CTLo25'] + metrics['l_CTLo20'] + metrics['l_CTLo16'] + metrics['l_CTLo12']

        # Construcción de Parqueaderos [19, 20]
        # p_CTPCl1, p_CTPCl2 [19]
        metrics['p_CTPCl1'] = metrics['g_TaZoAu'] * metrics['p_CoCoPa']
        metrics['p_CTPCl2'] = metrics['g_TaZoMo'] * metrics['p_CoCoPa']
        # Locus 141: p_SCTPar (Sumatoria Costo Total Parqueaderos) [20]
        metrics['p_SCTPar'] = metrics['p_CTPCl1'] + metrics['p_CTPCl2']

        # Construcción de Zonas Comunes [20-22]
        # Locus 103-106: z_CoTOAd, z_CoTBAd, z_CoTCAd, z_CoTBCo [20, 21]
        metrics['z_CoTOAd'] = metrics['z_TaOfAd'] * metrics['g_CoCoLo']
        metrics['z_CoTBAd'] = metrics['z_TaBoAd'] * metrics['g_CoCoLo']
        metrics['z_CoTCAd'] = metrics['z_TaCaAd'] * metrics['g_CoCoLo']
        metrics['z_CoTBCo'] = metrics['z_TaBaCo'] * metrics['g_CoCoLo']
        # Locus 107: z_CoTZCi [22]
        metrics['z_CoTZCi'] = metrics['g_TaZoCo'] * metrics['z_CoZoCi']
        # Locus 108, 109: z_TaUrAV, z_CoUrAV (para urbanismos) [22]
        metrics['z_TaUrAV'] = metrics['g_AreVer'] - (metrics['z_TaOfAd'] + metrics['z_TaBoAd'] + metrics['z_TaCaAd'] + metrics['z_TaBaCo'])
        metrics['z_TaUrAV'] = max(0, metrics['z_TaUrAV'])  # Asegurar que no sea negativo
        metrics['z_CoTUrAV'] = metrics['z_TaUrAV'] * metrics['z_CoUrAV']  # Renombrada para evitar conflicto con el gen z_CoUrAV

        # Locus 142: z_CTZcAv (Costo Total Zonas Comunes Areas Verdes) [22]
        metrics['z_CTZcAv'] = metrics['z_CoTOAd'] + metrics['z_CoTBAd'] + metrics['z_CoTCAd'] + \
                            metrics['z_CoTBCo'] + metrics['z_CoTZCi'] + metrics['z_CoTUrAV']

        # Locus 155: i_InvIni (Inversión Inicial) [48]
        metrics['i_InvIni'] = metrics['l_CTCLo'] + metrics['p_SCTPar'] + metrics['z_CTZcAv']

        # -----------------------------------------------------------
        # 4. Ingresos [49]
        # -----------------------------------------------------------

        # Arrendamiento de Locales [23-26]
        # Locus 110: a_VTLo25, a_SVTL25, a_VTAA25 [23]
        metrics['a_VTLo25'] = 25 * metrics['g_VaArMe'] * 0.9
        metrics['a_SVTL25'] = metrics['l_CLTi25'] * metrics['a_VTLo25']
        metrics['a_VTAA25'] = 12 * metrics['a_SVTL25']

        # Locus 111: a_VTLo20, a_SVTL20, a_VTAA20 [24]
        metrics['a_VTLo20'] = 20 * metrics['g_VaArMe'] * 0.93
        metrics['a_SVTL20'] = metrics['l_CLTi20'] * metrics['a_VTLo20']
        metrics['a_VTAA20'] = 12 * metrics['a_SVTL20']

        # Locus 112: a_VTLo16, a_SVTL16, a_VTAA16 [25]
        metrics['a_VTLo16'] = 16 * metrics['g_VaArMe'] * 0.95
        metrics['a_SVTL16'] = metrics['l_CLTi16'] * metrics['a_VTLo16']
        metrics['a_VTAA16'] = 12 * metrics['a_SVTL16']

        # Locus 113: a_VTLo12, a_SVTL12, a_VTAA12 [26]
        metrics['a_VTLo12'] = 12 * metrics['g_VaArMe']
        metrics['a_SVTL12'] = metrics['l_CLTi12'] * metrics['a_VTLo12']
        metrics['a_VTAA12'] = 12 * metrics['a_SVTL12']

        # Locus 143: a_ToArGa (Valor Total Arriendo Galería) [26]
        metrics['a_ToArGa'] = metrics['a_VTAA25'] + metrics['a_VTAA20'] + metrics['a_VTAA16'] + metrics['a_VTAA12']

        # Administración [27-30]
        # Locus 114: d_DVTL25, d_SDTL25, d_VTAD25 [27]
        metrics['d_DVTL25'] = 25 * metrics['g_DVALMe'] * 0.9
        metrics['d_SDTL25'] = metrics['l_CLTi25'] * metrics['d_DVTL25']
        metrics['d_VTAD25'] = 12 * metrics['d_SDTL25']

        # Locus 115: d_DVTL20, d_SDTL20, d_VTAD20 [28]
        metrics['d_DVTL20'] = 20 * metrics['g_DVALMe'] * 0.93
        metrics['d_SDTL20'] = metrics['l_CLTi20'] * metrics['d_DVTL20']
        metrics['d_VTAD20'] = 12 * metrics['d_SDTL20']

        # Locus 116: d_DVTL16, d_SDTL16, d_VTAD16 [29]
        metrics['d_DVTL16'] = 16 * metrics['g_DVALMe'] * 0.95
        metrics['d_SDTL16'] = metrics['l_CLTi16'] * metrics['d_DVTL16']
        metrics['d_VTAD16'] = 12 * metrics['d_SDTL16']

        # Locus 117: d_DVTL12, d_SDTL12, d_VTAD12 [30]
        metrics['d_DVTL12'] = 12 * metrics['g_DVALMe']
        metrics['d_SDTL12'] = metrics['l_CLTi12'] * metrics['d_DVTL12']
        metrics['d_VTAD12'] = 12 * metrics['d_SDTL12']

        # Locus 144: d_ToAdGa (Valor Total Administración Galería) [30]
        metrics['d_ToAdGa'] = metrics['d_VTAD25'] + metrics['d_VTAD20'] + metrics['d_VTAD16'] + metrics['d_VTAD12']

        # Servicio de Parqueadero [31-34]
        # Locus 118: q_CanCl1 [31]
        metrics['q_CanCl1'] = int(metrics['g_TaZoAu'] / metrics['q_EspCl1'])
        # q_Va1Cl1, q_Va2Cl1, q_VTMCl1, q_ToCCl1 [32]
        metrics['q_Va1Cl1'] = metrics['q_CanCl1'] * metrics['q_Co1Cl1']
        metrics['q_Va2Cl1'] = metrics['q_CanCl1'] * metrics['q_Co2Cl1']
        metrics['q_VTMCl1'] = metrics['q_Va1Cl1'] + metrics['q_Va2Cl1']
        metrics['q_ToCCl1'] = 12 * metrics['q_VTMCl1']

        # Locus 119: q_CanCl2 [33]
        metrics['q_CanCl2'] = int(metrics['g_TaZoMo'] / metrics['q_EspCl2'])
        # q_Va1Cl2, q_Va2Cl2, q_VTMCl2, q_ToCCl2 [34]
        metrics['q_Va1Cl2'] = metrics['q_CanCl2'] * metrics['q_Co1Cl2']
        metrics['q_Va2Cl2'] = metrics['q_CanCl2'] * metrics['q_Co2Cl2']
        metrics['q_VTMCl2'] = metrics['q_Va1Cl2'] + metrics['q_Va2Cl2']
        metrics['q_ToCCl2'] = 12 * metrics['q_VTMCl2']

        # Locus 145: q_ToPaGa (Valor Total Parqueadero Galería) [35]
        metrics['q_ToPaGa'] = metrics['q_ToCCl1'] + metrics['q_ToCCl2']

        # Locus 156: u_IngGal (Ingresos Galería) [49]
        metrics['u_IngGal'] = metrics['a_ToArGa'] + metrics['d_ToAdGa'] + metrics['q_ToPaGa']

        # -----------------------------------------------------------
        # 5. Egresos [49]
        # -----------------------------------------------------------

        # Mantenimiento [35-38]
        # m_TaArCo [35]
        metrics['m_TaArCo'] = metrics['g_TaZoCo'] + metrics['z_TaUrAV']
        # m_STMPCo, m_STAPCo [36]
        metrics['m_STMPCo'] = metrics['m_CaPeCo'] * metrics['m_SaPeCo']
        metrics['m_STAPCo'] = 12 * metrics['m_STMPCo']
        # m_CMReMe, m_CAnRMe, m_InReMe, m_InReAn [37]
        metrics['m_CMReMe'] = metrics['m_CaReMe'] * metrics['m_CUReMe']
        metrics['m_CAnRMe'] = 12 * metrics['m_CMReMe']
        metrics['m_InReMe'] = 0.6 * metrics['m_CUReMe']  # Asumiendo 0.6 es una constante
        metrics['m_InReAn'] = 12 * metrics['m_InReMe']

        # Locus 146: m_ToEgGa (Total Egreso Mantenimiento Galería) [38]
        metrics['m_ToEgGa'] = metrics['m_STAPCo'] + metrics['m_CAnRMe'] + metrics['m_InReAn']

        # Servicios Públicos [38-41]
        # s_CoToEM, s_CoToEA [39]
        metrics['s_CoToEM'] = metrics['s_CanBom'] * metrics['s_CoCoUB'] * metrics['s_TarCME']
        metrics['s_CoToEA'] = 12 * metrics['s_CoToEM']
        # s_CoToAM, s_CoToAA [40]
        metrics['s_CoToAM'] = metrics['s_MeCuAg'] * metrics['s_TarCMA']
        metrics['s_CoToAA'] = 12 * metrics['s_CoToAM']
        # s_CoToGM, s_CoToGA [41]
        metrics['s_CoToGM'] = metrics['s_MeCuGa'] * metrics['s_TarCMG']
        metrics['s_CoToGA'] = 12 * metrics['s_CoToGM']
        # s_CoToIA [41]
        metrics['s_CoToIA'] = 12 * metrics['s_CoToIM']

        # Locus 147: s_ToSPGa (Total Egreso Servicios Públicos Galería) [41]
        metrics['s_ToSPGa'] = metrics['s_CoToEA'] + metrics['s_CoToAA'] + metrics['s_CoToGM'] + metrics['s_CoToIA']

        # Salarios de Personal [42]
        # o_AdAnGa [42]
        metrics['o_AdAnGa'] = 12 * metrics['o_AdmGal']
        # o_AsAnGa [42]
        metrics['o_AsAnGa'] = 12 * metrics['o_AsiGal']

        # Locus 148: o_ToSaGa (Total Egreso Salario Personal Galería) [42]
        metrics['o_ToSaGa'] = metrics['o_AdAnGa'] + metrics['o_AsAnGa']

        # Servicios Operativos [43-45]
        # v_SaToML, v_SaAnLi [43]
        metrics['v_SaToML'] = metrics['v_CaEmLi'] * metrics['v_SaMeLi']
        metrics['v_SaAnLi'] = 12 * metrics['v_SaToML']
        # v_SaToMV, v_SaAnVi [44]
        metrics['v_SaToMV'] = metrics['v_CaEmVi'] * metrics['v_SaMeVi']
        metrics['v_SaAnVi'] = 12 * metrics['v_SaToMV']

        # Locus 149: v_ToSOGa (Total Egreso Servicio Operativo Galería) [45]
        metrics['v_ToSOGa'] = metrics['v_SaAnLi'] + metrics['v_SaAnVi']

        # Gastos Administrativos [45, 46]
        # n_PAOfAn [45]
        metrics['n_PAOfAn'] = 12 * metrics['n_PaOfMe']
        # n_SeAdAn [45]
        metrics['n_SeAdAn'] = 12 * metrics['n_SeAdMe']

        # Locus 150: n_ToGAGa (Total Egreso Gastos Administrativos Galería) [46]
        metrics['n_ToGAGa'] = metrics['n_PAOfAn'] + metrics['n_SeAdAn']

        # Gastos Legales [46]
        # t_ReMeAn [46]
        metrics['t_ReMeAn'] = 12 * metrics['t_ReMeMe']

        # Locus 151: t_ToRMGa (Total Egreso Registro Mercantil Galería) [46]
        metrics['t_ToRMGa'] = metrics['t_ReMeAn']

        # Locus 157: u_EgrGal (Egresos Galería) [49]
        metrics['u_EgrGal'] = metrics['m_ToEgGa'] + metrics['s_ToSPGa'] + metrics['o_ToSaGa'] + \
                            metrics['v_ToSOGa'] + metrics['n_ToGAGa'] + metrics['t_ToRMGa']

        # -----------------------------------------------------------
        # 6. Resultados Económicos y Financieros [48-50]
        # -----------------------------------------------------------

        # Locus 158: u_UtBrGa (Utilidad Bruta Galería) [49]
        metrics['u_UtBrGa'] = metrics['u_IngGal'] - metrics['u_EgrGal']

        # Impuestos [47, 48]
        # c_ICAAno [47]
        metrics['c_ICAAno'] = 0.06 * metrics['u_IngGal']  # Asumiendo 0.06 es una constante
        # c_PreAno [47]
        metrics['c_PreAno'] = 0.08 * metrics['i_InvIni']  # Asumiendo 0.08 es una constante
        # c_RenAno [47]
        metrics['c_RenAno'] = 0.35 * metrics['u_UtBrGa']  # Asumiendo 0.35 es una constante
        # c_LiFuAn [48]
        metrics['c_LiFuAn'] = 12 * metrics['c_LiFuMe']
        # c_LiAmAn [48]
        metrics['c_LiAmAn'] = 12 * metrics['c_LiAmMe']

        # Locus 159: u_ImpGas (Impuestos y Gastos) [49]
        metrics['u_ImpGas'] = metrics['c_ICAAno'] + metrics['c_PreAno'] + metrics['c_RenAno'] + \
                            metrics['c_LiFuAn'] + metrics['c_LiAmAn']

        # Locus 160: u_UtNeGa (Utilidad Neta Galería) [50]
        metrics['u_UtNeGa'] = metrics['u_UtBrGa'] - metrics['u_ImpGas']

        # Locus 165: u_MarUtN (Margen Utilidad Neta) [50]
        metrics['u_MarUtN'] = metrics['u_UtNeGa'] / metrics['u_IngGal'] if metrics['u_IngGal'] != 0 else 0

        # Locus 161: u_ROIGal (Retorno Sobre Inversión Porcentual) [50]
        metrics['u_ROIGal'] = (metrics['u_UtNeGa'] / metrics['i_InvIni']) * 100 if metrics['i_InvIni'] != 0 else 0

        # Locus 166: ROI (Retorno Sobre Inversión Equivalente) [50]
        metrics['ROI'] = metrics['u_ROIGal'] / 100

        # Locus 168: B/C (Relación beneficio y costo) [50]
        metrics['u_BenCos'] = metrics['u_UtNeGa'] / metrics['u_EgrGal'] if metrics['u_EgrGal'] != 0 else 0

        # -----------------------------------------------------------
        # 7. Beneficio Social - Accesibilidad [51-53, 62]
        # -----------------------------------------------------------

        # b_PonPri, b_PonSec [51]
        metrics['b_PonPri'] = metrics['b_CanPri'] / 4  # Asumiendo /4 es una constante
        metrics['b_PonSec'] = metrics['b_CanSec'] / 8  # Asumiendo /8 es una constante

        # Locus 152: b_ToPVia (Total Ponderación Vial) [52]
        sum_pon_vial = metrics['b_PonPri'] + metrics['b_PonSec']
        metrics['b_ToPVia'] = min(sum_pon_vial, 1)  # si((b_PonPri + b_PonSec)<1;b_PonPri + b_PonSec;1)

        # Locus 153: e_CaPoEs (Calificación Ponderada Estrategica) [53]
        metrics['e_CaPoEs'] = (metrics['e_CaZoRe'] + metrics['e_CaEsPa']) / 2

        # Locus 162: e_Accesi (Accesibilidad) [62]
        metrics['e_Accesi'] = (metrics['b_ToPVia'] + metrics['e_CaTrPu'] + metrics['e_CaPoEs']) / 3

        # -----------------------------------------------------------
        # 8. Beneficio Social - Impacto en la Comunidad [53-59, 62]
        # -----------------------------------------------------------

        # Empleo Directo [53-56]
        # Locus 120-121: w_TMED25, w_ToED25 [53]
        metrics['w_TMED25'] = metrics['l_CLTi25'] * 4
        metrics['w_ToED25'] = metrics['w_CPEm25'] * metrics['l_CLTi25']

        # Locus 122-123: w_TMED20, w_ToED20 [54]
        metrics['w_TMED20'] = metrics['l_CLTi20'] * 3
        metrics['w_ToED20'] = metrics['w_CPEm20'] * metrics['l_CLTi20']

        # Locus 124-125: w_TMED16, w_ToED16 [55]
        metrics['w_TMED16'] = metrics['l_CLTi16'] * 2
        metrics['w_ToED16'] = metrics['w_CPEm16'] * metrics['l_CLTi16']

        # Locus 126-127: w_TMED12, w_ToED12 [56]
        metrics['w_TMED12'] = metrics['l_CLTi12'] * 1
        metrics['w_ToED12'] = metrics['w_CPEm12'] * metrics['l_CLTi12']

        # Locus 128: w_STMEDi (Suma Tope Máximo Empleos Directos Galería) [56]
        # Se añaden constantes 1 para administrador y asistente, según la fórmula en el fuente
        metrics['w_STMEDi'] = metrics['w_TMED25'] + metrics['w_TMED20'] + metrics['w_TMED16'] + \
                            metrics['w_TMED12'] + metrics['m_CaPeCo'] + 1 + 1 + \
                            metrics['v_CaEmLi'] + metrics['v_CaEmVi']

        # Locus 129: w_STEmDi (Suma Total Empleados Directos Galería) [56]
        # Se añaden constantes 1 para administrador y asistente, según la fórmula en el fuente
        metrics['w_STEmDi'] = metrics['w_ToED25'] + metrics['w_ToED20'] + metrics['w_ToED16'] + \
                            metrics['w_ToED12'] + metrics['m_CaPeCo'] + 1 + 1 + \
                            metrics['v_CaEmLi'] + metrics['v_CaEmVi']

        # Empleo Indirecto [57-59]
        # Locus 130: x_TMEIPr (Tope Máximo Empleos Indirectos Proveedor) [57]
        metrics['x_TMEIPr'] = (metrics['l_CLTi25'] + metrics['l_CLTi20'] + metrics['l_CLTi16'] + metrics['l_CLTi12']) * 3
        # Locus 131: x_ToEIPr (Total Empleados Indirectos Proveedor) [57]
        metrics['x_ToEIPr'] = metrics['x_CPInPr'] * (metrics['l_CLTi25'] + metrics['l_CLTi20'] + metrics['l_CLTi16'] + metrics['l_CLTi12'])

        # Locus 132: x_STMEIn (Suma Tope Máximo Empleos Indirectos Galería) [59]
        metrics['x_STMEIn'] = metrics['x_TMEIPr'] + metrics['x_TMEILg'] + metrics['x_TMEIMn'] + \
                            metrics['x_TMEIZv'] + metrics['x_TMEIEx']

        # Locus 133: x_STEmIn (Suma Total Empleados Indirectos Galería) [59]
        # Nota: La fórmula en el fuente [59] parece tener un error tipográfico usando las variables x_CPInLg, x_CPInMn, etc.
        # en lugar de los totales correspondientes. Se sigue la fórmula como está escrita:
        metrics['x_STEmIn'] = metrics['x_ToEIPr'] + metrics['x_CPInLg'] + metrics['x_CPInMn'] + \
                            metrics['x_CPInZv'] + metrics['x_CPInEx']

        # Locus 134: x_Empleo (Empleo Generado) [59]
        metrics['x_Empleo'] = metrics['w_STEmDi'] + metrics['x_STEmIn']

        # Locus 154: x_InEmGe (Índice Empleo Generado) [59]
        metrics['x_InEmGe'] = metrics['x_Empleo'] / (metrics['w_STMEDi'] + metrics['x_STMEIn']) if (metrics['w_STMEDi'] + metrics['x_STMEIn']) != 0 else 0

        # Locus 163: h_ImpCom (Impacto en la Comunidad) [62]
        metrics['h_ImpCom'] = (metrics['x_InEmGe'] + metrics['h_CaMeIn'] + \
                            metrics['h_CaPeCo_comunidad'] + metrics['h_CaInSo']) / 4

        # -----------------------------------------------------------
        # 9. Resultados Beneficio Social (Calidad de Vida) [61, 62]
        # -----------------------------------------------------------

        # Locus 164: k_CalVid (Calidad Vida) [62]
        metrics['k_CalVid'] = (metrics['k_CaPABS'] + metrics['k_CaPeCo_comodidad'] + \
                            metrics['k_CaPeSe'] + metrics['k_CaPeEn']) / 4

        # Locus 167: u_BenSoc (Beneficio Social generado por la galería) [63]
        metrics['u_BenSoc'] = (metrics['e_Accesi'] + metrics['h_ImpCom'] + metrics['k_CalVid']) / 3

        # -----------------------------------------------------------
        # 10. Información Complementaria - Áreas Comerciales [63-65]
        # -----------------------------------------------------------

        # Locus 135: y_ToLoCo (Total Locales Comerciales) [63]
        metrics['y_ToLoCo'] = metrics['l_CLTi25'] + metrics['l_CLTi20'] + metrics['l_CLTi16'] + metrics['l_CLTi12']

        # Locus 136: y_CLoAlF (Cantidad Locales Alimentos Frescos) [63]
        metrics['y_CLoAlF'] = int(metrics['y_ProAlF'] * metrics['y_ToLoCo'])

        # Locus 137: y_CLoCoP (Cantidad Locales Comidas Preparadas) [64]
        metrics['y_CLoCoP'] = int(metrics['y_ProCoP'] * metrics['y_ToLoCo'])

        # Locus 138: y_CLoNAl (Cantidad Locales Productos No Alimentarios) [64]
        metrics['y_CLoNAl'] = int(metrics['y_ProNAl'] * metrics['y_ToLoCo'])

        # Locus 139: y_CLoSeC (Cantidad Locales Servicios Complementarios) [65]
        # La fórmula en el fuente [65] parece tener un error de signo: y_ToLoCo - (y_CLoAlF - y_CLoCoP - y_CLoNAl)
        # Debería ser la resta de la suma de los anteriores.
        metrics['y_CLoSeC'] = metrics['y_ToLoCo'] - (metrics['y_CLoAlF'] - metrics['y_CLoCoP'] - metrics['y_CLoNAl'])
    
    except Exception as e:
        # Registrar el error pero continuar con la ejecución
        print(f"Error en calculate_gallery_metrics: {str(e)}")
        
    return metrics

def calculate_fitness(metrics, weights=None):
    """
    Calcula la aptitud (fitness) de una galería comercial basándose en sus métricas económicas y sociales.

    Args:
        metrics (dict): Diccionario que contiene las métricas calculadas.
        weights (tuple): Tuple con los pesos (w1, w2, w3) para BE, BS, MUN. Por defecto (0.4, 0.5, 0.1)

    Returns:
        float: El valor de aptitud calculado.
    """
    # Usar pesos por defecto si no se proporcionan
    if weights is None:
        w1, w2, w3 = 0.4, 0.5, 0.1  # BE, BS, MUN
    else:
        w1, w2, w3 = weights
    
    # Asegurarse de que los valores existan en las métricas
    be_component = metrics.get('u_BenCos', 0.0)
    bs_component = metrics.get('u_BenSoc', 0.0)
    mun_component = metrics.get('u_MarUtN', 0.0)

    # Normalizar el componente BE (Beneficio Económico)
    def normalize_bc(x):
        k = 2.0  # Controla qué tan rápido la función se acerca a 1
        x0 = 1.0  # Punto donde la función vale 0.5 (B/C = 1)
        return 1 / (1 + math.exp(-k * (x - x0)))
    
    be_normalized = normalize_bc(be_component)
    
    fitness = (w1 * be_normalized) + (w2 * bs_component) + (w3 * mun_component)
    return fitness

def crossover_chromosomes(parent1, parent2, crossover_rate, user_gene_indices=None):
    """
    Realiza un cruce uniforme entre dos cromosomas padres para crear un hijo.
    
    Args:
        parent1 (list): El primer cromosoma padre.
        parent2 (list): El segundo cromosoma padre.
        crossover_rate (float): La probabilidad de heredar un gen del padre 1.
        user_gene_indices (list): Lista de índices de genes que son fijos y no deben cruzarse.
    
    Returns:
        Un nuevo cromosoma hijo (lista de valores) que es una combinación de los padres.
    """
    chromosome_length = len(parent1)
    
    if user_gene_indices is None:
        user_gene_indices = [0, 24, 25]  # Solo g_TamLot, b_CanPri, b_CanSec
        
    child_chromosome = [None] * chromosome_length
    
    for i in range(chromosome_length):
        # Los genes fijos del usuario se copian directamente del padre 1 (o 2, no importa)
        if i in user_gene_indices:
            child_chromosome[i] = parent1[i]
        # Para el resto de los genes, aplicar el cruce uniforme
        elif random.random() < crossover_rate:
            child_chromosome[i] = parent1[i]
        else:
            child_chromosome[i] = parent2[i]
            
    return child_chromosome

def select_elites(population, fitness_scores, elite_percentage):
    """
    Selecciona los mejores individuos de la población basándose en las puntuaciones de aptitud.
    
    Args:
        population (list): Lista de cromosomas.
        fitness_scores (list): Lista de puntuaciones de aptitud correspondientes.
        elite_percentage (float): Porcentaje de la población a seleccionar como élites.
    
    Returns:
        list: Lista de cromosomas élites.
    """
    num_elites = int(len(population) * elite_percentage)
    combined = list(zip(population, fitness_scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    elites = [individual for individual, score in combined[:num_elites]]
    return elites

def select_parents(population, fitness_scores, tournament_size=3):
    """
    Selecciona un padre usando selección por torneo.
    
    Args:
        population (list): Lista de cromosomas.
        fitness_scores (list): Lista de puntuaciones de aptitud correspondientes.
        tournament_size (int): Tamaño del torneo.
    
    Returns:
        list: Un cromosoma padre seleccionado.
    """
    selected_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in selected_indices]
    winner_index = selected_indices[np.argmax(tournament_fitness)]
    return population[winner_index]

def calculate_diversity(fitness_scores):
    """
    Calcula la diversidad de la población como la desviación estándar del fitness.
    
    Args:
        fitness_scores (list): Lista de puntuaciones de aptitud.
    
    Returns:
        float: Desviación estándar de las puntuaciones de aptitud.
    """
    return np.std(fitness_scores) if len(fitness_scores) > 1 else 0.0

def adjust_parameters(params, diversity, stagnation_count, improvement_count):
    """
    Ajusta los parámetros basándose en reglas heurísticas.
    
    Args:
        params (dict): Diccionario con los parámetros actuales.
        diversity (float): Diversidad de la población (desviación estándar del fitness).
        stagnation_count (int): Número de generaciones sin mejora.
        improvement_count (int): Número de generaciones con mejora continua.
    
    Returns:
        dict: Diccionario con los parámetros ajustados.
    """
    new_params = params.copy()

    # Reducir la frecuencia de los ajustes
    if random.random() < 0.3:  # Solo ajustar el 30% del tiempo
        return new_params
    # Regla 1: Estancamiento (sin mejora en 20 generaciones)
    if stagnation_count >= 20:
        new_params['mutation_rate'] = min(params['mutation_rate'] * 1.3, 0.1)  # Máximo 10%
        new_params['sigma_factor'] = min(params['sigma_factor'] * 1.3, 1.5)    # Máximo 2.0
        if random.random() < 0.1:  # Solo loguear el 10% de las veces
            print("Ajuste por estancamiento: mutación y sigma_factor aumentados.")
    
    # Regla 2: Baja diversidad (diversidad < umbral, e.g., 0.05)
    if diversity < 0.02:
        new_params['elite_percentage'] = max(0.05, params['elite_percentage'] * 0.8)
        new_params['mutation_rate'] = min(params['mutation_rate'] * 1.2, 0.15)  # Aumentar mutación ligeramente
        new_params['sigma_factor'] = min(params['sigma_factor'] * 1.2, 1.8)    # Aumentar sigma ligeramente
        if random.random() < 0.1:  # Solo loguear el 10% de las veces
            print("Ajuste por baja diversidad: elitismo reducido, mutación y sigma_factor aumentados.")
    
    # Regla 3: Mejora constante (10 generaciones)
    if improvement_count >= 10:
        new_params['mutation_rate'] = max(params['mutation_rate'] * 0.8, 0.01)  # Mínimo 1%
        new_params['sigma_factor'] = max(params['sigma_factor'] * 0.8, 0.5)     # Mínimo 0.1
        if random.random() < 0.1:  # Solo loguear el 10% de las veces
            print("Ajuste por mejora constante: mutación y sigma_factor reducidos.")
    
    return new_params

