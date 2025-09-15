from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash,current_app, abort
from datetime import datetime
from utils.genetic_algorithm import *
from extensions import db
from models import *
from sqlalchemy import text  
from dotenv import load_dotenv
from functools import wraps
import uuid
import numpy as np
import random
import threading
import time
import os

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
SYNC_MODE = os.getenv("SYNC_MODE", "1" if ENVIRONMENT == "production" else "0") == "1"

# Secret key
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-insecure-key")
if ENVIRONMENT == "production" and SECRET_KEY == "dev-insecure-key":
    raise RuntimeError("FLASK_SECRET_KEY es obligatorio en producción.")

# Database URL (normaliza postgres:// → postgresql://)
db_url = os.environ.get("DATABASE_URL", "")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
if ENVIRONMENT == "production" and not db_url:
    raise RuntimeError("DATABASE_URL es obligatorio en producción.")
if not db_url:
    # Fallback SOLO para desarrollo si no definiste DATABASE_URL
    db_url = "postgresql://postgres:postgres@localhost:5432/postgres"

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Configuracion de la base de datos - FORMA CORRECTA
app.config['SQLALCHEMY_DATABASE_URI'] = db_url or "sqlite:///local.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["ENVIRONMENT"] = ENVIRONMENT
app.config["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "INFO")

db.init_app(app)

app.config['EXECUTION_LOGS'] = {}

def only_dev(f):
    @wraps(f)
    def _w(*args, **kwargs):
        if current_app.config.get("ENVIRONMENT") == "production":
            abort(404)
        return f(*args, **kwargs)
    return _w

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/debug/database')
@only_dev
def debug_database():
    try:
        db.session.execute(text('SELECT 1'))  # ← Usar text()
        return jsonify({'status': 'ok', 'message': 'Conexion a la base de datos exitosa.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/debug/session')
@only_dev
def debug_session():
    return jsonify({
        'max_generations': session.get('max_generations'),
        'thread_id': session.get('thread_id'),
        # Agrega otras variables que quieras verificar
    })

@app.route('/parametrizacion', methods=['GET', 'POST'])
def parametrizacion():
    if request.method == 'POST':
        galerias_existentes = []
        
        for i in range(1, 7):  # Galerias 1 a 6 (existentes)
            # Obtener datos de cada galeria existente
            tam_lote = int(request.form[f'tam_lote_{i}'])
            can_pri = int(request.form[f'can_pri_{i}'])
            can_sec = int(request.form[f'can_sec_{i}'])
            
            # Calcular constantes para esta galeria existente
            g_TamPar_const = tam_lote - (tam_lote * 0.6)
            g_TaUtPa_const = g_TamPar_const - (g_TamPar_const * 0.3)
            
            user_inputs_existente = {
                'g_TamLot': tam_lote,
                'b_CanPri': can_pri,
                'b_CanSec': can_sec
            }
            
            constants_existente = {
                'g_TamPar': g_TamPar_const,
                'g_TaUtPa': g_TaUtPa_const,
            }
            
            # Crear un cromosoma basico para la galeria existente
            cromosoma_existente = [None] * len(gene_definitions)
            cromosoma_existente[0] = tam_lote  # g_TamLot
            cromosoma_existente[24] = can_pri  # b_CanPri
            cromosoma_existente[25] = can_sec  # b_CanSec
            
            # Recalcular genes dependientes
            cromosoma_existente = recalculate_dependent_genes(cromosoma_existente, constants_existente)
            
            # SOLO PREPARAR DATOS - EL ROI SE CALCULARa DESPUeS
            galeria_data = {
                'numero': i,
                'comuna': i,
                'tam_lote': tam_lote,
                'can_pri': can_pri,
                'can_sec': can_sec,
                'cromosoma': cromosoma_existente,
                'constants': constants_existente,
                'user_inputs': user_inputs_existente
            }
            galerias_existentes.append(galeria_data)
        
        # Preparar tambien la galeria 7 (nueva)
        TamLot = int(request.form['tam_lote_7'])
        CanPri = int(request.form['can_pri_7'])
        CanSec = int(request.form['can_sec_7'])
        
        # Calcular constantes para la nueva galeria
        g_TamPar_const = TamLot - (TamLot * 0.6)
        g_TaUtPa_const = g_TamPar_const - (g_TamPar_const * 0.3)

        # Preparar datos de la galeria 7 (comuna temporal, se determinara despues)
        galeria_7_data = {
            'numero': 7,
            'comuna': 7,  # Temporal, se cambiara despues
            'tam_lote': TamLot,
            'can_pri': CanPri,
            'can_sec': CanSec,
            'cromosoma': None,  # Se calculara en procesar_todas_galerias
            'constants': {
                'g_TamPar': g_TamPar_const,
                'g_TaUtPa': g_TaUtPa_const,
            },
            'user_inputs': {
                'g_TamLot': TamLot,
                'b_CanPri': CanPri,
                'b_CanSec': CanSec,
                'comuna': 7  # Temporal
            }
        }
        
        # Agregar la galeria 7 a la lista
        galerias_existentes.append(galeria_7_data)
        
        # Obtener pesos para la funcion de fitness
        peso_bs = float(request.form['peso_bs']) / 100
        peso_be = float(request.form['peso_be']) / 100
        peso_mun = float(request.form['peso_mun']) / 100
        
        # Validar que los pesos sumen 1
        if abs(peso_bs + peso_be + peso_mun - 1.0) > 0.01:
            return render_template('parametrizacion.html', 
                                 error="Los pesos deben sumar 100%")
        
        # Guardar en sesion
        session['weights'] = (peso_be, peso_bs, peso_mun)
        if not (ENVIRONMENT == "production" and SYNC_MODE):
            session['galerias_existentes'] = galerias_existentes
        session['comuna_nueva_galeria'] = None  # Se determinara despues
        
        # Configurar parametros por defecto si no existen
        session.setdefault('population_size', 50)
        session.setdefault('max_generations', 300)
        session.setdefault('elite_percentage', 0.1)
        session.setdefault('mutation_rate', 0.05)
        session.setdefault('sigma_factor', 0.1)
        session.setdefault('crossover_rate', 0.7)
        
        # Iniciar ejecucion en segundo plano
        thread_id = str(time.time())
        session['thread_id'] = thread_id

        # Inicializar logs si no existen
        if 'EXECUTION_LOGS' not in app.config:
            app.config['EXECUTION_LOGS'] = {}
        app.config['EXECUTION_LOGS'][thread_id] = []
        
        run_id = str(uuid.uuid4())
        session['last_run_id'] = run_id

        # ✅ En Vercel (SYNC_MODE=1) ejecuta SIN hilo y redirige a resultados
        if SYNC_MODE:
            try:
                procesar_todas_galerias(
                    app, thread_id, galerias_existentes,
                    session['population_size'], session['max_generations'],
                    session['elite_percentage'], session['mutation_rate'],
                    session['sigma_factor'], session['crossover_rate'],
                    (peso_be, peso_bs, peso_mun),
                    run_id
                )
            except Exception as e:
                app.config['EXECUTION_LOGS'][thread_id].append(f"ERROR: {e}")
                flash("Ocurrió un error durante el procesamiento.", "danger")
                return render_template('parametrizacion.html', show_progress_modal=False)

            # Terminado: ir directo a /resultados con el run_id
            return redirect(url_for('resultados', run_id=run_id))

        # 🧵 Modo local/asíncrono (cuando SYNC_MODE=0): conserva tu hilo + modal
        thread = threading.Thread(
            target=procesar_todas_galerias,
            args=(app, thread_id, galerias_existentes,
                  session['population_size'], session['max_generations'],
                  session['elite_percentage'], session['mutation_rate'],
                  session['sigma_factor'], session['crossover_rate'],
                  (peso_be, peso_bs, peso_mun),
                  run_id)
        )
        thread.daemon = True
        thread.start()
        print(f"Thread {thread_id} iniciado para procesar 7 galerías (run_id={run_id})")

        return render_template('parametrizacion.html',
                               show_progress_modal=True,
                               thread_id=thread_id,
                               run_id=run_id,
                               galerias_existentes=galerias_existentes)
    
    return render_template('parametrizacion.html', show_progress_modal=False)

@app.route('/procesar_todas_galerias', methods=['POST'])
def procesar_todas_galerias_route():
    # 1) Recuperar datos desde sesión
    galerias_existentes = session.get('galerias_existentes')
    weights = session.get('weights')
    if not galerias_existentes or not weights:
        return jsonify({
            "status": "error",
            "message": "Faltan datos en sesión. Cargue parámetros en /parametrizacion."
        }), 400

    # 2) Parámetros con tipos seguros (int/float) y valores por defecto
    try:
        population_size = int(session.get('population_size', 50))
        max_generations = int(session.get('max_generations', 1000))
        elite_percentage = float(session.get('elite_percentage', 0.1))
        mutation_rate = float(session.get('mutation_rate', 0.05))
        sigma_factor = float(session.get('sigma_factor', 0.1))
        crossover_rate = float(session.get('crossover_rate', 0.7))
    except Exception:
        return jsonify({
            "status": "error",
            "message": "Parámetros inválidos en sesión."
        }), 400

    # 3) Identificadores y logs en memoria
    thread_id = str(time.time())  # (conserva tu formato para compatibilidad)
    session['thread_id'] = thread_id
    run_id = str(uuid.uuid4())

    if 'EXECUTION_LOGS' not in app.config:
        app.config['EXECUTION_LOGS'] = {}
    app.config['EXECUTION_LOGS'][thread_id] = []

    session['last_run_id'] = run_id

    # 4) En producción y modo demo, limitar carga para no exceder timeouts serverless
    if ENVIRONMENT == "production" and SYNC_MODE:
        population_size = min(population_size, 100)
        max_generations = min(max_generations, 500)

    # 5) Rama según SYNC_MODE
    if SYNC_MODE:
        # --- SÍNCRONO: ejecutar directamente dentro de la petición ---
        try:
            procesar_todas_galerias(
                app, thread_id, galerias_existentes,
                population_size, max_generations,
                elite_percentage, mutation_rate,
                sigma_factor, crossover_rate, weights,
                run_id
            )
        except Exception as e:
            # Registra el error en logs del thread para trazabilidad
            app.config['EXECUTION_LOGS'][thread_id].append(f"ERROR: {e}")
            return jsonify({
                "status": "error",
                "message": "Error durante el procesamiento (modo síncrono)."
            }), 500

        # Terminado: el front puede redirigir a /resultados usando run_id
        return jsonify({
            "status": "ok",
            "thread_id": thread_id,
            "run_id": run_id,
            "completed": True
        })
    else:
        # --- ASÍNCRONO: comportamiento actual con thread ---
        def _target():
            try:
                procesar_todas_galerias(
                    app, thread_id, galerias_existentes,
                    population_size, max_generations,
                    elite_percentage, mutation_rate,
                    sigma_factor, crossover_rate, weights,
                    run_id
                )
            except Exception as e:
                # Guarda el error en los logs de ejecución
                app.config['EXECUTION_LOGS'][thread_id].append(f"ERROR: {e}")

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()

        # Respuesta inmediata; el front puede hacer polling de logs si lo usas
        return jsonify({
            "status": "ok",
            "thread_id": thread_id,
            "run_id": run_id
        })
    
def calcular_roi_seguro(metrics):
    """
    Calcula el ROI de forma segura, manejando casos donde los valores puedan ser None.
    """
    try:
        inversion = metrics.get('i_InvIni')
        utilidad = metrics.get('u_UtNeGa')
        
        # Verificar que ambos valores existan y sean numericos
        if inversion is None or utilidad is None:
            return 0.0
        
        # Verificar que sean convertibles a float
        try:
            inversion_val = float(inversion)
            utilidad_val = float(utilidad)
        except (ValueError, TypeError):
            return 0.0
        
        # Evitar division por cero
        if inversion_val == 0:
            return 0.0
            
        return (utilidad_val / inversion_val) * 100
        
    except Exception:
        return 0.0

def procesar_todas_galerias(app, thread_id, galerias_a_procesar, 
                            population_size, max_generations, elite_percentage,
                            mutation_rate, sigma_factor, crossover_rate, weights,
                            run_id):
    """
    Nota: requiere que el modelo Ejecucion tenga la columna:
      run_id = db.Column(db.String(36), index=True, nullable=False)
    """

    with app.app_context():
        logs = app.config['EXECUTION_LOGS'].get(thread_id, [])
        
        logs.append("=" * 60)
        logs.append("PROCESANDO LAS 7 GALERiAS COMERCIALES")
        logs.append("=" * 60)

        # Dejar evidencia de que no se borra el historial
        logs.append(f"[{datetime.utcnow().isoformat()}Z] Preservando historial (no se borra la tabla).")
        logs.append(f"[{datetime.utcnow().isoformat()}Z] run_id={run_id} asignado a esta corrida.")
        app.config['EXECUTION_LOGS'][thread_id] = logs

        resultados_galerias = {}
        
        try:
            # Procesar cada una de las 7 galerías
            for galeria in galerias_a_procesar:
                galeria_num = galeria['numero']

                # --- VERIFICACIÓN para evitar duplicar dentro del MISMO run_id ---
                existing_execution = Ejecucion.query.filter_by(comuna=galeria_num, run_id=run_id).first()
                if existing_execution:
                    logs.append(f"GALERIA_{galeria_num} ya existe en este run_id={run_id}. Se omite la inserción duplicada.")
                    app.config['EXECUTION_LOGS'][thread_id] = logs
                    continue
                # -----------------------------------------------------------------

                logs.append(f"Procesando Galeria {galeria_num}...")
                app.config['EXECUTION_LOGS'][thread_id] = logs
                time.sleep(0.1)
                
                # Calcular constantes para esta galeria
                g_TamPar_const = galeria['tam_lote'] - (galeria['tam_lote'] * 0.6)
                g_TaUtPa_const = g_TamPar_const - (g_TamPar_const * 0.3)

                user_inputs = {
                    'g_TamLot': galeria['tam_lote'],
                    'b_CanPri': galeria['can_pri'],
                    'b_CanSec': galeria['can_sec'],
                    'comuna': galeria_num
                }

                constants = {
                    'g_TamPar': g_TamPar_const,
                    'g_TaUtPa': g_TaUtPa_const,
                }
                
                # Combinar constantes globales (si las usas dentro del GA)
                from utils.genetic_algorithm import CONSTANTS as GLOBAL_CONSTANTS, gene_definitions
                full_constants = {**GLOBAL_CONSTANTS, **constants}
                
                # EJECUTAR GA
                result  = run_genetic_algorithm(
                    app, 
                    f"{thread_id}_galeria_{galeria_num}",
                    user_inputs, 
                    constants,  # Si tu GA necesita full_constants, cámbialo aquí
                    population_size, 
                    max_generations,
                    elite_percentage,
                    mutation_rate,
                    sigma_factor,
                    crossover_rate,
                    weights,
                    None
                )

                # Verificar si el resultado es válido
                if result is None:
                    logs.append(f"Error: run_genetic_algorithm retornó None para Galeria {galeria_num}")
                    # Crear valores por defecto
                    best_chromosome = [0] * len(gene_definitions)
                    best_metrics = {'u_ROIGal': 0.0, 'u_UtNeGa': 0.0}
                    best_fitness = 0.0
                else:
                    best_chromosome, best_metrics, best_fitness = result
                
                # Guardar resultados en memoria
                resultados_galerias[galeria_num] = {
                    'best_chromosome': best_chromosome,
                    'best_metrics': best_metrics,
                    'best_fitness': best_fitness,
                    'user_inputs': user_inputs,
                    'constants': constants
                }
                
                # Guardar en BD solo las primeras 6 galerías
                if galeria_num <= 6:
                    try:
                        nueva_ejecucion = Ejecucion(
                            comuna=galeria_num,  
                            tam_lote_m2=user_inputs['g_TamLot'],
                            can_pri_unidades=user_inputs['b_CanPri'],
                            can_sec_unidades=user_inputs['b_CanSec'],

                            peso_bs=weights[1],
                            peso_be=weights[0],
                            peso_mun=weights[2],

                            poblacion_inicial=population_size,
                            generaciones=max_generations,
                            tasa_mutacion=mutation_rate,

                            mejor_fitness=best_fitness,
                            inv_inicial_usd=best_metrics.get('i_InvIni', 0.0),
                            roi=best_metrics.get('u_ROIGal', 0.0),
                            utilidad_neta_usd=best_metrics.get('u_UtNeGa', 0.0),
                            margen_utilidad=best_metrics.get('u_MarUtN', 0.0),
                            empleos_directos=best_metrics.get('x_Empleo', 0.0),
                            beneficio_social=best_metrics.get('u_BenSoc', 0.0),

                            cromosoma_optimo=best_chromosome,

                            # 👇 NUEVO
                            run_id=run_id
                        )
                        db.session.add(nueva_ejecucion)
                        db.session.commit()
                        try:
                            det = EjecucionDetalle(
                                ejecucion_id=nueva_ejecucion.id,
                                inv_total=best_metrics.get('i_InvIni'),
                                inv_loc=best_metrics.get('l_CTCLo'),
                                inv_parq=best_metrics.get('p_SCTPar'),
                                inv_zonas=best_metrics.get('z_CTZcAv'),
                                ing_total=best_metrics.get('u_IngGal'),
                                ing_arr=best_metrics.get('a_ToArGa'),
                                ing_adm=best_metrics.get('d_ToAdGa'),
                                ing_parq=best_metrics.get('q_ToPaGa'),
                                egr_total=best_metrics.get('u_EgrGal'),
                                egr_mant=best_metrics.get('m_ToEgGa'),
                                egr_servpub=best_metrics.get('s_CoToEA') and best_metrics.get('s_ToSPGa') or best_metrics.get('s_ToSPGa'),
                                egr_salarios=best_metrics.get('o_ToSaGa'),
                                egr_operativos=best_metrics.get('v_ToSOGa'),
                                egr_admin=best_metrics.get('n_ToGAGa'),
                                egr_legales=best_metrics.get('t_ToRMGa'),
                                egr_impuestos=best_metrics.get('u_ImpGas'),
                                bs_accesibilidad=best_metrics.get('e_Accesi', None),
                                bs_emp_dir=best_metrics.get('w_STEmDi'),
                                bs_emp_ind=best_metrics.get('x_STEmIn'),
                                bs_calidad_vida=best_metrics.get('k_CalVid', None),
                                ar_alimentos_frescos=best_metrics.get('y_CLoAlF'),
                                ar_comidas_preparadas=best_metrics.get('y_CLoCoP'),
                                ar_no_alimentarios=best_metrics.get('y_CLoNAl'),
                                ar_complementarios=best_metrics.get('y_CLoSeC'),
                            )
                            db.session.add(det)
                            db.session.commit()
                        except Exception as det_e:
                            db.session.rollback()
                            logs.append(f"[WARN] No se pudo guardar detalle de la Galería {galeria_num}: {det_e}")
                        logs.append(f"GALERIA_{galeria_num}_ROI:{best_metrics.get('u_ROIGal', 0):.2f}")
                        logs.append(f"GALERIA_{galeria_num}_COMPLETADA (run_id={run_id})")
                    except Exception as db_e:
                        db.session.rollback()
                        logs.append(f"Error al guardar Galeria {galeria_num}: {str(db_e)}")
                
                app.config['EXECUTION_LOGS'][thread_id] = logs
            
            # Después de procesar las 6 galerías, encontrar la mejor
            mejores_roi = []
            for i in range(1, 7):
                if i in resultados_galerias:
                    roi = resultados_galerias[i]['best_metrics'].get('u_ROIGal', 0)
                    mejores_roi.append((i, roi))
            
            if mejores_roi:
                # Selección de comuna óptima
                mejor_par = max(mejores_roi, key=lambda x: x[1])
                mejor_comuna = mejor_par[0]
                logs.append(f"Mejor comuna encontrada: {mejor_comuna} con ROI: {mejor_par[1]:.2f}%")
                
                galeria_7 = resultados_galerias.get(7)
                if galeria_7 is None:
                    # Si no se había calculado (no debería ocurrir), crear desde la configuración original
                    g7_cfg = next((g for g in galerias_a_procesar if g['numero'] == 7), None)
                    if g7_cfg is None:
                        logs.append("No se encontró configuración para la Galería 7.")
                        app.config['EXECUTION_LOGS'][thread_id] = logs
                        return
                    galeria_7 = {
                        'user_inputs': {
                            'g_TamLot': g7_cfg['tam_lote'],
                            'b_CanPri': g7_cfg['can_pri'],
                            'b_CanSec': g7_cfg['can_sec'],
                            'comuna': mejor_comuna
                        },
                        'constants': {
                            'g_TamPar': g7_cfg['tam_lote'] - (g7_cfg['tam_lote'] * 0.6),
                            'g_TaUtPa': (g7_cfg['tam_lote'] - (g7_cfg['tam_lote'] * 0.6)) * (1 - 0.3)
                        }
                    }
                else:
                    # Ajustar comuna a la óptima
                    galeria_7['user_inputs']['comuna'] = mejor_comuna
                    galeria_7['best_metrics']['comuna'] = mejor_comuna

                logs.append(f"Optimizando Galeria 7 para la Comuna {mejor_comuna}...")
                app.config['EXECUTION_LOGS'][thread_id] = logs

                result_final_7 = run_genetic_algorithm(
                    app,
                    f"{thread_id}_galeria_7_final",
                    galeria_7['user_inputs'],
                    galeria_7['constants'],
                    population_size,
                    max_generations,
                    elite_percentage,
                    mutation_rate,
                    sigma_factor,
                    crossover_rate,
                    weights,
                    None
                )

                if result_final_7 is None:
                    logs.append("Error: run_genetic_algorithm retornó None para Galeria 7 (final). Se usarán valores por defecto.")
                    best_chromosome_7 = [0] * len(gene_definitions)
                    best_metrics_7 = {'u_ROIGal': 0.0, 'u_UtNeGa': 0.0}
                    best_fitness_7 = 0.0
                else:
                    best_chromosome_7, best_metrics_7, best_fitness_7 = result_final_7

                # GUARDAR EN BD LA GALERÍA 7
                try:
                    ejec_7 = Ejecucion(
                        comuna=7,  # Identidad de "galería nueva"
                        tam_lote_m2=galeria_7['user_inputs']['g_TamLot'],
                        can_pri_unidades=galeria_7['user_inputs']['b_CanPri'],
                        can_sec_unidades=galeria_7['user_inputs']['b_CanSec'],

                        peso_bs=weights[1],
                        peso_be=weights[0],
                        peso_mun=weights[2],

                        poblacion_inicial=population_size,
                        generaciones=max_generations,
                        tasa_mutacion=mutation_rate,

                        mejor_fitness=best_fitness_7,
                        inv_inicial_usd=best_metrics_7.get('i_InvIni', 0.0),
                        roi=best_metrics_7.get('u_ROIGal', 0.0),
                        utilidad_neta_usd=best_metrics_7.get('u_UtNeGa', 0.0),
                        margen_utilidad=best_metrics_7.get('u_MarUtN', 0.0),
                        empleos_directos=best_metrics_7.get('x_Empleo', 0.0),
                        beneficio_social=best_metrics_7.get('u_BenSoc', 0.0),

                        cromosoma_optimo=best_chromosome_7,

                        run_id=run_id
                    )
                    db.session.add(ejec_7)
                    db.session.commit()
                    
                    try:
                        det7 = EjecucionDetalle(
                            ejecucion_id=ejec_7.id,
                            inv_total=best_metrics_7.get('i_InvIni'),
                            inv_loc=best_metrics_7.get('l_CTCLo'),
                            inv_parq=best_metrics_7.get('p_SCTPar'),
                            inv_zonas=best_metrics_7.get('z_CTZcAv'),
                            ing_total=best_metrics_7.get('u_IngGal'),
                            ing_arr=best_metrics_7.get('a_ToArGa'),
                            ing_adm=best_metrics_7.get('d_ToAdGa'),
                            ing_parq=best_metrics_7.get('q_ToPaGa'),
                            egr_total=best_metrics_7.get('u_EgrGal'),
                            egr_mant=best_metrics_7.get('m_ToEgGa'),
                            egr_servpub=best_metrics_7.get('s_ToSPGa'),
                            egr_salarios=best_metrics_7.get('o_ToSaGa'),
                            egr_operativos=best_metrics_7.get('v_ToSOGa'),
                            egr_admin=best_metrics_7.get('n_ToGAGa'),
                            egr_legales=best_metrics_7.get('t_ToRMGa'),
                            egr_impuestos=best_metrics_7.get('u_ImpGas'),
                            bs_accesibilidad=best_metrics_7.get('e_Accesi', None),
                            bs_emp_dir=best_metrics_7.get('w_STEmDi'),
                            bs_emp_ind=best_metrics_7.get('x_STEmIn'),
                            bs_calidad_vida=best_metrics_7.get('k_CalVid', None),
                            ar_alimentos_frescos=best_metrics_7.get('y_CLoAlF'),
                            ar_comidas_preparadas=best_metrics_7.get('y_CLoCoP'),
                            ar_no_alimentarios=best_metrics_7.get('y_CLoNAl'),
                            ar_complementarios=best_metrics_7.get('y_CLoSeC'),
                        )
                        db.session.add(det7)
                        db.session.commit()
                    except Exception as det_e:
                        db.session.rollback()
                        logs.append(f"[WARN] No se pudo guardar detalle de la Galería {galeria_num}: {det_e}")

                    logs.append(f"GALERIA_7_GUARDADA (comuna óptima {mejor_comuna}) ROI:{best_metrics_7.get('u_ROIGal', 0):.2f} (run_id={run_id})")
                except Exception as e:
                    db.session.rollback()
                    logs.append(f"Error al guardar Galeria 7: {str(e)}")

                # Almacenar resultados finales para UI
                if 'RESULTS' not in app.config:
                    app.config['RESULTS'] = {}
                
                app.config['RESULTS'][thread_id] = {
                    'best_chromosome': galeria_7.get('best_chromosome'),
                    'best_metrics': galeria_7.get('best_metrics'),
                    'best_fitness': galeria_7.get('best_fitness'),
                    'mejor_comuna': mejor_comuna,
                    'resultados_galerias': resultados_galerias,
                    'run_id': run_id
                }
                
                logs.append("PROCESAMIENTO COMPLETADO")
                app.config['EXECUTION_LOGS'][thread_id] = logs

        except Exception as e:
            error_msg = f"Error en el procesamiento: {str(e)}"
            logs.append(error_msg)
            import traceback
            traceback_str = traceback.format_exc()
            logs.append(traceback_str)
            
            if 'RESULTS' not in app.config:
                app.config['RESULTS'] = {}
                
            app.config['RESULTS'][thread_id] = {
                'error': error_msg,
                'traceback': traceback_str,
                'run_id': run_id
            }
            
            app.config['EXECUTION_LOGS'][thread_id] = logs

@app.route('/ejecucion')
def ejecucion():
    thread_id = request.args.get('thread_id', session.get('thread_id', ''))
    return render_template('ejecucion.html', thread_id=thread_id)

@app.route('/api/logs/<thread_id>')
def get_logs(thread_id):
    logs = app.config['EXECUTION_LOGS'].get(thread_id, [])
    
    # Verificar si hay resultados disponibles
    has_results = 'RESULTS' in app.config and thread_id in app.config['RESULTS']
    
    # Determinar el estado
    status = "ejecutando"
    if has_results:
        status = "completado"
    elif any("MEJOR SOLUCIoN ENCONTRADA" in log for log in logs):
        status = "completado"
    
    return jsonify({"logs": logs, "status": status, "has_results": has_results})

@app.route('/resultados')
def resultados():
    """
    Muestra resultados por run_id:
      - Lista las 7 galerías (1..6 + 7) guardadas en BD para ese run_id.
      - Incluye parámetros (population_size, max_generations, etc.) que se usaron.
      - Si la corrida sigue en curso (no hay 7 filas aún), muestra progreso por logs.
    Uso:
      GET /resultados?run_id=<uuid>
      Si no llega run_id, usa session['last_run_id'].
    """
    run_id = request.args.get('run_id') or session.get('last_run_id')
    thread_id = session.get('thread_id')

    if not run_id:
        flash("Falta run_id. Vuelve a parametrizar para iniciar una nueva corrida.", "warning")
        return redirect(url_for('parametrizacion'))

    # Parámetros asignados a la corrida (desde sesión)
    params = {
        "population_size": session.get('population_size', 50),
        "max_generations": session.get('max_generations', 300),
        "elite_percentage": session.get('elite_percentage', 0.1),
        "mutation_rate": session.get('mutation_rate', 0.05),
        "sigma_factor": session.get('sigma_factor', 0.1),
        "crossover_rate": session.get('crossover_rate', 0.7),
        "weights": session.get('weights')  # (BE, BS, MUN)
    }

    # 1) Leer resultados persistidos (BD) por run_id
    ejecuciones = (Ejecucion.query
                   .filter_by(run_id=run_id)
                   .order_by(Ejecucion.comuna.asc())
                   .all())

    # 2) Estado de la corrida
    total_guardadas = len(ejecuciones)
    completa = (total_guardadas >= 7)  # esperamos 7 filas

    # 3) Progreso en vivo (logs) mientras corre
    logs = []
    if thread_id and 'EXECUTION_LOGS' in app.config:
        logs = app.config['EXECUTION_LOGS'].get(thread_id, [])

    # 4) Resumen del worker en memoria (back-compat con tu UI actual)
    resumen_worker = None
    if 'RESULTS' in app.config and thread_id in app.config['RESULTS']:
        resumen_worker = app.config['RESULTS'][thread_id]
        # No lo borramos aquí para no perderlo en recargas del modal

    # 5) KPIs útiles: mejor ROI entre 1..6
    mejor_comuna, mejor_roi = None, None
    if ejecuciones:
        base = [e for e in ejecuciones if e.comuna and e.comuna <= 6]
        if base:
            best = max(base, key=lambda e: (e.roi or 0))
            mejor_comuna = best.comuna
            mejor_roi = best.roi

    # 6) Render al template
    return render_template(
        'resultados.html',
        run_id=run_id,
        completa=completa,
        ejecuciones=ejecuciones,   # 0..7 filas por run_id
        params=params,             # parámetros usados
        logs=logs,                 # progreso del modal
        resumen_worker=resumen_worker,  # métricas/cromosoma/fitness en memoria
        mejor_comuna=mejor_comuna,
        mejor_roi=mejor_roi
    )

def run_genetic_algorithm(app, thread_id, user_inputs, constants, 
                         population_size, max_generations, elite_percentage,
                         mutation_rate, sigma_factor, crossover_rate, weights,
                         galerias_existentes):
    with app.app_context():

        # Asegurate de que el array de logs existe
        if thread_id not in app.config['EXECUTION_LOGS']:
            app.config['EXECUTION_LOGS'][thread_id] = []

        logs = app.config['EXECUTION_LOGS'][thread_id]

        # LOG INICIAL CRíTICO
        logs.append("=" * 60)
        logs.append("ALGORITMO GENÉTICO INICIADO")
        logs.append("=" * 60)
        
        # Inicializar valores por defecto para retorno
        best_chromosome = None
        best_metrics = {}
        best_fitness = 0.0

           # Mostrar informacion de galerias existentes
        logs.append("ROI de galerias existentes:")
        if galerias_existentes is not None:
            for i, galeria in enumerate(galerias_existentes):
                logs.append(f"Galeria Comuna {galeria['comuna']} procesada - ROI: {galeria.get('roi', 0):.2f}%")
                app.config['EXECUTION_LOGS'][thread_id] = logs
                time.sleep(0.5)
            
            if galerias_existentes and len(galerias_existentes) > 0:
                mejor_galeria = max(galerias_existentes, key=lambda x: x.get('roi', 0))
                comuna_valor = mejor_galeria.get('comuna', 7)
            else:
                comuna_valor = user_inputs.get('comuna', 7)
        else:
            comuna_valor = user_inputs.get('comuna', 7)  # Para procesamiento individual
            logs.append("Procesamiento individual - sin galerias existentes para comparar")

        logs.append("=" * 60)

        logs.append("ROI de galerias existentes:")
        if galerias_existentes is not None:
            for i, galeria in enumerate(galerias_existentes):
                logs.append(f"Galeria Comuna {galeria['comuna']} procesada - ROI: {galeria.get('roi', 0):.2f}%")
                app.config['EXECUTION_LOGS'][thread_id] = logs
                time.sleep(0.5)
            
            if galerias_existentes and len(galerias_existentes) > 0:
                mejor_galeria = max(galerias_existentes, key=lambda x: x.get('roi', 0))
                comuna_valor = mejor_galeria.get('comuna', 7)
                logs.append(f"Mejor ROI: Comuna {mejor_galeria['comuna']}")
            else:
                comuna_valor = user_inputs.get('comuna', 7)
        else:
            comuna_valor = user_inputs.get('comuna', 7)  # Para procesamiento individual
            logs.append("Procesamiento individual - sin galerias existentes para comparar")
        logs.append("=" * 60)
        
        # ACTUALIZA INMEDIATAMENTE LOS LOGS
        app.config['EXECUTION_LOGS'][thread_id] = logs
        print(f"Logs iniciales configurados para thread {thread_id}")

        try:
            # Combinar constantes globales con constantes especificas de esta ejecucion
            from utils.genetic_algorithm import CONSTANTS as GLOBAL_CONSTANTS
            full_constants = {**GLOBAL_CONSTANTS, **constants}

            logs.append("Constantes combinadas correctamente")
            app.config['EXECUTION_LOGS'][thread_id] = logs

             # Asegurarse de que todas las constantes necesarias esten presentes
            required_constants = ['z_TaOfAd', 'z_TaBoAd', 'z_TaCaAd', 'z_TaBaCo']
            for const in required_constants:
                if const not in full_constants:
                    logs.append(f"ADVERTENCIA: Constante {const} no encontrada, usando valor por defecto")
                    # Establecer valores por defecto para constantes criticas
                    if const == 'z_TaOfAd':
                        full_constants[const] = 20
                    elif const == 'z_TaBoAd':
                        full_constants[const] = 20
                    elif const == 'z_TaCaAd':
                        full_constants[const] = 12
                    elif const == 'z_TaBaCo':
                        full_constants[const] = 68

            app.config['EXECUTION_LOGS'][thread_id] = logs
            print(f" Constantes combinadas para thread {thread_id}")
            
            # Inicializar poblacion
            population = create_initial_population(population_size, gene_definitions, user_inputs, full_constants)
            
            logs.append(f"Poblacion inicial creada con {len(population)} individuos")
            app.config['EXECUTION_LOGS'][thread_id] = logs

            logs.append("Poblacion inicial generada:")
            for i, chromosome in enumerate(population[:3]):
                logs.append(f"Individuo {i+1}: {chromosome}")
            logs.append("-" * 50)
            
            # Mostrar ejemplo de un cromosoma y sus metricas
            sample_chromosome = population[0]
            logs.append(f"Cromosoma de ejemplo: {sample_chromosome}")
            
            gallery_metrics = calculate_gallery_metrics(sample_chromosome, full_constants, GENE_INDEX_MAP)
            
            logs.append("Metricas calculadas para el cromosoma de ejemplo:")
            logs.append(f"  Inversion Inicial (i_InvIni): {gallery_metrics.get('i_InvIni'):,.2f}")
            logs.append(f"Ejemplo - ROI Porcentual: {gallery_metrics.get('u_ROIGal'):.2f}%")
            logs.append(f"Ejemplo - Utilidad Neta: {gallery_metrics.get('u_UtNeGa'):,.2f}")
            logs.append(f"Ejemplo - Empleo Generado: {gallery_metrics.get('x_Empleo')}")
            logs.append(f"  Ingresos Galeria (u_IngGal): {gallery_metrics.get('u_IngGal'):,.2f}")
            logs.append(f"  Egresos Galeria (u_EgrGal): {gallery_metrics.get('u_EgrGal'):,.2f}")
            logs.append(f"  Utilidad Bruta (u_UtBrGa): {gallery_metrics.get('u_UtBrGa'):,.2f}")
            logs.append(f"  Utilidad Neta (u_UtNeGa): {gallery_metrics.get('u_UtNeGa'):,.2f}")
            logs.append(f"  Margen Utilidad Neta (u_MarUtN): {gallery_metrics.get('u_MarUtN'):.4f}")
            logs.append(f"  ROI Porcentual (u_ROIGal): {gallery_metrics.get('u_ROIGal'):.4f}%")
            logs.append(f"  Beneficio Social (u_BenSoc): {gallery_metrics.get('u_BenSoc'):.4f}")
            logs.append(f"  Empleo Generado (x_Empleo): {gallery_metrics.get('x_Empleo')}")
            logs.append(f"  Total Locales Comerciales (y_ToLoCo): {gallery_metrics.get('y_ToLoCo')}")
            logs.append("-" * 50)
            
            # Calcular aptitud del cromosoma de ejemplo
            fitness_score = calculate_fitness(gallery_metrics, weights)
            logs.append(f"Puntuacion de Aptitud (Fitness) para el cromosoma de ejemplo: {fitness_score:.4f}")
            logs.append("-" * 50)
            
            # Ejemplo de cruce
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            logs.append(f"Padre 1: {parent1}")
            logs.append(f"Padre 2: {parent2}")
            child = crossover_chromosomes(parent1, parent2, crossover_rate)
            final_child = recalculate_dependent_genes(child, full_constants)
            logs.append(f"Hijo despues del cruce: {final_child}")
            
            # Iniciar el algoritmo genetico principal
            logs.append("Iniciando algoritmo genetico principal...")
            
            # Variables para seguimiento de estancamiento y mejora
            best_fitness = -np.inf
            stagnation_count = 0
            improvement_count = 0
            best_chromosome = None
            best_metrics = None

            for generation in range(1, max_generations + 1):
                # Evaluar la aptitud de cada individuo en la poblacion
                fitness_scores = []
                all_metrics = []
                for chromosome in population:
                    metrics = calculate_gallery_metrics(chromosome, full_constants, GENE_INDEX_MAP)
                    fitness = calculate_fitness(metrics, weights)
                    fitness_scores.append(fitness)
                    all_metrics.append(metrics)
                
                # Encontrar la mejor aptitud actual
                current_best_fitness = max(fitness_scores)
                current_best_index = fitness_scores.index(current_best_fitness)
                
                # Actualizar seguimiento de estancamiento y mejora
                if current_best_fitness > best_fitness:
                    improvement_count += 1
                    stagnation_count = 0
                    best_fitness = current_best_fitness
                    best_chromosome = population[current_best_index].copy()
                    best_metrics = all_metrics[current_best_index]
                    logs.append(f"Nueva mejor fitness {best_fitness:.4f} en generacion {generation}")
                else:
                    stagnation_count += 1
                    improvement_count = 0
                
                # Calcular diversidad
                diversity = calculate_diversity(fitness_scores)
                
                # Seleccionar elites
                elites = select_elites(population, fitness_scores, elite_percentage)
                
                # Verificar criterio de parada (aptitud promedio > 0.85 o maximo de generaciones)
                average_fitness = np.mean(fitness_scores)
                if average_fitness > 0.85 or generation == max_generations:
                    logs.append(f"Criterio de parada alcanzado en la generacion {generation}.")
                    logs.append(f"Mejor fitness: {best_fitness:.4f}, Fitness promedio: {average_fitness:.4f}")
                    break
                
                # Ajustar parametros basandose en reglas heuristicas
                params = {
                    'mutation_rate': mutation_rate,
                    'sigma_factor': sigma_factor,
                    'elite_percentage': elite_percentage
                }
                new_params = adjust_parameters(params, diversity, stagnation_count, improvement_count)
                mutation_rate = new_params['mutation_rate']
                sigma_factor = new_params['sigma_factor']
                elite_percentage = new_params['elite_percentage']
                
                # Crear nueva generacion
                new_population = []
                
                # Agregar elites
                new_population.extend(elites)
                
                # Agregar individuos aleatorios (5%)
                num_random = int(0.05 * population_size)
                while len(new_population) < len(elites) + num_random:
                    random_individual = create_initial_population(1, gene_definitions, user_inputs, full_constants)[0]
                    if random_individual not in new_population:
                        new_population.append(random_individual)
                
                # Generar hijos mediante cruce y mutacion hasta completar la poblacion
                while len(new_population) < population_size:
                    # Seleccionar dos padres
                    parent1 = select_parents(population, fitness_scores)
                    parent2 = select_parents(population, fitness_scores)
                    
                    # Cruce
                    child = crossover_chromosomes(parent1, parent2, crossover_rate)
                    
                    # Recalcular genes dependientes despues del cruce
                    child = recalculate_dependent_genes(child, full_constants)
                    
                    # Mutacion
                    child = mutate_chromosome(child, gene_definitions, mutation_rate, sigma_factor)
                    
                    # Recalcular genes dependientes despues de la mutacion
                    child = recalculate_dependent_genes(child, full_constants)
                    
                    # Agregar hijo a la nueva poblacion
                    new_population.append(child)
                
                # Reemplazar la poblacion antigua con la nueva
                population = new_population
                
                # Imprimir progreso cada 10 generaciones
                if generation % 10 == 0:
                    logs.append(f"Progreso - Generacion {generation}/{max_generations}")
                    logs.append(f"Mejor fitness: {best_fitness:.4f}")
                    logs.append(f"Diversidad de poblacion: {diversity:.4f}")
                    logs.append(f"Tasa de mutacion actual: {mutation_rate:.3f}")
                            
            # Mostrar el mejor resultado al finalizar
            logs.append("\n" + "="*60)
            logs.append("MEJOR SOLUCIoN ENCONTRADA")
            logs.append("="*60)
            
            if best_chromosome is not None:
                # Almacenar resultados en app.config
                if 'RESULTS' not in app.config:
                    app.config['RESULTS'] = {}
                
                app.config['RESULTS'][thread_id] = {
                    'best_chromosome': best_chromosome,
                    'best_metrics': best_metrics,
                    'best_fitness': best_fitness
                }

                logs.append("\n" + "="*60)
                logs.append("RESULTADOS ALMACENADOS CORRECTAMENTE EN app.config")
                logs.append("="*60)
                logs.append(f"Cromosoma optimo: {best_chromosome}")
                logs.append(f"Fitness: {best_fitness:.4f}")
            
                
                best_metrics = calculate_gallery_metrics(best_chromosome, full_constants, GENE_INDEX_MAP)
                best_fitness = calculate_fitness(best_metrics, weights)
                
                logs.append(f"Cromosoma optimo: {best_chromosome}")
                logs.append(f"Fitness: {best_fitness:.4f}")
                
                logs.append("Metricas clave de la mejor solucion:")
                logs.append(f"  Inversion Inicial: {best_metrics.get('i_InvIni'):,.2f}")
                logs.append(f"  Ingresos Anuales: {best_metrics.get('u_IngGal'):,.2f}")
                logs.append(f"  Egresos Anuales: {best_metrics.get('u_EgrGal'):,.2f}")
                logs.append(f"  Utilidad Neta: {best_metrics.get('u_UtNeGa'):,.2f}")
                logs.append(f"  ROI: {best_metrics.get('u_ROIGal'):.2f}%")
                logs.append(f"  Beneficio Social: {best_metrics.get('u_BenSoc'):.4f}")
                logs.append(f"  Empleo Generado: {best_metrics.get('x_Empleo')}")
                
                if 'RESULTS' not in app.config:
                    app.config['RESULTS'] = {}

                app.config['RESULTS'][thread_id] = {
                    'best_chromosome': best_chromosome,
                    'best_metrics': best_metrics,
                    'best_fitness': best_fitness
                }

                logs.append("Resultados almacenados correctamente en app.config")
            else:
                logs.append("No se encontro una solucion optima.")
            
            app.config['EXECUTION_LOGS'][thread_id] = logs

        except Exception as e:
            error_msg = f"Error en el algoritmo genético: {str(e)}"
            print(error_msg)
            logs.append(error_msg)
            import traceback
            traceback_str = traceback.format_exc()
            print(traceback_str)
            logs.append(traceback_str)
            app.config['EXECUTION_LOGS'][thread_id] = logs
            
            # Retornar valores por defecto en caso de error
            return best_chromosome, best_metrics, best_fitness
        
        # Retornar los resultados al final de la función
        return best_chromosome, best_metrics, best_fitness

@app.route('/historial')
def historial():
    try:
        ejecuciones = (Ejecucion.query
                      .order_by(Ejecucion.created_at.desc())
                      .limit(200)
                      .all())
        return render_template('historial.html', ejecuciones=ejecuciones or [])
    except Exception as e:
        return render_template('historial.html', error=f"Error al consultar el historial: {e}")

@app.route('/api/check_completion/<thread_id>')
@only_dev
def check_completion(thread_id):
    if 'RESULTS' in app.config and thread_id in app.config['RESULTS']:
        return jsonify({"completed": True})
    return jsonify({"completed": False})

@app.route('/api/diagnostic')
@only_dev
def diagnostic():
    """Ruta de diagnostico para verificar el estado de los threads"""
    try:
        active_threads = list(app.config.get('EXECUTION_LOGS', {}).keys())
        return jsonify({
            'active_threads': active_threads,
            'results_available': list(app.config.get('RESULTS', {}).keys()),
            'thread_count': len(active_threads)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detalle_ejecucion/<int:ejecucion_id>')
def detalle_ejecucion(ejecucion_id):
    try:
        e = Ejecucion.query.get_or_404(ejecucion_id)

        # Parámetros (de la fila)
        params = {
            "population_size": getattr(e, 'poblacion_inicial', None),
            "max_generations": getattr(e, 'generaciones', None),
            "elite_percentage": getattr(e, 'elite_percentage', None),
            "mutation_rate": getattr(e, 'tasa_mutacion', None),
            "sigma_factor": getattr(e, 'sigma_factor', None),
            "crossover_rate": getattr(e, 'crossover_rate', None),
            "weights": (getattr(e, 'peso_be', None),
                        getattr(e, 'peso_bs', None),
                        getattr(e, 'peso_mun', None))
        }

        return render_template('detalle_ejecucion.html',
                               run_id=getattr(e, 'run_id', None),
                               ejecucion=e,
                               params=params)
    except Exception as ex:
        return render_template('detalle_ejecucion.html',
                               error=f"Error al cargar el detalle: {str(ex)}")

@app.route("/comparativo")
def comparativo():
    run_id = request.args.get("run_id")

    q = Ejecucion.query.order_by(Ejecucion.comuna.asc(), Ejecucion.fecha_ejecucion.desc())
    if run_id:
        q = q.filter(Ejecucion.run_id == run_id)

    # Tomamos la última por comuna (1..7). Si ya lo haces en SQL, perfecto; si no:
    ejecs_por_comuna = {}

    for e in q.all():
        # guarda solo la primera que veas por comuna (ya viene desc por fecha)
        ejecs_por_comuna.setdefault(e.comuna, e)

    print(">>> comparativo ejecuciones:", ejecs_por_comuna)
    ejecuciones = [ejecs_por_comuna[k] for k in sorted(ejecs_por_comuna.keys())]

    # Helper para 0 si None
    def z(x): 
        return 0 if x is None else x

    labels = [f"Comuna {e.comuna}" if e.comuna != 7 else "Comuna 7 (Nueva)" for e in ejecuciones]

    # Inicializar arreglos
    be_inv_total = []
    be_inv_loc   = []; be_inv_parq  = []; be_inv_zonas = []
    be_ing_total = []; be_ing_arr   = []; be_ing_adm   = []; be_ing_parq = []
    be_egr_total = []; be_egr_mant  = []; be_egr_serv  = []; be_egr_sal  = []
    be_egr_ope   = []; be_egr_adm   = []; be_egr_leg   = []; be_egr_imp  = []
    bs_acces     = []; bs_emp_dir   = []; bs_emp_ind   = []; bs_calidad  = []
    ar_af        = []; ar_cp        = []; ar_nal       = []; ar_sc       = []
    idx_mun      = []; idx_bc       = []; idx_bs_idx      = []

    for e in ejecuciones:
        # trae su detalle (1–a–1). Si definiste relationship one-to-one "detalles" como lista, ajusta:
        det = None
        # Caso 1: relationship one-to-one
        if hasattr(e, "detalles") and isinstance(e.detalles, list):
            det = e.detalles[0] if e.detalles else None
        # Caso 2: relationship one-to-one con atributo singular (ajusta si lo llamaste distinto)
        if hasattr(e, "detalle") and det is None:
            det = e.detalle

        # si no hay detalle, todo 0
        if det is None:
            be_inv_total.append( z(e.inv_inicial_usd) )
            be_inv_loc.append(0); be_inv_parq.append(0); be_inv_zonas.append(0)
            be_ing_total.append(0); be_ing_arr.append(0); be_ing_adm.append(0); be_ing_parq.append(0)
            be_egr_total.append(0); be_egr_mant.append(0); be_egr_serv.append(0); be_egr_sal.append(0)
            be_egr_ope.append(0); be_egr_adm.append(0); be_egr_leg.append(0); be_egr_imp.append(0)
            bs_acces.append(0); bs_emp_dir.append(z(e.empleos_directos)); bs_emp_ind.append(0); bs_calidad.append(0)
            ar_af.append(0); ar_cp.append(0); ar_nal.append(0); ar_sc.append(0)
        else:
            be_inv_total.append( z(det.inv_total or e.inv_inicial_usd) )  # usa inv_total si lo guardaste; si no, el agregado de Ejecucion
            be_inv_loc.append(   z(det.inv_loc) )
            be_inv_parq.append(  z(det.inv_parq) )
            be_inv_zonas.append( z(det.inv_zonas) )

            be_ing_total.append( z(det.ing_total) )
            be_ing_arr.append(   z(det.ing_arr) )
            be_ing_adm.append(   z(det.ing_adm) )
            be_ing_parq.append(  z(det.ing_parq) )

            be_egr_total.append( z(det.egr_total) )
            be_egr_mant.append(  z(det.egr_mant) )
            be_egr_serv.append(  z(det.egr_servpub) )
            be_egr_sal.append(   z(det.egr_salarios) )
            be_egr_ope.append(   z(det.egr_operativos) )
            be_egr_adm.append(   z(det.egr_admin) )
            be_egr_leg.append(   z(det.egr_legales) )
            be_egr_imp.append(   z(det.egr_impuestos) )

            bs_acces.append(     z(det.bs_accesibilidad) )   # << si en DB está NULL, aquí lo forzamos a 0
            bs_emp_dir.append(   z(det.bs_emp_dir) )
            bs_emp_ind.append(   z(det.bs_emp_ind) )
            bs_calidad.append(   z(det.bs_calidad_vida) )    # << idem

            ar_af.append(  z(det.ar_alimentos_frescos) )
            ar_cp.append(  z(det.ar_comidas_preparadas) )
            ar_nal.append( z(det.ar_no_alimentarios) )
            ar_sc.append(  z(det.ar_complementarios) )

        # índices (si los calculas en Ejecucion: usa esos campos; si no, 0)
        idx_mun.append( z(e.peso_mun) )   # o tu índice real de MUN si lo tienes
        # B/C: si tienes utilidad/ingresos/egresos, puedes calcularlo. De momento toma utilidad / inversión si aplica:
        try:
            bc = (e.utilidad_neta_usd or 0) / (e.inv_inicial_usd or 1)
        except ZeroDivisionError:
            bc = 0
        idx_bc.append(bc)
        idx_bs_idx.append( z(e.beneficio_social) )

        def pick(seq, i):
            try:
                v = seq[i]
                return 0 if v is None else v
            except Exception:
                return 0

        tabla_rows = []
        for i, label in enumerate(labels):
            fila = {
                "label": label,

                # Económico - inversión
                "inv_total":   pick(be_inv_total, i),
                "inv_loc":     pick(be_inv_loc, i),
                "inv_parq":    pick(be_inv_parq, i),
                "inv_zonas":   pick(be_inv_zonas, i),

                # Ingresos/Egresos totales
                "ing_total":   pick(be_ing_total, i),
                "egr_total":   pick(be_egr_total, i),

                # Ingresos detalle
                "ing_arr":     pick(be_ing_arr, i),
                "ing_adm":     pick(be_ing_adm, i),
                "ing_parq":    pick(be_ing_parq, i),

                # Egresos detalle
                "egr_mant":    pick(be_egr_mant, i),
                "egr_serv":    pick(be_egr_serv, i),
                "egr_sal":     pick(be_egr_sal, i),
                "egr_ope":     pick(be_egr_ope, i),
                "egr_adm":     pick(be_egr_adm, i),
                "egr_leg":     pick(be_egr_leg, i),
                "egr_imp":     pick(be_egr_imp, i),

                # Beneficio social
                "bs_acces":    pick(bs_acces, i),
                "bs_emp_dir":  pick(bs_emp_dir, i),
                "bs_emp_ind":  pick(bs_emp_ind, i),
                "bs_calidad":  pick(bs_calidad, i),

                # Áreas comerciales
                "ar_af":       pick(ar_af, i),
                "ar_cp":       pick(ar_cp, i),
                "ar_nal":      pick(ar_nal, i),
                "ar_sc":       pick(ar_sc, i),

                # Índices
                "idx_mun":     pick(idx_mun, i),
                "idx_bc":      pick(idx_bc, i),
                "idx_bs_idx":     pick(idx_bs_idx, i),
            }
            tabla_rows.append(fila)

    return render_template(
        "comparativo.html",
        run_id=run_id,
        ejecuciones=ejecuciones,
        # JSON para el cliente
        labels=labels,
        be_inv_total=be_inv_total,
        be_inv_loc=be_inv_loc, be_inv_parq=be_inv_parq, be_inv_zonas=be_inv_zonas,
        be_ing_total=be_ing_total, be_ing_arr=be_ing_arr, be_ing_adm=be_ing_adm, be_ing_parq=be_ing_parq,
        be_egr_total=be_egr_total, be_egr_mant=be_egr_mant, be_egr_serv=be_egr_serv, be_egr_sal=be_egr_sal,
        be_egr_ope=be_egr_ope, be_egr_adm=be_egr_adm, be_egr_leg=be_egr_leg, be_egr_imp=be_egr_imp,
        bs_acces=bs_acces, bs_emp_dir=bs_emp_dir, bs_emp_ind=bs_emp_ind, bs_calidad=bs_calidad,
        ar_af=ar_af, ar_cp=ar_cp, ar_nal=ar_nal, ar_sc=ar_sc,
        idx_mun=idx_mun, idx_bc=idx_bc, idx_bs_idx=idx_bs_idx, tabla_rows=tabla_rows
    )

if __name__ == '__main__':
    # Inicializar estructuras de datos si no existen
    if 'EXECUTION_LOGS' not in app.config:
        app.config['EXECUTION_LOGS'] = {}
        print("EXECUTION_LOGS inicializado")
        
    if 'RESULTS' not in app.config:
        app.config['RESULTS'] = {}
        print("RESULTS inicializado")

    app.run(debug=True, threaded=True)