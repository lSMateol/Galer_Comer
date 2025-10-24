from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash,current_app, abort, g, Response
from datetime import datetime
from utils.genetic_algorithm import *
from extensions import db
from models import *
from sqlalchemy import text, func, cast, Integer
from dotenv import load_dotenv
from functools import wraps
import uuid
import traceback
import numpy as np
import random
import threading
import time
import os
import io
import csv
from flask_cors import CORS

load_dotenv()

ANON_COOKIE = "user_key"

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

# CONFIGURACIÓN CORS MEJORADA PARA APP MÓVIL
CORS(app, 
     resources={
         r"/*": {
             "origins": [
                 "http://localhost:8100", 
                 "http://localhost:4200",
                 "http://192.168.0.112:5000",
                 "http://192.168.0.112", 
                 "capacitor://localhost", 
                 "http://localhost",
                 "https://localhost"  
             ],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "User-Key"],
             "supports_credentials": True,
             "max_age": 86400  # 24 horas
         }
     },
     supports_credentials=True)

app.config['EXECUTION_LOGS'] = {}
app.config['EXECUTION_PROGRESS'] = {}  
app.config['RESULTS'] = {}
# Configuracion de la base de datos - FORMA CORRECTA
app.config['SQLALCHEMY_DATABASE_URI'] = db_url or "sqlite:///local.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["ENVIRONMENT"] = ENVIRONMENT
app.config["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "INFO")

db.init_app(app)

# Manejo de pre-flight OPTIONS para CORS
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", 
                           request.headers.get("Origin", "*"))
        response.headers.add("Access-Control-Allow-Headers", 
                           "Content-Type,Authorization,X-Requested-With,Accept,User-Key")
        response.headers.add("Access-Control-Allow-Methods", 
                           "GET,POST,PUT,DELETE,OPTIONS")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response

# Headers de seguridad adicionales
@app.after_request
def after_request(response):
    # Headers adicionales para seguridad en móvil
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('X-Frame-Options', 'DENY')
    response.headers.add('X-XSS-Protection', '1; mode=block')
    response.headers.add('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    
    return response

def only_dev(f):
    @wraps(f)
    def _w(*args, **kwargs):
        if current_app.config.get("ENVIRONMENT") == "production":
            abort(404)
        return f(*args, **kwargs)
    return _w

def get_or_create_user_key(resp=None):
    """
    Obtiene el user_key de los headers HTTP (prioridad) o genera uno nuevo
    Para la API móvil que usa headers en lugar de cookies
    """
    # 1. PRIORIDAD: Obtener del header 'User-Key'
    uk = request.headers.get('User-Key')
    
    if uk:
        print(f"✅ User-Key obtenido del header: {uk}")
        return uk
    
    # 2. FALLBACK: Intentar de la cookie (para compatibilidad con web)
    uk = request.cookies.get(ANON_COOKIE)
    if uk:
        print(f"🔄 User-Key obtenido de cookie: {uk}")
        return uk
    
    # 3. ÚLTIMO RECURSO: Generar nuevo user_key
    uk = str(uuid.uuid4())
    print(f"🆕 User-Key generado: {uk}")
    
    # 4. Opcional: Setear cookie solo si se solicita (para compatibilidad web)
    # Pero para móvil, el frontend maneja el almacenamiento
    if resp is not None:
        resp.set_cookie(ANON_COOKIE, uk, max_age=60*60*24*180,
                        httponly=True, samesite="Lax", secure=True)
        print(f"🍪 Cookie set for new user_key: {uk}")
    else:
        # Para contexto móvil, solo almacenar en g si es necesario
        g._set_user_cookie = uk
        print(f"📱 User-Key almacenado en contexto: {uk}")
    
    return uk

@app.after_request
def _set_user_cookie(resp):
    uk = getattr(g, "_set_user_cookie", None)
    if uk:
        resp.set_cookie(ANON_COOKIE, uk, max_age=60*60*24*180,
                        httponly=True, samesite="Lax", secure=True)
    return resp

def _ns(user_key: str, thread_id: str) -> str:
    return f"{user_key}:{thread_id}"

@app.route('/')
def index():
    return render_template('index.html')

def guardar_resumen_run(run_id: str, user_key: str):
    # Agregados básicos de las 7 filas del run
    agg = (db.session.query(
            func.count(Ejecucion.id),
            func.sum(Ejecucion.inv_inicial_usd),
            func.sum(Ejecucion.utilidad_neta_usd),
            func.avg(Ejecucion.roi),
            func.avg(Ejecucion.margen_utilidad),
            func.avg(Ejecucion.mejor_fitness),
            func.avg(Ejecucion.beneficio_social),
        )
        .filter(Ejecucion.run_id == run_id, Ejecucion.user_key == user_key)
        .one()
    )

    filas, sum_inv, sum_ut, avg_roi, avg_margen, avg_fit, avg_ben = agg

    # Mejor comuna base (1..6) por ROI
    best_base = (Ejecucion.query
                 .filter(Ejecucion.user_key == user_key,
                         Ejecucion.run_id == run_id,
                         Ejecucion.comuna <= 6)
                 .order_by(Ejecucion.roi.desc())
                 .first())

    mejor_comuna_base = best_base.comuna if best_base else None
    mejor_roi_base    = best_base.roi if best_base else None

    # UPSERT (create or update)
    resumen = (ResumenRun.query
               .filter_by(user_key=user_key, run_id=run_id)
               .first())
    if not resumen:
        resumen = ResumenRun(
            user_key=user_key,
            run_id=run_id,
            total_inversion = float(sum_inv or 0),
            total_utilidad  = float(sum_ut  or 0),
            prom_roi        = float(avg_roi or 0),
            prom_margen     = float(avg_margen or 0),
            prom_fitness    = float(avg_fit or 0),
            prom_ben_social = float(avg_ben or 0),
            mejor_comuna_base = mejor_comuna_base,
            mejor_roi_base    = float(mejor_roi_base or 0),
        )
        db.session.add(resumen)
    else:
        resumen.total_inversion = float(sum_inv or 0)
        resumen.total_utilidad  = float(sum_ut  or 0)
        resumen.prom_roi        = float(avg_roi or 0)
        resumen.prom_margen     = float(avg_margen or 0)
        resumen.prom_fitness    = float(avg_fit or 0)
        resumen.prom_ben_social = float(avg_ben or 0)
        resumen.mejor_comuna_base = mejor_comuna_base
        resumen.mejor_roi_base    = float(mejor_roi_base or 0)

    db.session.commit()
    return resumen

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

        uk = get_or_create_user_key()
        nskey = _ns(uk, thread_id)

        # Inicializar logs si no existen
        if 'EXECUTION_LOGS' not in app.config:
            app.config['EXECUTION_LOGS'] = {}
        app.config['EXECUTION_LOGS'][nskey] = []
        
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
                    run_id,uk
                )
            except Exception as e:
                app.config['EXECUTION_LOGS'][nskey].append(f"ERROR: {e}")
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
                  run_id,uk)
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

    uk = get_or_create_user_key()
    nskey = _ns(uk, thread_id)

    if 'EXECUTION_LOGS' not in app.config:
        app.config['EXECUTION_LOGS'] = {}
    app.config['EXECUTION_LOGS'][nskey] = []
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
                run_id, uk
            )
        except Exception as e:
            # Registra el error en logs del thread para trazabilidad
            app.config['EXECUTION_LOGS'][nskey].append(f"ERROR: {e}")
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
                    run_id, uk
                )
            except Exception as e:
                # Guarda el error en los logs de ejecución
                app.config['EXECUTION_LOGS'][nskey].append(f"ERROR: {e}")

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
                            run_id, user_key):
    """
    Nota: requiere que el modelo Ejecucion tenga la columna:
      run_id = db.Column(db.String(36), index=True, nullable=False)
    """
    nskey = f"{user_key}:{thread_id}"

    with app.app_context():
        # INICIALIZACIÓN ROBUSTA
        if 'EXECUTION_LOGS' not in app.config:
            app.config['EXECUTION_LOGS'] = {}
        if 'EXECUTION_PROGRESS' not in app.config:
            app.config['EXECUTION_PROGRESS'] = {}
        if 'RESULTS' not in app.config:
            app.config['RESULTS'] = {}
            
        logs = app.config['EXECUTION_LOGS'].get(nskey, [])
        
        # INICIALIZAR PROGRESO DE FORMA EXPLÍCITA
        app.config['EXECUTION_PROGRESS'][nskey] = {
            'total': len(galerias_a_procesar),
            'completadas': 0,
            'progreso': 0,
            'estado_actual': 'Iniciando procesamiento...',
            'galerias_completadas': 0,
            'galerias_totales': len(galerias_a_procesar)
        }
        
        app.config['EXECUTION_LOGS'][nskey] = logs
        
        print(f"✅ INICIALIZADO: nskey={nskey}, progreso=0%")

        resultados_galerias = {}
        
        try:
            # Procesar cada una de las 7 galerías
            for idx, galeria in enumerate(galerias_a_procesar):
                galeria_num = galeria['numero']

                # ACTUALIZAR PROGRESO ANTES DE CADA GALERÍA
                app.config['EXECUTION_PROGRESS'][nskey]['estado_actual'] = f'Procesando Galería {galeria_num}...'
                app.config['EXECUTION_PROGRESS'][nskey]['completadas'] = idx
                app.config['EXECUTION_PROGRESS'][nskey]['progreso'] = int((idx / len(galerias_a_procesar)) * 80)  # 80% para las 7 galerías
                print(f"DEBUG: Progreso actualizado - Galería {galeria_num} - {app.config['EXECUTION_PROGRESS'][nskey]['progreso']}%")

                # --- VERIFICACIÓN para evitar duplicar dentro del MISMO run_id ---
                existing_execution = Ejecucion.query.filter_by(comuna=galeria_num, run_id=run_id).first()
                if existing_execution:
                    logs.append(f"GALERIA_{galeria_num} ya existe en este run_id={run_id}. Se omite la inserción duplicada.")
                    app.config['EXECUTION_LOGS'][nskey] = logs
                    continue
                # -----------------------------------------------------------------

                logs.append(f"Procesando Galeria {galeria_num}...")
                app.config['EXECUTION_LOGS'][nskey] = logs
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
                result = run_genetic_algorithm(
                    app, 
                    f"{thread_id}_galeria_{galeria_num}",
                    user_inputs, 
                    constants,
                    population_size, 
                    max_generations,
                    elite_percentage,
                    mutation_rate,
                    sigma_factor,
                    crossover_rate,
                    weights,
                    None,
                    user_key
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
                            porcentaje_elite=elite_percentage,
                            fuerza_sigma=sigma_factor,
                            tasa_cruzamiento=crossover_rate,

                            mejor_fitness=best_fitness,
                            inv_inicial_usd=best_metrics.get('i_InvIni', 0.0),
                            roi=best_metrics.get('u_ROIGal', 0.0),
                            utilidad_neta_usd=best_metrics.get('u_UtNeGa', 0.0),
                            margen_utilidad=best_metrics.get('u_MarUtN', 0.0),
                            empleos_directos=best_metrics.get('x_Empleo', 0.0),
                            beneficio_social=best_metrics.get('u_BenSoc', 0.0),
                            cromosoma_optimo=best_chromosome,

                            locales_12=int(best_metrics.get('l_CLTi12', 0) or 0),
                            locales_16=int(best_metrics.get('l_CLTi16', 0) or 0),
                            locales_20=int(best_metrics.get('l_CLTi20', 0) or 0),
                            locales_25=int(best_metrics.get('l_CLTi25', 0) or 0),

                            run_id=run_id,
                            user_key=user_key  
                        )
                        db.session.add(nueva_ejecucion)
                        db.session.commit()
                        logs.append(f"✅ GALERIA_{galeria_num} GUARDADA EN BD - ID: {nueva_ejecucion.id}")
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
                        logs.append(f"TRACEBACK: {traceback.format_exc()}")
                
                # ACTUALIZAR PROGRESO DESPUÉS DE CADA GALERÍA
                app.config['EXECUTION_PROGRESS'][nskey]['completadas'] = idx + 1
                app.config['EXECUTION_PROGRESS'][nskey]['progreso'] = int(((idx + 1) / len(galerias_a_procesar)) * 80)
                logs.append(f"✅ Galeria {galeria_num} completada - Progreso: {app.config['EXECUTION_PROGRESS'][nskey]['progreso']}%")
                app.config['EXECUTION_LOGS'][nskey] = logs
            
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
                        app.config['EXECUTION_LOGS'][nskey] = logs
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

                # ACTUALIZAR PROGRESO PARA GALERÍA 7
                app.config['EXECUTION_PROGRESS'][nskey]['estado_actual'] = f'Optimizando Galería 7 para Comuna {mejor_comuna}...'
                app.config['EXECUTION_PROGRESS'][nskey]['progreso'] = 85
                logs.append(f"Optimizando Galeria 7 para la Comuna {mejor_comuna}...")
                app.config['EXECUTION_LOGS'][nskey] = logs

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
                    None,
                    user_key
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
                        porcentaje_elite=elite_percentage,
                        fuerza_sigma=sigma_factor,
                        tasa_cruzamiento=crossover_rate,

                        mejor_fitness=best_fitness_7,
                        inv_inicial_usd=best_metrics_7.get('i_InvIni', 0.0),
                        roi=best_metrics_7.get('u_ROIGal', 0.0),
                        utilidad_neta_usd=best_metrics_7.get('u_UtNeGa', 0.0),
                        margen_utilidad=best_metrics_7.get('u_MarUtN', 0.0),
                        empleos_directos=best_metrics_7.get('x_Empleo', 0.0),
                        beneficio_social=best_metrics_7.get('u_BenSoc', 0.0),
                        cromosoma_optimo=best_chromosome_7,
                        
                        locales_12=int(best_metrics_7.get('l_CLTi12', 0) or 0),
                        locales_16=int(best_metrics_7.get('l_CLTi16', 0) or 0),
                        locales_20=int(best_metrics_7.get('l_CLTi20', 0) or 0),
                        locales_25=int(best_metrics_7.get('l_CLTi25', 0) or 0),

                        user_key=user_key,
                        run_id=run_id
                    )
                    db.session.add(ejec_7)
                    db.session.commit()
                    logs.append(f"✅ GALERIA_{7} GUARDADA EN BD - ID: {nueva_ejecucion.id}")
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
                        logs.append(f"[WARN] No se pudo guardar detalle de la Galería 7: {det_e}")

                    logs.append(f"GALERIA_7_GUARDADA (comuna óptima {mejor_comuna}) ROI:{best_metrics_7.get('u_ROIGal', 0):.2f} (run_id={run_id})")
                except Exception as e:
                    db.session.rollback()
                    logs.append(f"Error al guardar Galeria 7: {str(e)}")

                # COMPLETAR PROGRESO
                app.config['EXECUTION_PROGRESS'][nskey]['completadas'] = app.config['EXECUTION_PROGRESS'][nskey]['total']
                app.config['EXECUTION_PROGRESS'][nskey]['progreso'] = 100
                app.config['EXECUTION_PROGRESS'][nskey]['estado_actual'] = 'Procesamiento completado'

                # Almacenar resultados finales para UI
                if 'RESULTS' not in app.config:
                    app.config['RESULTS'] = {}
                
                app.config['RESULTS'][nskey] = {
                    'best_chromosome': galeria_7.get('best_chromosome'),
                    'best_metrics': galeria_7.get('best_metrics'),
                    'best_fitness': galeria_7.get('best_fitness'),
                    'mejor_comuna': mejor_comuna,
                    'resultados_galerias': resultados_galerias,
                    'run_id': run_id
                }
                
                logs.append("✅ PROCESAMIENTO COMPLETADO")

                try:
                    guardar_resumen_run(run_id, user_key)
                    logs.append(f"[OK] Resumen guardado para run_id={run_id}")
                except Exception as e:
                    db.session.rollback()
                    logs.append(f"[WARN] No se pudo guardar el resumen del run: {e}")
                app.config['EXECUTION_LOGS'][nskey] = logs

        except Exception as e:
            error_msg = f"Error en el procesamiento: {str(e)}"
            logs.append(error_msg)
            traceback_str = traceback.format_exc()
            logs.append(traceback_str)
            
            # ACTUALIZAR PROGRESO EN CASO DE ERROR
            app.config['EXECUTION_PROGRESS'][nskey]['estado_actual'] = f'Error: {str(e)}'
            
            if 'RESULTS' not in app.config:
                app.config['RESULTS'] = {}
                
            app.config['RESULTS'][nskey] = {
                'error': error_msg,
                'traceback': traceback_str,
                'run_id': run_id
            }
            
            app.config['EXECUTION_LOGS'][nskey] = logs

@app.route('/ejecucion')
def ejecucion():
    thread_id = request.args.get('thread_id', session.get('thread_id', ''))
    return render_template('ejecucion.html', thread_id=thread_id)

@app.route('/api/logs/<thread_id>')
def get_logs(thread_id):
    uk = get_or_create_user_key()
    nskey = _ns(uk, thread_id)
    logs = app.config.get('EXECUTION_LOGS', {}).get(nskey, [])
    
    # Verificar si hay resultados disponibles
    has_results = 'RESULTS' in app.config and nskey in app.config['RESULTS']
    
    # Determinar el estado
    status = "ejecutando"
    if has_results:
        status = "completado"
    elif any("MEJOR SOLUCIoN ENCONTRADA" in log for log in logs):
        status = "completado"
    
    return jsonify({"logs": logs, "status": status, "has_results": has_results})

@app.route('/api/debug/progress')
def debug_progress():
    """Endpoint de diagnóstico mejorado"""
    progress_data = {}
    for key, value in app.config.get('EXECUTION_PROGRESS', {}).items():
        progress_data[key] = {
            'progreso': value.get('progreso', 0),
            'estado': value.get('estado_actual', 'Unknown'),
            'completadas': value.get('completadas', 0),
            'total': value.get('total', 0)
        }
    
    return jsonify({
        'EXECUTION_PROGRESS': progress_data,
        'EXECUTION_LOGS_keys': list(app.config.get('EXECUTION_LOGS', {}).keys()),
        'RESULTS_keys': list(app.config.get('RESULTS', {}).keys()),
        'total_threads': len(app.config.get('EXECUTION_PROGRESS', {}))
    })

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

    uk = get_or_create_user_key()

    # 1) Leer resultados persistidos (BD) por run_id
    ejecuciones = (Ejecucion.query
                   .filter_by(run_id=run_id, user_key=uk)
                   .order_by(Ejecucion.comuna.asc())
                   .all())

    # 2) Estado de la corrida
    total_guardadas = len(ejecuciones)
    completa = (total_guardadas >= 7)  # esperamos 7 filas

    # 3) Progreso en vivo (logs) mientras corre
    logs = []
    if thread_id and 'EXECUTION_LOGS' in app.config:
        uk = get_or_create_user_key()
        nskey = _ns(uk, thread_id)
        logs = app.config['EXECUTION_LOGS'].get(nskey, [])

    # 4) KPIs de la mejor solución (desde BD, mejor ROI del run_id)
    resumen_worker = None
    if ejecuciones:
        mejor_fila_global = max(ejecuciones, key=lambda e: (e.roi or 0))
        resumen_worker = {
            "best_metrics": {
                # mismas llaves que usas en el template actual
                "u_UtNeGa": mejor_fila_global.utilidad_neta_usd or 0,
                "u_ROIGal": mejor_fila_global.roi or 0,
                "u_BenSoc": mejor_fila_global.beneficio_social or 0,
            },
            "best_chromosome": mejor_fila_global.cromosoma_optimo or [],
            "best_fitness": mejor_fila_global.mejor_fitness or 0,
        }

    # 5) KPIs útiles: mejor ROI entre 1..6 (para "mejor comuna" de las existentes)
    mejor_comuna, mejor_roi = None, None
    if ejecuciones:
        base = [e for e in ejecuciones if e.comuna and e.comuna <= 6]
        if base:
            best = max(base, key=lambda e: (e.roi or 0))
            mejor_comuna = best.comuna
            mejor_roi = best.roi

    # === Agregado: tabla de locales por tipo y totales ===
    def _zint(x):
        try:
            return 0 if x is None else int(x)
        except Exception:
            return 0

    locales_rows = []
    totales_locales = {"l12": 0, "l16": 0, "l20": 0, "l25": 0}

    for e in ejecuciones:
        fila = {
            "comuna": e.comuna,
            "l12": _zint(getattr(e, "locales_12", 0)),
            "l16": _zint(getattr(e, "locales_16", 0)),
            "l20": _zint(getattr(e, "locales_20", 0)),
            "l25": _zint(getattr(e, "locales_25", 0)),
        }
        locales_rows.append(fila)
        totales_locales["l12"] += fila["l12"]
        totales_locales["l16"] += fila["l16"]
        totales_locales["l20"] += fila["l20"]
        totales_locales["l25"] += fila["l25"]

    if completa:
        try:
            guardar_resumen_run(run_id, uk)
        except Exception as e:
            current_app.logger.warning(f"No se pudo guardar resumen_run: {e}")
    resumen_run = ResumenRun.query.filter_by(user_key=uk, run_id=run_id).first()
    # 6) Render al template
    return render_template(
        'resultados.html',
        run_id=run_id,
        completa=completa,
        ejecuciones=ejecuciones,   # 0..7 filas por run_id
        params=params,             # parámetros usados
        logs=logs,                 # progreso del modal
        resumen_worker=resumen_worker,  # ahora viene de la BD
        mejor_comuna=mejor_comuna,
        mejor_roi=mejor_roi,
        resumen_run=resumen_run,
        locales_rows=locales_rows,
        totales_locales=totales_locales
    )

def run_genetic_algorithm(app, thread_id, user_inputs, constants, 
                         population_size, max_generations, elite_percentage,
                         mutation_rate, sigma_factor, crossover_rate, weights,
                         galerias_existentes, user_key):
    with app.app_context():
        nskey = _ns(user_key, thread_id)

        # Asegurate de que el array de logs existe
        if nskey not in app.config['EXECUTION_LOGS']:
            app.config['EXECUTION_LOGS'][nskey] = []

        logs = app.config['EXECUTION_LOGS'][nskey]

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
                app.config['EXECUTION_LOGS'][nskey] = logs
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

        if galerias_existentes is not None:
            for i, galeria in enumerate(galerias_existentes):
                logs.append(f"Galeria Comuna {galeria['comuna']} procesada - ROI: {galeria.get('roi', 0):.2f}%")
                app.config['EXECUTION_LOGS'][nskey] = logs
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
        app.config['EXECUTION_LOGS'][nskey] = logs
        print(f"Logs iniciales configurados para thread {thread_id}")

        try:
            # Combinar constantes globales con constantes especificas de esta ejecucion
            from utils.genetic_algorithm import CONSTANTS as GLOBAL_CONSTANTS
            full_constants = {**GLOBAL_CONSTANTS, **constants}

            logs.append("Constantes combinadas correctamente")
            app.config['EXECUTION_LOGS'][nskey] = logs

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

            app.config['EXECUTION_LOGS'][nskey] = logs
            print(f" Constantes combinadas para thread {thread_id}")
            
            # Inicializar poblacion
            population = create_initial_population(population_size, gene_definitions, user_inputs, full_constants)
            
            logs.append(f"Poblacion inicial creada con {len(population)} individuos")
            app.config['EXECUTION_LOGS'][nskey] = logs

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
                
                app.config['RESULTS'][nskey] = {
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

                app.config['RESULTS'][nskey] = {
                    'best_chromosome': best_chromosome,
                    'best_metrics': best_metrics,
                    'best_fitness': best_fitness
                }

                logs.append("Resultados almacenados correctamente en app.config")
            else:
                logs.append("No se encontro una solucion optima.")
            
            app.config['EXECUTION_LOGS'][nskey] = logs

        except Exception as e:
            error_msg = f"Error en el algoritmo genético: {str(e)}"
            print(error_msg)
            logs.append(error_msg)
            traceback_str = traceback.format_exc()
            print(traceback_str)
            logs.append(traceback_str)
            app.config['EXECUTION_LOGS'][nskey] = logs
            
            # Retornar valores por defecto en caso de error
            return best_chromosome, best_metrics, best_fitness
        
        # Retornar los resultados al final de la función
        return best_chromosome, best_metrics, best_fitness

@app.route("/historial")
def historial():
    """
    Lista las corridas del usuario actual (anónimo) agrupadas por run_id.
    Muestra: fecha última, si está completa (7/7), y el mejor ROI de esa corrida.
    Permite filtrar por run_id (?q=) y paginar (?page=, ?per_page=).
    """
    uk = get_or_create_user_key()  # ← identifica al "usuario" por cookie

    # Filtros opcionales
    term = (request.args.get("q") or "").strip()
    page = int(request.args.get("page", 1) or 1)
    per_page = int(request.args.get("per_page", 10) or 10)

    # Agregado por corrida (run_id) SOLO del user_key actual
    base = (
        db.session.query(
            Ejecucion.run_id.label("run_id"),
            func.count(Ejecucion.id).label("filas"),
            func.max(Ejecucion.fecha_ejecucion).label("ultima_fecha"),
            func.max(Ejecucion.roi).label("mejor_roi"),
        )
        .filter(Ejecucion.user_key == uk)                     # ← filtro clave
        .group_by(Ejecucion.run_id)
        .order_by(func.max(Ejecucion.fecha_ejecucion).desc())
    )

    if term:
        base = base.filter(Ejecucion.run_id.ilike(f"%{term}%"))

    # Paginar (si usas Flask-SQLAlchemy ≥3, usa .paginate sobre db.session.execute/Select)
    try:
        pag = base.paginate(page=page, per_page=per_page, error_out=False)
        rows = pag.items
        total = pag.total
        pages = pag.pages
    except Exception:
        # Fallback sin paginate si tu versión no lo soporta en queries agregadas
        rows = base.all()
        total = len(rows)
        pages = 1

    # Empaquetar para el template
    corridas = []
    for r in rows:
        corridas.append({
            "run_id": r.run_id,
            "filas": r.filas,
            "completa": (r.filas or 0) >= 7,
            "ultima_fecha": r.ultima_fecha,
            "mejor_roi": r.mejor_roi or 0,
        })

    return render_template(
        "historial.html",
        corridas=corridas,
        q=term,
        page=page,
        per_page=per_page,
        total=total,
        pages=pages,
    )

@app.route('/api/check_completion/<thread_id>')
@only_dev
def check_completion(thread_id):
    uk = get_or_create_user_key()
    nskey = _ns(uk, thread_id)
    if 'RESULTS' in app.config and nskey in app.config['RESULTS']:
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
            "elite_percentage": getattr(e, 'porcentaje_elite', None),
            "mutation_rate": getattr(e, 'tasa_mutacion', None),
            "sigma_factor": getattr(e, 'fuerza_sigma', None),
            "crossover_rate": getattr(e, 'tasa_cruzamiento', None),
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
    uk = get_or_create_user_key()

    q = (Ejecucion.query
        .filter(Ejecucion.user_key == uk)  
        .order_by(Ejecucion.comuna.asc(), Ejecucion.fecha_ejecucion.desc()))
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
    idx_mun      = []; idx_bc       = []; idx_bs_idx      = []; idx_fitness = []

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
        idx_fitness.append( z(e.mejor_fitness) )

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
            "idx_bs_idx":  pick(idx_bs_idx, i),
            "idx_fitness": pick(idx_fitness, i),
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
        idx_mun=idx_mun, idx_bc=idx_bc, idx_bs_idx=idx_bs_idx, idx_fitness=idx_fitness,tabla_rows=tabla_rows
    )

@app.route("/resumen-corrida")
def resumen_corrida():
    # 🔥 GLOBAL: no usamos user_key (sin backfill aquí para mantenerlo simple)

    # Filtros
    only_run = (request.args.get("run_id") or "").strip()
    limit = int(request.args.get("limit", 50) or 50)

    # ResumenRun global
    q = ResumenRun.query
    if only_run:
        q = q.filter(ResumenRun.run_id == only_run)

    filas = (q.order_by(ResumenRun.prom_fitness.desc(),
                        ResumenRun.created_at.desc())
               .limit(limit)
               .all())

    # Agregados por run_id (locales, áreas, ingresos/egresos) – GLOBAL
    aggs = (
        db.session.query(
            Ejecucion.run_id.label("run_id"),
            func.coalesce(func.sum(cast(Ejecucion.locales_12, Integer)), 0).label("sum_l12"),
            func.coalesce(func.sum(cast(Ejecucion.locales_16, Integer)), 0).label("sum_l16"),
            func.coalesce(func.sum(cast(Ejecucion.locales_20, Integer)), 0).label("sum_l20"),
            func.coalesce(func.sum(cast(Ejecucion.locales_25, Integer)), 0).label("sum_l25"),
            func.coalesce(func.sum(cast(EjecucionDetalle.ar_alimentos_frescos, Integer)), 0).label("sum_ar_af"),
            func.coalesce(func.sum(cast(EjecucionDetalle.ar_comidas_preparadas, Integer)), 0).label("sum_ar_cp"),
            func.coalesce(func.sum(cast(EjecucionDetalle.ar_no_alimentarios, Integer)), 0).label("sum_ar_nal"),
            func.coalesce(func.sum(cast(EjecucionDetalle.ar_complementarios, Integer)), 0).label("sum_ar_sc"),
            func.coalesce(func.sum(EjecucionDetalle.ing_total), 0.0).label("sum_ing_total"),
            func.coalesce(func.sum(EjecucionDetalle.egr_total), 0.0).label("sum_egr_total"),
        )
        .outerjoin(EjecucionDetalle, EjecucionDetalle.ejecucion_id == Ejecucion.id)
        # 🔥 sin filtro por Ejecucion.user_key
        .group_by(Ejecucion.run_id)
        .all()
    )

    aggs_map = {
        a.run_id: {
            "sum_l12": a.sum_l12, "sum_l16": a.sum_l16, "sum_l20": a.sum_l20, "sum_l25": a.sum_l25,
            "sum_ar_af": a.sum_ar_af, "sum_ar_cp": a.sum_ar_cp, "sum_ar_nal": a.sum_ar_nal, "sum_ar_sc": a.sum_ar_sc,
            "sum_ing_total": float(a.sum_ing_total or 0.0),
            "sum_egr_total": float(a.sum_egr_total or 0.0),
        }
        for a in aggs
    }

    rows_with_aggs = []
    for r in filas:
        sums = aggs_map.get(r.run_id, {
            "sum_l12": 0, "sum_l16": 0, "sum_l20": 0, "sum_l25": 0,
            "sum_ar_af": 0, "sum_ar_cp": 0, "sum_ar_nal": 0, "sum_ar_sc": 0,
            "sum_ing_total": 0.0, "sum_egr_total": 0.0,
        })
        rows_with_aggs.append({"resumen": r, **sums})

    return render_template(
        "resumen_corrida.html",
        rows_with_aggs=rows_with_aggs,
        only_run=only_run,
        limit=limit,
    )

@app.route("/resumen-corrida.csv")
def resumen_corrida_csv():
    """Exporta el mismo resumen como CSV (GLOBAL) respetando los filtros."""
    only_run = (request.args.get("run_id") or "").strip()
    limit = int(request.args.get("limit", 50) or 50)

    q = ResumenRun.query
    if only_run:
        q = q.filter(ResumenRun.run_id == only_run)

    filas = (q.order_by(ResumenRun.prom_fitness.desc(),
                        ResumenRun.created_at.desc())
               .limit(limit)
               .all())

    aggs = (
        db.session.query(
            Ejecucion.run_id.label("run_id"),
            func.coalesce(func.sum(cast(Ejecucion.locales_12, Integer)), 0).label("sum_l12"),
            func.coalesce(func.sum(cast(Ejecucion.locales_16, Integer)), 0).label("sum_l16"),
            func.coalesce(func.sum(cast(Ejecucion.locales_20, Integer)), 0).label("sum_l20"),
            func.coalesce(func.sum(cast(Ejecucion.locales_25, Integer)), 0).label("sum_l25"),
            func.coalesce(func.sum(cast(EjecucionDetalle.ar_alimentos_frescos, Integer)), 0).label("sum_ar_af"),
            func.coalesce(func.sum(cast(EjecucionDetalle.ar_comidas_preparadas, Integer)), 0).label("sum_ar_cp"),
            func.coalesce(func.sum(cast(EjecucionDetalle.ar_no_alimentarios, Integer)), 0).label("sum_ar_nal"),
            func.coalesce(func.sum(cast(EjecucionDetalle.ar_complementarios, Integer)), 0).label("sum_ar_sc"),
            func.coalesce(func.sum(EjecucionDetalle.ing_total), 0.0).label("sum_ing_total"),
            func.coalesce(func.sum(EjecucionDetalle.egr_total), 0.0).label("sum_egr_total"),
        )
        .outerjoin(EjecucionDetalle, EjecucionDetalle.ejecucion_id == Ejecucion.id)
        .group_by(Ejecucion.run_id)
        .all()
    )

    aggs_map = {
        a.run_id: {
            "sum_l12": a.sum_l12, "sum_l16": a.sum_l16, "sum_l20": a.sum_l20, "sum_l25": a.sum_l25,
            "sum_ar_af": a.sum_ar_af, "sum_ar_cp": a.sum_ar_cp, "sum_ar_nal": a.sum_ar_nal, "sum_ar_sc": a.sum_ar_sc,
            "sum_ing_total": float(a.sum_ing_total or 0.0),
            "sum_egr_total": float(a.sum_egr_total or 0.0),
        }
        for a in aggs
    }

    # Generar CSV
    output = io.StringIO(newline="")
    writer = csv.writer(output)

    writer.writerow([
        "run_id",
        "prom_fitness", "prom_roi", "prom_margen", "prom_ben_social",
        "total_inversion", "total_utilidad",
        "mejor_comuna_base", "mejor_roi_base",
        "locales_12", "locales_16", "locales_20", "locales_25",
        "ar_alimentos_frescos", "ar_comidas_preparadas", "ar_no_alimentarios", "ar_complementarios",
        "ing_total", "egr_total",
    ])

    for r in filas:
        sums = aggs_map.get(r.run_id, {
            "sum_l12": 0, "sum_l16": 0, "sum_l20": 0, "sum_l25": 0,
            "sum_ar_af": 0, "sum_ar_cp": 0, "sum_ar_nal": 0, "sum_ar_sc": 0,
            "sum_ing_total": 0.0, "sum_egr_total": 0.0,
        })
        writer.writerow([
            r.run_id,
            (r.prom_fitness or 0),
            (r.prom_roi or 0),
            (r.prom_margen or 0),
            (r.prom_ben_social or 0),
            (r.total_inversion or 0),
            (r.total_utilidad or 0),
            (r.mejor_comuna_base if r.mejor_comuna_base is not None else ""),
            (r.mejor_roi_base or 0),
            sums["sum_l12"], sums["sum_l16"], sums["sum_l20"], sums["sum_l25"],
            sums["sum_ar_af"], sums["sum_ar_cp"], sums["sum_ar_nal"], sums["sum_ar_sc"],
            sums["sum_ing_total"], sums["sum_egr_total"],
        ])

    csv_bytes = output.getvalue().encode("utf-8-sig")  # BOM para Excel
    output.close()

    return Response(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="resumen_corrida.csv"'},
    )

@app.route('/como-usar')
def como_usar():
    return render_template('como_usar.html')

@app.route('/legal')
def legal():
    return render_template('legal.html')

# =============================================
# RUTAS ESPECÍFICAS PARA APP MÓVIL
# =============================================

# =============================================================================
# ENDPOINTS PARA APP MÓVIL - AGREGAR ESTO AL FINAL DEL ARCHIVO app.py
# =============================================================================

@app.route('/api/mobile/historial', methods=['GET'])
def api_mobile_historial():
    """Endpoint para obtener el historial de ejecuciones (para app móvil)"""
    try:
        uk = get_or_create_user_key()
        print(f"📊 Obteniendo historial para user_key: {uk}")
        
        # Obtener resumen de ejecuciones del usuario
        corridas = (db.session.query(
                        Ejecucion.run_id,
                        func.count(Ejecucion.id).label('filas'),
                        func.max(Ejecucion.fecha_ejecucion).label('ultima_fecha'),
                        func.max(Ejecucion.roi).label('mejor_roi'),
                        func.max(Ejecucion.mejor_fitness).label('mejor_fitness')
                    )
                    .filter(Ejecucion.user_key == uk)
                    .group_by(Ejecucion.run_id)
                    .order_by(func.max(Ejecucion.fecha_ejecucion).desc())
                    .all())
        
        historial_list = []
        for c in corridas:
            historial_list.append({
                "run_id": c.run_id,
                "filas": c.filas,
                "ultima_fecha": c.ultima_fecha.isoformat() if c.ultima_fecha else None,
                "completa": c.filas >= 7,
                "mejor_roi": float(c.mejor_roi) if c.mejor_roi else 0,
                "mejor_fitness": float(c.mejor_fitness) if c.mejor_fitness else 0
            })
        
        return jsonify({
            "status": "success",
            "data": historial_list
        })
        
    except Exception as e:
        print(f"Error obteniendo historial móvil: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Error interno: {str(e)}"
        }), 500

@app.route('/api/mobile/resultados/<run_id>', methods=['GET'])
def api_mobile_resultados(run_id):
    """Endpoint MEJORADO para obtener resultados específicos (para app móvil)"""
    try:
        uk = get_or_create_user_key()
        print(f"📄 Obteniendo resultados DETALLADOS para user_key: {uk}, run: {run_id}")

        # Obtener ejecuciones para este run_id
        ejecuciones = (Ejecucion.query
                      .filter_by(run_id=run_id, user_key=uk)
                      .order_by(Ejecucion.comuna.asc())
                      .all())
        
        if not ejecuciones:
            return jsonify({
                "status": "error",
                "message": "No se encontraron resultados para este run_id"
            }), 404

        # Obtener parámetros de la corrida (de la primera ejecución)
        primera_ejecucion = ejecuciones[0]
        parametros = {
            "population_size": primera_ejecucion.poblacion_inicial,
            "max_generations": primera_ejecucion.generaciones,
            "elite_percentage": primera_ejecucion.porcentaje_elite,
            "mutation_rate": primera_ejecucion.tasa_mutacion,
            "sigma_factor": primera_ejecucion.fuerza_sigma,
            "crossover_rate": primera_ejecucion.tasa_cruzamiento,
            "weights": (
                primera_ejecucion.peso_be,
                primera_ejecucion.peso_bs,
                primera_ejecucion.peso_mun
            )
        }

        # Mejor comuna base (1..6) por ROI
        mejor_comuna = None
        mejor_roi = 0
        base_ejecuciones = [e for e in ejecuciones if e.comuna <= 6]
        if base_ejecuciones:
            mejor_base = max(base_ejecuciones, key=lambda x: x.roi or 0)
            mejor_comuna = mejor_base.comuna
            mejor_roi = mejor_base.roi or 0

        # Convertir a formato JSON para móvil CON TODOS LOS CAMPOS
        resultados_list = []
        for e in ejecuciones:
            resultados_list.append({
                "comuna": e.comuna,
                "fitness": float(e.mejor_fitness) if e.mejor_fitness else 0,
                "roi": float(e.roi) if e.roi else 0,
                "inversion": float(e.inv_inicial_usd) if e.inv_inicial_usd else 0,
                "utilidad": float(e.utilidad_neta_usd) if e.utilidad_neta_usd else 0,
                "empleos": int(e.empleos_directos) if e.empleos_directos else 0,
                "margen": float(e.margen_utilidad) if e.margen_utilidad else 0,
                "beneficio_social": float(e.beneficio_social) if e.beneficio_social else 0,
                "locales_12": e.locales_12 or 0,
                "locales_16": e.locales_16 or 0,
                "locales_20": e.locales_20 or 0,
                "locales_25": e.locales_25 or 0
            })

        # Calcular métricas totales
        total_inversion = sum(r['inversion'] for r in resultados_list)
        total_utilidad = sum(r['utilidad'] for r in resultados_list)
        promedio_roi = (total_utilidad / total_inversion) * 100 if total_inversion > 0 else 0
        promedio_fitness = sum(r['fitness'] for r in resultados_list) / len(resultados_list)

        # Tabla de locales por tipo (totales) - COMO EN LA WEB
        totales_locales = {
            "l12": sum(r['locales_12'] for r in resultados_list),
            "l16": sum(r['locales_16'] for r in resultados_list),
            "l20": sum(r['locales_20'] for r in resultados_list),
            "l25": sum(r['locales_25'] for r in resultados_list)
        }

        # Obtener resumen run si existe
        resumen_run = ResumenRun.query.filter_by(user_key=uk, run_id=run_id).first()
        
        metricas_totales = {
            "total_inversion": total_inversion,
            "total_utilidad": total_utilidad,
            "promedio_roi": promedio_roi,
            "promedio_fitness": promedio_fitness,
            "promedio_margen": resumen_run.prom_margen if resumen_run else 0,
            "promedio_beneficio_social": resumen_run.prom_ben_social if resumen_run else 0
        }

        return jsonify({
            "status": "success",
            "resultados": resultados_list,
            "metricas_totales": metricas_totales,
            "parametros": parametros,
            "mejor_comuna": mejor_comuna,
            "mejor_roi": mejor_roi,
            "totales_locales": totales_locales,
            "locales_rows": [{
                "comuna": r["comuna"],
                "l12": r["locales_12"],
                "l16": r["locales_16"], 
                "l20": r["locales_20"],
                "l25": r["locales_25"]
            } for r in resultados_list],
            "completa": len(ejecuciones) >= 7
        })
        
    except Exception as e:
        print(f"Error obteniendo resultados móviles: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Error interno: {str(e)}"
        }), 500


##Parametrizacion Original
# # @app.route('/api/mobile/parametrizacion', methods=['POST', 'OPTIONS'])
# # def mobile_ejecutar_parametrizacion():
# #     """Endpoint CORREGIDO para ejecutar optimización desde la app móvil y GUARDAR en BD"""
    
# #     if request.method == 'OPTIONS':
# #         return jsonify({'status':'ok'}), 200  
    
# #     try:
# #         # Obtener user_key para móvil PRIMERO
# #         uk = get_or_create_user_key()
# #         print(f"🚀 Ejecutando parametrización para user_key: {uk}")
        
# #         data = request.get_json()
# #         print(f"📨 Datos recibidos: {data}")

# #         # Validar que hay datos
# #         if not data:
# #             return jsonify({
# #                 "status": "error", 
# #                 "message": "No se recibieron datos"
# #             }), 400

# #         # Extraer parámetros del request
# #         weights = (
# #             float(data.get('peso_be', 0.33)),
# #             float(data.get('peso_bs', 0.33)),
# #             float(data.get('peso_mun', 0.34))
# #         )

# #         # Crear estructura de galerías CORREGIDA
# #         galerias_a_procesar = []
# #         for i in range(1, 8):  # Galerías 1-7
# #             galeria_data = {
# #                 'numero': i,
# #                 'comuna': i,
# #                 'tam_lote': int(data.get(f'tam_lote_{i}', 5000)),
# #                 'can_pri': int(data.get(f'can_pri_{i}', 2)),
# #                 'can_sec': int(data.get(f'can_sec_{i}', 1))
# #             }
# #             galerias_a_procesar.append(galeria_data)
# #             print(f"🏗️  Galería {i}: {galeria_data}")

# #         # Extraer parámetros del algoritmo
# #         population_size = int(data.get('population_size', 50))
# #         max_generations = int(data.get('max_generations', 100))
# #         mutation_rate = float(data.get('mutation_rate', 0.05))
# #         crossover_rate = float(data.get('crossover_rate', 0.7))
# #         elite_percentage = float(data.get('elite_percentage', 0.1))
# #         sigma_factor = float(data.get('sigma_factor', 0.1))

# #         parametros_algoritmo = {
# #             'population_size': population_size,
# #             'max_generations': max_generations,
# #             'mutation_rate': mutation_rate,
# #             'crossover_rate': crossover_rate,
# #             'elite_percentage': elite_percentage,
# #             'sigma_factor': sigma_factor
# #         }

# #         print(f"⚙️  Parámetros algoritmo: {parametros_algoritmo}")
# #         print(f"⚖️  Pesos: {weights}")
        
# #         # Generar run_id único
# #         run_id = str(uuid.uuid4())
# #         thread_id = str(time.time())
        
# #         print(f"📝 IDs generados - Run: {run_id}, Thread: {thread_id}")

# #         # Inicializar logs
# #         nskey = _ns(uk, thread_id)
# #         if 'EXECUTION_LOGS' not in app.config:
# #             app.config['EXECUTION_LOGS'] = {}
# #         app.config['EXECUTION_LOGS'][nskey] = []
        
# #         # ✅ EJECUTAR Y GUARDAR EN BD (SÍNCRONO para móvil)
# #         try:
# #             # OPCIÓN 1: Si la función espera "galerias_a_procesar" en lugar de "galerias"
# #             procesar_todas_galerias(
# #                 app=app,
# #                 thread_id=thread_id,
# #                 galerias_a_procesar=galerias_a_procesar,  # ← NOMBRE CORREGIDO
# #                 population_size=population_size,
# #                 max_generations=max_generations,
# #                 elite_percentage=elite_percentage,
# #                 mutation_rate=mutation_rate,
# #                 sigma_factor=sigma_factor,
# #                 crossover_rate=crossover_rate,
# #                 weights=weights,
# #                 run_id=run_id,
# #                 user_key=uk
# #             )
            
# #             # Verificar que se guardaron datos
# #             ejecuciones_guardadas = Ejecucion.query.filter_by(run_id=run_id, user_key=uk).count()
# #             print(f"💾 Ejecuciones guardadas en BD: {ejecuciones_guardadas}")
            
# #             return jsonify({
# #                 "status": "success",
# #                 "message": "Optimización completada y guardada en BD",
# #                 "run_id": run_id,
# #                 "thread_id": thread_id,
# #                 "ejecuciones_guardadas": ejecuciones_guardadas,
# #                 "user_key": uk
# #             })
            
# #         except Exception as e:
# #             error_msg = f"Error en procesamiento: {str(e)}"
# #             print(f"❌ ERROR en procesar_todas_galerias: {error_msg}")
# #             import traceback
# #             traceback.print_exc()
            
# #             return jsonify({
# #                 "status": "error",
# #                 "message": error_msg
# #             }), 500
            
# #     except Exception as e:
# #         print(f"❌ Error en mobile_ejecutar_parametrizacion: {str(e)}")
# #         import traceback
# #         traceback.print_exc()
        
# #         return jsonify({
# #             "status": "error", 
# #             "message": f"Error interno: {str(e)}"
# #         }), 500

@app.route('/api/mobile/parametrizacion', methods=['POST', 'OPTIONS'])
def mobile_ejecutar_parametrizacion():
    """Endpoint CORREGIDO para ejecutar optimización desde la app móvil y GUARDAR en BD"""
    
    if request.method == 'OPTIONS':
        return jsonify({'status':'ok'}), 200  
    
    try:
        # Obtener user_key para móvil PRIMERO
        uk = get_or_create_user_key()
        print(f"🚀 Ejecutando parametrización para user_key: {uk}")
        
        data = request.get_json()
        print(f"📨 Datos recibidos: {data}")

        # Validar que hay datos
        if not data:
            return jsonify({
                "status": "error", 
                "message": "No se recibieron datos"
            }), 400

        # Extraer parámetros del request
        weights = (
            float(data.get('peso_be', 0.33)),
            float(data.get('peso_bs', 0.33)),
            float(data.get('peso_mun', 0.34))
        )

        # Crear estructura de galerías CORREGIDA
        galerias_a_procesar = []
        for i in range(1, 8):  # Galerías 1-7
            galeria_data = {
                'numero': i,
                'comuna': i,
                'tam_lote': int(data.get(f'tam_lote_{i}', 5000)),
                'can_pri': int(data.get(f'can_pri_{i}', 2)),
                'can_sec': int(data.get(f'can_sec_{i}', 1))
            }
            galerias_a_procesar.append(galeria_data)
            print(f"🏗️  Galería {i}: {galeria_data}")

        # Extraer parámetros del algoritmo
        population_size = int(data.get('population_size', 50))
        max_generations = int(data.get('max_generations', 100))
        mutation_rate = float(data.get('mutation_rate', 0.05))
        crossover_rate = float(data.get('crossover_rate', 0.7))
        elite_percentage = float(data.get('elite_percentage', 0.1))
        sigma_factor = float(data.get('sigma_factor', 0.1))

        parametros_algoritmo = {
            'population_size': population_size,
            'max_generations': max_generations,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'elite_percentage': elite_percentage,
            'sigma_factor': sigma_factor
        }

        print(f"⚙️  Parámetros algoritmo: {parametros_algoritmo}")
        print(f"⚖️  Pesos: {weights}")
        
        # Generar run_id único
        run_id = str(uuid.uuid4())
        thread_id = str(time.time())
        nskey = f"{uk}:{thread_id}"
        
        print(f"📝 IDs generados - Run: {run_id}, Thread: {thread_id}")

        # INICIALIZAR PROGRESO Y LOGS ANTES de iniciar el procesamiento
        if 'EXECUTION_PROGRESS' not in app.config:
            app.config['EXECUTION_PROGRESS'] = {}
        if 'EXECUTION_LOGS' not in app.config:
            app.config['EXECUTION_LOGS'] = {}
        
        # Configurar estado inicial
        app.config['EXECUTION_PROGRESS'][nskey] = {
            'progreso': 0,
            'estado_actual': 'Iniciando procesamiento...',
            'completadas': 0,
            'total': 7,
            'galerias_completadas': 0,
            'galerias_totales': 7
        }
        
        app.config['EXECUTION_LOGS'][nskey] = [
            f"Iniciando optimización - Run ID: {run_id}",
            f"Thread ID: {thread_id}",
            "Preparando datos iniciales..."
        ]
        
        print(f"✅ Progreso inicializado para nskey: {nskey}")
        
        # ✅ EJECUTAR EN SEGUNDO PLANO CON PROGRESO REAL
        def ejecutar_optimizacion_real():
            try:
                # Función para actualizar progreso
                def actualizar_progreso(progreso, estado, log=None):
                    if nskey in app.config['EXECUTION_PROGRESS']:
                        app.config['EXECUTION_PROGRESS'][nskey].update({
                            'progreso': progreso,
                            'estado_actual': estado,
                            'completadas': min(7, int(progreso / 14.28))
                        })
                    
                    if log and nskey in app.config['EXECUTION_LOGS']:
                        app.config['EXECUTION_LOGS'][nskey].append(log)
                        print(f"📝 [{nskey}] {log}")
                    
                    print(f"📊 [{nskey}] Progreso: {progreso}% - Estado: {estado}")
                
                # ✅ EJECUTAR DIRECTAMENTE LA FUNCIÓN REAL DE OPTIMIZACIÓN
                print(f"🎯 Ejecutando función real de optimización...")
                actualizar_progreso(5, 'Iniciando algoritmo genético...', 'Iniciando procesamiento real')
                
                # LLAMAR A LA FUNCIÓN REAL DENTRO DEL CONTEXTO DE LA APLICACIÓN
                with app.app_context():
                    procesar_todas_galerias(
                        app=app,
                        thread_id=thread_id,
                        galerias_a_procesar=galerias_a_procesar,
                        population_size=population_size,
                        max_generations=max_generations,
                        elite_percentage=elite_percentage,
                        mutation_rate=mutation_rate,
                        sigma_factor=sigma_factor,
                        crossover_rate=crossover_rate,
                        weights=weights,
                        run_id=run_id,
                        user_key=uk
                    )
                
                # MARCO COMO 100% COMPLETADO
                actualizar_progreso(100, 'Procesamiento completado', '✅ Optimización finalizada correctamente')
                
                # Verificar que se guardaron datos
                with app.app_context():
                    ejecuciones_guardadas = Ejecucion.query.filter_by(run_id=run_id, user_key=uk).count()
                    print(f"💾 Ejecuciones guardadas en BD: {ejecuciones_guardadas}")
                
                # GUARDAR RESULTADOS
                if 'RESULTS' not in app.config:
                    app.config['RESULTS'] = {}
                
                app.config['RESULTS'][nskey] = {
                    'status': 'success',
                    'run_id': run_id,
                    'thread_id': thread_id,
                    'ejecuciones_guardadas': ejecuciones_guardadas
                }
                
                print(f"✅ Optimización REAL completada - Run: {run_id}")
                
            except Exception as e:
                error_msg = f'Error en procesamiento: {str(e)}'
                print(f"❌ Error en procesamiento: {error_msg}")
                import traceback
                traceback.print_exc()
                
                if nskey in app.config['EXECUTION_LOGS']:
                    app.config['EXECUTION_LOGS'][nskey].append(f'❌ ERROR: {error_msg}')
                if nskey in app.config['EXECUTION_PROGRESS']:
                    app.config['EXECUTION_PROGRESS'][nskey]['progreso'] = 0
                    app.config['EXECUTION_PROGRESS'][nskey]['estado_actual'] = f'Error: {str(e)}'
        
        # INICIAR EN SEGUNDO PLANO
        import threading
        thread = threading.Thread(target=ejecutar_optimizacion_real)
        thread.daemon = True
        thread.start()
        
        # Respuesta inmediata al frontend
        return jsonify({
            "status": "success",
            "message": "Optimización iniciada en segundo plano",
            "run_id": run_id,
            "thread_id": thread_id,
            "ejecuciones_guardadas": 0,  # Aún no se han guardado
            "user_key": uk
        })
            
    except Exception as e:
        print(f"❌ Error en mobile_ejecutar_parametrizacion: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "status": "error", 
            "message": f"Error interno: {str(e)}"
        }), 500

# # @app.route('/api/mobile/estado/<thread_id>', methods=['GET'])
# # def mobile_status(thread_id):
# #     """Verificar estado del procesamiento - MEJORADO"""
# #     try:
# #         print(f"DEBUG mobile_status: Buscando progress para thread_id={thread_id}")
        
# #         # OBTENER USER_KEY PRIMERO
# #         uk = get_or_create_user_key()
# #         nskey = f"{uk}:{thread_id}"
        
# #         print(f"DEBUG mobile_status: user_key={uk}, nskey={nskey}")
        
# #         # INICIALIZAR SI NO EXISTE
# #         if 'EXECUTION_PROGRESS' not in app.config:
# #             app.config['EXECUTION_PROGRESS'] = {}
# #             print("DEBUG: EXECUTION_PROGRESS inicializado")
        
# #         if 'EXECUTION_LOGS' not in app.config:
# #             app.config['EXECUTION_LOGS'] = {}
# #             print("DEBUG: EXECUTION_LOGS inicializado")
        
# #         # BUSCAR DIRECTAMENTE CON LA CLAVE CORRECTA
# #         progress_data = app.config['EXECUTION_PROGRESS'].get(nskey, {})
# #         logs = app.config.get('EXECUTION_LOGS', {}).get(nskey, [])
# #         has_results = 'RESULTS' in app.config and nskey in app.config['RESULTS']
        
# #         # Si no hay datos de progreso, crear estructura básica
# #         if not progress_data:
# #             print(f"DEBUG: No se encontró progress_data para {nskey}, creando estructura básica")
# #             progress_data = {
# #                 'progreso': 0,
# #                 'estado_actual': 'Iniciando procesamiento...',
# #                 'completadas': 0,
# #                 'total': 7,
# #                 'galerias_completadas': 0,
# #                 'galerias_totales': 7
# #             }
# #             app.config['EXECUTION_PROGRESS'][nskey] = progress_data
        
# #         progreso = progress_data.get('progreso', 0)
# #         estado_actual = progress_data.get('estado_actual', 'Procesando...')
# #         galerias_completadas = progress_data.get('completadas', 0)
# #         galerias_totales = progress_data.get('total', 7)
        
# #         print(f"DEBUG mobile_status: progreso={progreso}, estado={estado_actual}")
        
# #         # ✅ CORREGIDO: Solo marcar como completado cuando realmente termine
# #         # No marcar como completado solo por tener resultados o progreso alto
# #         if has_results and progreso >= 95:
# #             status = 'completado'
# #         elif progreso >= 100:  # Solo cuando llegue explícitamente a 100%
# #             status = 'completado'
# #         else:
# #             status = 'procesando'
        
# #         response_data = {
# #             'status': 'success',
# #             'progreso': progreso,
# #             'estado_actual': estado_actual,
# #             'logs_recientes': logs[-10:] if logs else ['Esperando logs...'],
# #             'has_results': has_results,
# #             'galerias_completadas': galerias_completadas,
# #             'galerias_totales': galerias_totales,
# #             'nskey_encontrada': nskey
# #         }
        
# #         print(f"DEBUG mobile_status: Enviando respuesta: {response_data}")
# #         return jsonify(response_data)
        
# #     except Exception as e:
# #         error_msg = f'Error verificando estado: {str(e)}'
# #         print(f"ERROR mobile_status: {error_msg}")
# #         import traceback
# #         traceback.print_exc()
        
# #         return jsonify({
# #             'status': 'error',
# #             'message': error_msg,
# #             'progreso': 0,
# #             'estado_actual': 'Error en servidor',
# #             'logs_recientes': [],
# #             'has_results': False
# #         }), 500

@app.route('/api/mobile/estado/<thread_id>', methods=['GET'])
def mobile_status(thread_id):
    """Verificar estado del procesamiento - SOLO PROGRESO REAL"""
    try:
        uk = get_or_create_user_key()
        nskey = f"{uk}:{thread_id}"
        
        print(f"🔍 Consultando estado para: {nskey}")
        
        # INICIALIZAR SI NO EXISTE
        if 'EXECUTION_PROGRESS' not in app.config:
            app.config['EXECUTION_PROGRESS'] = {}
            print("⚠️  EXECUTION_PROGRESS inicializado")
        
        if 'EXECUTION_LOGS' not in app.config:
            app.config['EXECUTION_LOGS'] = {}
            print("⚠️  EXECUTION_LOGS inicializado")
        
        # OBTENER DATOS ACTUALES
        progress_data = app.config['EXECUTION_PROGRESS'].get(nskey, {})
        logs = app.config.get('EXECUTION_LOGS', {}).get(nskey, [])
        has_results = 'RESULTS' in app.config and nskey in app.config['RESULTS']
        
        # Si no existe, crear estructura básica
        if not progress_data:
            print(f"⚠️  No se encontró progress_data para {nskey}, creando estructura básica")
            progress_data = {
                'progreso': 0,
                'estado_actual': 'Esperando inicio del procesamiento...',
                'completadas': 0,
                'total': 7
            }
            app.config['EXECUTION_PROGRESS'][nskey] = progress_data
        
        progreso = progress_data.get('progreso', 0)
        estado_actual = progress_data.get('estado_actual', 'Procesando...')
        galerias_completadas = progress_data.get('completadas', 0)
        
        print(f"📊 Estado actual - Progreso: {progreso}%, Estado: {estado_actual}, Galerías: {galerias_completadas}/7")
        
        # Determinar estado general - SOLO completado cuando sea 100%
        if progreso >= 100:
            status = 'completado'
        else:
            status = 'procesando'
        
        response_data = {
            'status': 'success',
            'progreso': progreso,
            'estado_actual': estado_actual,
            'logs_recientes': logs[-20:] if logs else ['Esperando inicio del procesamiento...'],
            'has_results': has_results,
            'galerias_completadas': galerias_completadas,
            'galerias_totales': 7,
            'nskey': nskey
        }
        
        print(f"✅ Enviando respuesta - Progreso: {progreso}%")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f'Error verificando estado: {str(e)}'
        print(f"❌ ERROR mobile_status: {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'progreso': 0,
            'estado_actual': 'Error en servidor',
            'logs_recientes': [f'Error: {error_msg}'],
            'has_results': False
        }), 500

@app.route('/api/debug/check-data')
def debug_check_data():
    """Verifica si hay datos en la base de datos"""
    try:
        uk = get_or_create_user_key()
        ejecuciones = Ejecucion.query.filter_by(user_key=uk).all()
        
        return jsonify({
            "total_ejecuciones": len(ejecuciones),
            "ejecuciones": [
                {
                    "id": e.id,
                    "run_id": e.run_id,
                    "comuna": e.comuna,
                    "fecha": e.fecha_ejecucion.isoformat() if e.fecha_ejecucion else None,
                    "locales_12": e.locales_12,
                    "locales_16": e.locales_16,
                    "locales_20": e.locales_20,
                    "locales_25": e.locales_25
                } for e in ejecuciones
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/check-run/<run_id>')
def debug_check_run(run_id):
    """Verificar datos específicos de un run_id"""
    try:
        uk = get_or_create_user_key()
        
        # Verificar ejecuciones
        ejecuciones = Ejecucion.query.filter_by(run_id=run_id, user_key=uk).all()
        
        # Verificar resumen
        resumen = ResumenRun.query.filter_by(run_id=run_id, user_key=uk).first()
        
        return jsonify({
            "run_id": run_id,
            "total_ejecuciones": len(ejecuciones),
            "ejecuciones": [
                {
                    "id": e.id,
                    "comuna": e.comuna,
                    "roi": e.roi,
                    "inversion": e.inv_inicial_usd,
                    "utilidad": e.utilidad_neta_usd
                } for e in ejecuciones
            ],
            "resumen": {
                "existe": resumen is not None,
                "total_inversion": resumen.total_inversion if resumen else None,
                "total_utilidad": resumen.total_utilidad if resumen else None,
                "prom_roi": resumen.prom_roi if resumen else None
            } if resumen else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/routes')
def debug_routes():
    """Mostrar todas las rutas registradas en Flask"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify(routes)

@app.route('/api/mobile/diagnostico', methods=['GET', 'POST', 'OPTIONS'])
def mobile_diagnostico():
    """Endpoint de diagnóstico para probar la conexión móvil"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    uk = get_or_create_user_key()
    
    diagnostico = {
        "status": "success",
        "message": "✅ Backend funcionando correctamente",
        "user_key": uk,
        "metodo": request.method,
        "headers": dict(request.headers),
        "datos_recibidos": request.get_json() if request.method == 'POST' else None,
        "timestamp": time.time()
    }
    
    print(f"🔍 Diagnóstico solicitado - User-Key: {uk}")
    return jsonify(diagnostico)

@app.route("/api/mobile/resumen-corrida", methods=["GET"])
def mobile_resumen_corrida():
    """Resumen de corridas para la aplicación móvil"""
    try:
        # Obtener user_key del header
        user_key = request.headers.get("User-Key")
        if not user_key:
            return jsonify({
                "status": "error", 
                "message": "User-Key header requerido"
            }), 400

        # Filtros opcionales
        only_run = (request.args.get("run_id") or "").strip()
        limit = int(request.args.get("limit", 20) or 20)

        # ResumenRun filtrado por user_key
        q = ResumenRun.query.filter(ResumenRun.user_key == user_key)
        
        if only_run:
            q = q.filter(ResumenRun.run_id == only_run)

        filas = (q.order_by(ResumenRun.prom_fitness.desc(),
                           ResumenRun.created_at.desc())
                  .limit(limit)
                  .all())

        # Agregados por run_id (locales, áreas, ingresos/egresos)
        aggs = (
            db.session.query(
                Ejecucion.run_id.label("run_id"),
                func.coalesce(func.sum(cast(Ejecucion.locales_12, Integer)), 0).label("sum_l12"),
                func.coalesce(func.sum(cast(Ejecucion.locales_16, Integer)), 0).label("sum_l16"),
                func.coalesce(func.sum(cast(Ejecucion.locales_20, Integer)), 0).label("sum_l20"),
                func.coalesce(func.sum(cast(Ejecucion.locales_25, Integer)), 0).label("sum_l25"),
                func.coalesce(func.sum(cast(EjecucionDetalle.ar_alimentos_frescos, Integer)), 0).label("sum_ar_af"),
                func.coalesce(func.sum(cast(EjecucionDetalle.ar_comidas_preparadas, Integer)), 0).label("sum_ar_cp"),
                func.coalesce(func.sum(cast(EjecucionDetalle.ar_no_alimentarios, Integer)), 0).label("sum_ar_nal"),
                func.coalesce(func.sum(cast(EjecucionDetalle.ar_complementarios, Integer)), 0).label("sum_ar_sc"),
                func.coalesce(func.sum(EjecucionDetalle.ing_total), 0.0).label("sum_ing_total"),
                func.coalesce(func.sum(EjecucionDetalle.egr_total), 0.0).label("sum_egr_total"),
            )
            .outerjoin(EjecucionDetalle, EjecucionDetalle.ejecucion_id == Ejecucion.id)
            .filter(Ejecucion.user_key == user_key)
            .group_by(Ejecucion.run_id)
            .all()
        )

        aggs_map = {
            a.run_id: {
                "sum_l12": a.sum_l12, 
                "sum_l16": a.sum_l16, 
                "sum_l20": a.sum_l20, 
                "sum_l25": a.sum_l25,
                "sum_ar_af": a.sum_ar_af, 
                "sum_ar_cp": a.sum_ar_cp, 
                "sum_ar_nal": a.sum_ar_nal, 
                "sum_ar_sc": a.sum_ar_sc,
                "sum_ing_total": float(a.sum_ing_total or 0.0),
                "sum_egr_total": float(a.sum_egr_total or 0.0),
            }
            for a in aggs
        }

        # Preparar datos para móvil
        resumen_data = []
        for r in filas:
            sums = aggs_map.get(r.run_id, {
                "sum_l12": 0, "sum_l16": 0, "sum_l20": 0, "sum_l25": 0,
                "sum_ar_af": 0, "sum_ar_cp": 0, "sum_ar_nal": 0, 
                "sum_ar_sc": 0,
                "sum_ing_total": 0.0, "sum_egr_total": 0.0,
            })
            
            resumen_item = {
                "run_id": r.run_id,
                "prom_fitness": float(r.prom_fitness or 0),
                "prom_roi": float(r.prom_roi or 0),
                "prom_margen": float(r.prom_margen or 0),
                "prom_ben_social": float(r.prom_ben_social or 0),
                "total_inversion": float(r.total_inversion or 0),
                "total_utilidad": float(r.total_utilidad or 0),
                "mejor_comuna_base": r.mejor_comuna_base or "-",
                "mejor_roi_base": float(r.mejor_roi_base or 0),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                # Agregados
                "sum_l12": sums["sum_l12"],
                "sum_l16": sums["sum_l16"],
                "sum_l20": sums["sum_l20"],
                "sum_l25": sums["sum_l25"],
                "sum_ar_af": sums["sum_ar_af"],
                "sum_ar_cp": sums["sum_ar_cp"],
                "sum_ar_nal": sums["sum_ar_nal"],
                "sum_ar_sc": sums["sum_ar_sc"],
                "sum_ing_total": sums["sum_ing_total"],
                "sum_egr_total": sums["sum_egr_total"],
                "total_locales": sums["sum_l12"] + sums["sum_l16"] + sums["sum_l20"] + sums["sum_l25"]
            }
            resumen_data.append(resumen_item)

        return jsonify({
            "status": "success",
            "data": resumen_data,
            "total": len(resumen_data)
        })

    except Exception as e:
        print(f"❌ Error en mobile_resumen_corrida: {str(e)}")
        import traceback
        print(f"🔍 Traceback completo: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": f"Error interno del servidor: {str(e)}"
        }), 500
# =============================================
# FIN RUTAS ESPECÍFICAS PARA APP MÓVIL
# =============================================

if __name__ == '__main__':
    # Inicializar estructuras de datos si no existen
    if 'EXECUTION_LOGS' not in app.config:
        app.config['EXECUTION_LOGS'] = {}
        print("EXECUTION_LOGS inicializado")
        
    if 'RESULTS' not in app.config:
        app.config['RESULTS'] = {}
        print("RESULTS inicializado")

    app.run(host="0.0.0.0",port=5000,debug=True, threaded=True)