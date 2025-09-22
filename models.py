from extensions import db
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

class Ejecucion(db.Model):
    __tablename__ = 'ejecuciones'
    
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(64), index=True, nullable=False)
    # NUEVO: Identificador de comuna (1-6 para existentes, 7 para nueva)
    comuna = db.Column(db.Integer, nullable=False)
    # models.py
    user_key = db.Column(db.String(64), index=True, nullable=False, default="")

    
    # ------------------
    #  Inputs
    # ------------------
    tam_lote_m2 = db.Column(db.Float, nullable=False)
    can_pri_unidades = db.Column(db.Integer, nullable=False)
    can_sec_unidades = db.Column(db.Integer, nullable=False)
    
    # ------------------
    #  Parámetros Algoritmo
    # ------------------
    peso_bs = db.Column(db.Float, nullable=False)
    peso_be = db.Column(db.Float, nullable=False)
    peso_mun = db.Column(db.Float, nullable=False)
    poblacion_inicial = db.Column(db.Integer, nullable=False)
    generaciones = db.Column(db.Integer, nullable=False)
    tasa_mutacion = db.Column(db.Float, nullable=False)
    porcentaje_elite = db.Column(db.Float, nullable=True)   
    fuerza_sigma = db.Column(db.Float, nullable=True)        
    tasa_cruzamiento = db.Column(db.Float, nullable=True) 
    
    # ------------------
    #  Resultados
    # ------------------
    mejor_fitness = db.Column(db.Float, nullable=False)
    inv_inicial_usd = db.Column(db.Float, nullable=False)
    roi = db.Column(db.Float, nullable=False)
    utilidad_neta_usd = db.Column(db.Float, nullable=False)
    margen_utilidad = db.Column(db.Float, nullable=False)
    empleos_directos = db.Column(db.Integer, nullable=False)
    beneficio_social = db.Column(db.Float, nullable=False)
    
    # ------------------
    #  Detalles de Solución
    # ------------------
    cromosoma_optimo = db.Column(JSONB, nullable=False)
    fecha_ejecucion = db.Column(db.DateTime, server_default=func.now())
    created_at = db.Column(db.DateTime, server_default=func.now())

    __table_args__ = (
        db.UniqueConstraint('user_key','run_id', 'comuna', name='uq_ejec_run_comuna'),
    )

    # relación 1:N hacia detalles
    detalles = db.relationship(
        'EjecucionDetalle',
        back_populates='ejecucion',
        cascade='all, delete-orphan',
        lazy='selectin'
    )

class ResumenRun(db.Model):
    __tablename__ = 'resumen_runs'

    id = db.Column(db.Integer, primary_key=True)
    user_key = db.Column(db.String(64), index=True, nullable=False)
    run_id   = db.Column(db.String(64), index=True, nullable=False)

    # agregados
    total_inversion   = db.Column(db.Float, nullable=False, default=0.0)
    total_utilidad    = db.Column(db.Float, nullable=False, default=0.0)

    prom_roi          = db.Column(db.Float, nullable=False, default=0.0)
    prom_margen       = db.Column(db.Float, nullable=False, default=0.0)
    prom_fitness      = db.Column(db.Float, nullable=False, default=0.0)
    prom_ben_social   = db.Column(db.Float, nullable=False, default=0.0)

    # extras útiles para la portada
    mejor_comuna_base = db.Column(db.Integer, nullable=True)   # mejor entre 1..6
    mejor_roi_base    = db.Column(db.Float, nullable=True)
    created_at        = db.Column(db.DateTime, server_default=func.now())

    __table_args__ = (
        db.UniqueConstraint('user_key', 'run_id', name='uq_resumen_user_run'),
    )


class EjecucionDetalle(db.Model):
    __tablename__ = 'ejecucion_detalle'

    id = db.Column(db.Integer, primary_key=True)
    # >>> CLAVE FORÁNEA OBLIGATORIA <<<
    ejecucion_id = db.Column(
        db.Integer,
        db.ForeignKey('ejecuciones.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        unique=True
    )

    # -------- Beneficio Económico: Inversión inicial (costos)
    inv_total = db.Column(db.Float, nullable=False, default=0.0)            # metrics['i_InvIni']
    inv_loc = db.Column(db.Float, nullable=False, default=0.0)              # metrics['l_CTCLo']
    inv_parq = db.Column(db.Float, nullable=False, default=0.0)             # metrics['p_SCTPar']
    inv_zonas = db.Column(db.Float, nullable=False, default=0.0)            # metrics['z_CTZcAv']

    # Ingresos (costos/valores anuales)
    ing_total = db.Column(db.Float, nullable=False, default=0.0)            # metrics['u_IngGal']
    ing_arr = db.Column(db.Float, nullable=False, default=0.0)              # metrics['a_ToArGa']
    ing_adm = db.Column(db.Float, nullable=False, default=0.0)              # metrics['d_ToAdGa']
    ing_parq = db.Column(db.Float, nullable=False, default=0.0)             # metrics['q_ToPaGa']

    # Egresos
    egr_total = db.Column(db.Float, nullable=False, default=0.0)            # metrics['u_EgrGal']
    egr_mant = db.Column(db.Float, nullable=False, default=0.0)             # metrics['m_ToEgGa']
    egr_servpub = db.Column(db.Float, nullable=False, default=0.0)          # metrics['s_ToSPGa']
    egr_salarios = db.Column(db.Float, nullable=False, default=0.0)         # metrics['o_ToSaGa']
    egr_operativos = db.Column(db.Float, nullable=False, default=0.0)       # metrics['v_ToSOGa']
    egr_admin = db.Column(db.Float, nullable=False, default=0.0)            # metrics['n_ToGAGa']
    egr_legales = db.Column(db.Float, nullable=False, default=0.0)          # metrics['t_ToRMGa']
    egr_impuestos = db.Column(db.Float, nullable=False, default=0.0)        # metrics['u_ImpGas']

    # -------- Beneficio Social
    bs_accesibilidad = db.Column(db.Float, nullable=False, default=0.0)     # si tienes índice; si no, deja None
    bs_emp_dir = db.Column(db.Integer, nullable=False, default=0)         # metrics['w_STEmDi']
    bs_emp_ind = db.Column(db.Integer, nullable=False, default=0)         # metrics['x_STEmIn']
    bs_calidad_vida = db.Column(db.Float, nullable=False, default=0.0)      # si tienes índice; si no, None

    # -------- Áreas Comerciales (cantidades)
    ar_alimentos_frescos = db.Column(db.Integer, nullable=False, default=0)   # metrics['y_CLoAlF']
    ar_comidas_preparadas = db.Column(db.Integer, nullable=False, default=0)  # metrics['y_CLoCoP']
    ar_no_alimentarios = db.Column(db.Integer, nullable=False, default=0)     # metrics['y_CLoNAl']
    ar_complementarios = db.Column(db.Integer, nullable=False, default=0)     # metrics['y_CLoSeC']

    ejecucion = db.relationship('Ejecucion', back_populates='detalles')

    def __repr__(self):
        return f"<EjecucionDetalle id={self.id} ejecucion_id={self.ejecucion_id}>"