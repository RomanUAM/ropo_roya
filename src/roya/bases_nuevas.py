"""Cascada portátil: aprende de archivos nuevos sin consultar el catálogo histórico.

Las variables actuales producen diagnóstico en la fecha de muestreo. El perfil
temporal usa exclusivamente observaciones anteriores del mismo sitio.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import re
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

from roya.modelos.genetica_ordinal import GeneticOrdinal3
from roya.modelos.bayes_poisson import BayesianPoissonMLP
from roya.niveles_bayes import InverseBayes

FAMILIES = ('RF', 'LR', 'GP', 'MIXTA')
VIEWS = ('topografia', 'excel', 'simulacion')
REGRESSORS = ('lineal_multiple', 'log1p_lineal', 'polinomio_2', 'polinomio_3',
              'armonica', 'armonica_polinomio_2', 'red_neuronal',
              'bayes_3x13', 'bayes_4x13', 'bayes_5x13')
TARGET = 'Incidencia de roya del Café'
KEYS = ('Municipio', 'Parcela', 'Coordenada_X', 'Coordenada_Y')
CLIMATE = ('Temperatura promedio', 'Punto_de_rocio', 'Temperatura',
           'Humedad_especifica', 'Humedad_relativa', 'Precipitacion')
STAGES = ('Brotacion', 'Floracion', 'Amarre', 'Lechoso', 'Consistente', 'Maduro')


def normalized(value):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower())


def canonical(frame):
    known = [TARGET, *KEYS, *CLIMATE, *STAGES, 'Mes_registrado', 'Quincena', 'Fecha_muestreo_exacta']
    lookup = {normalized(c): c for c in known}
    lookup['mes'] = 'Mes_registrado'
    names = [lookup.get(normalized(c), str(c).strip()) for c in frame]
    if len(set(names)) != len(names):
        raise ValueError('Hay nombres de columnas duplicados después de reconocer acentos y espacios.')
    out = frame.copy().reset_index(drop=True)
    out.columns = names
    return out


def attach_features(frame, features, keys=KEYS):
    """Une una tabla estática por sitio o una tabla dinámica por sitio y fecha."""
    frame, features = canonical(frame), canonical(features)
    if TARGET in features:
        raise ValueError('La tabla complementaria no debe incluir la incidencia observada.')
    missing = [c for c in keys if c not in frame or c not in features]
    if missing:
        raise ValueError('Faltan identificadores para unir la tabla complementaria: ' + ', '.join(missing))
    on = list(keys)
    dynamic = 'Fecha_muestreo_exacta' in features or ('Mes_registrado' in features and 'Quincena' in features)
    if dynamic:
        frame['_fecha_union'], features['_fecha_union'] = dates(frame), dates(features)
        on.append('_fecha_union')
    extra = [c for c in features if c not in on + ['Mes_registrado','Quincena','Fecha_muestreo_exacta']]
    overlaps = [c for c in extra if c in frame]
    if overlaps:
        raise ValueError('La tabla complementaria repite variables de la base: ' + ', '.join(overlaps))
    if features.duplicated(on).any():
        raise ValueError('La tabla complementaria tiene claves repetidas; se requiere una sola fila por sitio/fecha.')
    for f in (frame, features):
        for c in keys:
            if c.startswith('Coordenada_'):
                f[c] = pd.to_numeric(f[c], errors='raise')
            else:
                f[c] = f[c].astype(str).str.strip().str.replace(r'^(\d+)\.0$',r'\1',regex=True)
    out = frame.merge(features[on+extra], on=on, how='left', validate='many_to_one', sort=False, indicator=True)
    if out['_merge'].ne('both').any():
        raise ValueError('Hay sitios o fechas sin información en la tabla complementaria.')
    return out.drop(columns=['_merge','_fecha_union'], errors='ignore')


def complete_inputs(frame, contract, policies=None, simulation=None, fields=None):
    """Completa sólo predictores, con reglas explícitas y procedencia por fila.

    policies[col] = {'mode': 'value'|'zero'|'simulation', 'value': ..., 'source': ...}.
    Sin regla no se sustituye el valor. No completa identidad, fecha ni respuesta.
    """
    frame = canonical(frame)
    policies = policies or {}
    identity_missing=[c for c in contract.keys if c not in frame or frame[c].isna().any()]
    if identity_missing:
        raise ValueError('La identidad del sitio no se puede asumir ni completar con cero: '+', '.join(identity_missing))
    fields = sum((list(contract.views[v]) for v in VIEWS), []) if fields is None else list(fields)
    records = {name: [[] for _ in range(len(frame))] for name in ('cero','simulacion','usuario')}
    source = canonical(simulation) if simulation is not None else None
    source_index = None
    if source is not None:
        source['_fecha_completar'] = dates(source)
        source_index = lookup_keys(source, contract.keys, source['_fecha_completar'])
        if source_index.has_duplicates:
            raise ValueError('La simulación contiene más de un valor por sitio y fecha.')
    for col in fields:
        if col not in frame:
            frame[col] = np.nan
        if col in contract.categorical:
            frame[col] = frame[col].astype(object)
        missing = frame[col].isna() | frame[col].astype(str).str.strip().eq('')
        if not missing.any():
            continue
        policy = policies.get(col)
        if policy is None:
            raise ValueError('Se necesita aportar '+col+', elegir una simulación disponible o autorizar cero.')
        mode = policy.get('mode')
        if mode in ('zero','value'):
            value = 0 if mode=='zero' else policy.get('value')
            if value is None or str(value).strip()=='':
                raise ValueError('Falta el valor solicitado al usuario para '+col)
            frame.loc[missing, col] = value
            category = 'cero' if mode=='zero' else 'usuario'
            for i in np.flatnonzero(missing):records[category][i].append(col)
        elif mode=='simulation':
            field = policy.get('source', col)
            if normalized(field) in {normalized(contract.target), normalized(TARGET)}:
                raise ValueError('La incidencia observada no es una variable simulada ni un predictor.')
            if source is None or field not in source:
                raise ValueError('No hay simulación disponible para '+col)
            query = lookup_keys(frame, contract.keys, dates(frame))
            indices = source_index.get_indexer(query)
            selected = np.flatnonzero(missing)
            if (indices[selected] < 0).any():
                raise ValueError('No hay simulación para todos los sitios/fechas de '+col+'. Aporte esos valores o elija explícitamente cero.')
            values = source.iloc[indices[selected]][field].to_numpy()
            if pd.isna(values).any():
                raise ValueError('La simulación de '+col+' también tiene valores faltantes.')
            frame.loc[selected, col] = values
            for i in selected:records['simulacion'][i].append(col+' ← '+field)
        else:
            raise ValueError('Regla de sustitución no reconocida para '+col)
    provenance = pd.DataFrame({name:['; '.join(row) for row in rows] for name,rows in records.items()})
    return frame, provenance


def lookup_keys(frame, keys, when):
    f=frame[list(keys)].copy()
    if f.isna().any().any():
        raise ValueError('No se puede buscar simulación sin identidad completa del sitio.')
    for c in keys:
        if c.startswith('Coordenada_'):
            f[c]=pd.to_numeric(f[c],errors='raise')
        else:
            f[c]=f[c].astype(str).str.strip().str.replace(r'^(\d+)\.0$',r'\1',regex=True)
    f['_fecha']=np.asarray(when)
    return pd.MultiIndex.from_frame(f)


@dataclass
class Contract:
    views: dict
    numeric: list
    categorical: list
    keys: tuple = KEYS
    target: str = TARGET
    temporal: bool = False
    representation: str = 'ACP'
    max_level: float = 40.

    def validate(self):
        columns = sum((list(self.views.get(v, [])) for v in VIEWS), [])
        if any(not self.views.get(v) for v in VIEWS):
            raise ValueError('Se necesitan variables propias en las tres vistas: topografía, Excel y simulación.')
        if len(columns) != len(set(columns)):
            raise ValueError('Asigne cada variable a una sola vista; el calendario se incorpora automáticamente.')
        forbidden = {normalized(c) for c in [self.target, 'fila_excel', 'fila_archivo', 'Planta',
                                            '_fecha', 'Mes_registrado', 'Quincena', 'Fecha_muestreo_exacta',
                                            'mes_sen','mes_cos','dia_sen','dia_cos','roya_t_1','roya_t_2',*self.keys[:2]]}
        forbidden.update(normalized(c) for c in self.keys if c not in ('Coordenada_X', 'Coordenada_Y'))
        if any(normalized(c) in forbidden for c in columns):
            raise ValueError('La incidencia actual y los identificadores no pueden ser predictores.')
        if set(self.numeric) & set(self.categorical) or set(columns) != set(self.numeric + self.categorical):
            raise ValueError('Declare una sola vez el tipo numérico o categórico de cada predictor.')
        if self.representation not in ('ACP', 'completas') or self.max_level <= 0:
            raise ValueError('Representación o máximo de incidencia inválidos.')


def dates(frame):
    if 'Fecha_muestreo_exacta' in frame:
        exact = pd.to_datetime(frame.Fecha_muestreo_exacta, errors='raise')
    else:
        exact = pd.Series(pd.NaT, index=frame.index)
    if 'Mes_registrado' not in frame or 'Quincena' not in frame:
        if exact.isna().any():
            raise ValueError('Se requiere Fecha_muestreo_exacta o Mes_registrado con año y Quincena 1/2.')
        return exact
    text = frame.Mes_registrado.astype(str)
    if not text.str.match(r'^\d{4}-\d{2}($|[- T])').all():
        raise ValueError('Mes_registrado requiere año y mes ISO, por ejemplo 2026-10.')
    month = pd.to_datetime(text, format='ISO8601', errors='raise')
    quin = pd.to_numeric(frame.Quincena.astype(str).str.replace(r'^Quincena\s*', '', regex=True), errors='raise')
    if not quin.isin([1, 2]).all():
        raise ValueError('Quincena debe ser 1 o 2.')
    result = month.dt.to_period('M').dt.to_timestamp() + pd.to_timedelta(np.where(quin.eq(1), 0, 15), unit='D')
    if ((exact.notna()) & ((exact.dt.month != month.dt.month) | (exact.dt.year != month.dt.year))).any():
        raise ValueError('La fecha exacta contradice el mes registrado.')
    return exact.fillna(result)


def prepare(frame, contract, training=False, history=None):
    contract.validate()
    frame = canonical(frame)
    selected = sum((list(contract.views[v]) for v in VIEWS), [])
    required = list(dict.fromkeys([*contract.keys, *selected]))
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError('Faltan columnas: ' + ', '.join(missing))
    if frame.empty or frame[list(contract.keys)].isna().any().any():
        raise ValueError('La tabla no puede estar vacía y la identidad del sitio debe estar completa.')
    for col in contract.keys:
        if col.startswith('Coordenada_'):
            frame[col] = pd.to_numeric(frame[col], errors='raise')
            if not np.isfinite(frame[col]).all():
                raise ValueError('Coordenadas no finitas: ' + col)
        else:
            frame[col] = frame[col].astype(str).str.strip().str.replace(r'^(\d+)\.0$',r'\1',regex=True)
            if frame[col].eq('').any():
                raise ValueError('Identidad de sitio vacía: ' + col)
    frame['_fecha'] = dates(frame)
    if frame['_fecha'].isna().any() or frame.duplicated([*contract.keys, '_fecha']).any():
        raise ValueError('Fechas vacías o sitio/fecha duplicado. Si hay plantas distintas, incorpore Planta a la identidad, nunca a los predictores.')
    for col in contract.numeric:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if np.isinf(frame[col]).any():
            raise ValueError('Valor infinito: ' + col)
    for col in STAGES:
        if col in frame and not pd.to_numeric(frame[col], errors='raise').dropna().isin([0, 1]).all():
            raise ValueError(col + ' debe ser binaria 0/1; varias etapas pueden coexistir.')
    for col in contract.categorical:
        frame[col] = frame[col].fillna('Sin_dato').astype(str)
    y = None
    if training:
        if contract.target not in frame:
            raise ValueError('Para entrenar se necesita incidencia observada: ' + contract.target)
        y = pd.to_numeric(frame[contract.target], errors='raise').to_numpy(float)
        if not np.isfinite(y).all() or (y < 0).any() or (y > contract.max_level).any():
            raise ValueError('Incidencia observada vacía o fuera del rango declarado.')
    x = frame[selected].copy()
    eligible = np.ones(len(frame), bool)
    if contract.temporal:
        # No se usa la etiqueta de la solicitud, ni la predicción de otra fila,
        # como antecedente. La historia externa contiene sólo mediciones reales.
        hist = frame if training else canonical(history) if history is not None else None
        if hist is None:
            raise ValueError('El perfil temporal requiere historial medido de los mismos sitios para t-1 y t-2.')
        if not training:
            hist, _, _, _ = prepare(hist, contract, training=True)
        hist = hist.sort_values([*contract.keys, '_fecha'])
        histgroups = {key if isinstance(key, tuple) else (key,): g for key, g in hist.groupby(list(contract.keys), sort=False)}
        for row in range(len(frame)):
            key = tuple(frame.loc[row, list(contract.keys)])
            g = histgroups.get(key)
            prev = g[g._fecha.lt(frame.loc[row, '_fecha'])].tail(2) if g is not None else None
            if prev is None or len(prev) < 2 or prev[contract.target].isna().any():
                eligible[row] = False
                x.loc[row, selected] = np.nan
                continue
            for col in selected:
                if col not in contract.views['topografia']:
                    x.loc[row, col] = prev.iloc[-1][col]
            x.loc[row, 'roya_t_1'] = float(prev.iloc[-1][contract.target])
            x.loc[row, 'roya_t_2'] = float(prev.iloc[-2][contract.target])
    angle = 2 * np.pi * (frame._fecha.dt.month - 1) / 12
    day = 2 * np.pi * (frame._fecha.dt.day - 1) / frame._fecha.dt.days_in_month
    for c, values in [('mes_sen', np.sin(angle)), ('mes_cos', np.cos(angle)), ('dia_sen', np.sin(day)), ('dia_cos', np.cos(day))]:
        x[c] = values.to_numpy()
    # Orden estable del archivo; el orden temporal se decide aparte para entrenar.
    return frame, x, y, eligible


class Inputs:
    def fit(self, frame, numeric, categorical, representation='ACP'):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        if self.numeric and frame[self.numeric].isna().all().any():
            raise ValueError('Una variable numérica carece completamente de valores en entrenamiento.')
        self.imputer = SimpleImputer(strategy='median').fit(frame[self.numeric]) if self.numeric else None
        self.scaler = StandardScaler().fit(self.imputer.transform(frame[self.numeric])) if self.numeric else None
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(frame[self.categorical]) if self.categorical else None
        z = self.raw(frame)
        self.pca = None
        if representation == 'ACP' and z.shape[1] > 1 and len(z) > 1 and np.var(z, axis=0).sum() > 1e-12:
            # Máximo 10 componentes para controlar el tamaño del grado tres.
            self.pca = PCA(n_components=min(10, z.shape[1], len(z)), svd_solver='full').fit(z)
        return self

    def raw(self, frame):
        parts = []
        if self.numeric:
            parts.append(self.scaler.transform(self.imputer.transform(frame[self.numeric])))
        if self.categorical:
            parts.append(self.encoder.transform(frame[self.categorical]))
        return np.column_stack(parts)

    def transform(self, frame):
        z = self.raw(frame)
        return self.pca.transform(z) if self.pca is not None else z


def expert(family, config, seed):
    if family == 'RF':
        return RandomForestClassifier(n_estimators=config['trees'], min_samples_leaf=3, class_weight='balanced_subsample', random_state=seed, n_jobs=2)
    if family == 'LR':
        return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
    return GeneticOrdinal3(config['gp_population'], config['gp_generations'], seed)


def view_input(x, contract, view):
    cols = list(contract.views[view]) + [c for c in ('roya_t_1', 'roya_t_2', 'mes_sen', 'mes_cos', 'dia_sen', 'dia_cos') if c in x]
    return cols, [c for c in cols if c not in contract.categorical], [c for c in cols if c in contract.categorical]


def fit_experts(x, y, contract, config, seed):
    if len(np.unique(y)) < 2:
        raise ValueError('Cada bloque de entrenamiento del detector necesita presencias y ausencias reales.')
    models = {}
    for family in FAMILIES[:3]:
        for view in VIEWS:
            cols, nums, cats = view_input(x, contract, view)
            pre = Inputs().fit(x[cols], nums, cats, contract.representation)
            model = expert(family, config, seed).fit(pre.transform(x[cols]), y)
            models[(family, view)] = (pre, model)
    return models


def probabilities(models, x):
    return {family: np.stack([models[(family, view)][1].predict_proba(models[(family, view)][0].transform(x))[:, 1] for view in VIEWS], axis=1) for family in FAMILIES[:3]}


def mixture(probs, entries):
    return np.column_stack([probs[fam][:, j] for fam, j in entries])


def probability_tensor(p):
    return np.stack([1-p, p], axis=-1)


def context(frame):
    return pd.DataFrame({'mes': frame._fecha.dt.month.astype(str).to_numpy()})


def binary_scores(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), Brier=float(brier_score_loss(y, p)),
                sensibilidad=float(tp/(tp+fn)) if tp+fn else None,
                especificidad=float(tn/(tn+fp)) if tn+fp else None)


def scores(y, p):
    if not len(y):
        return dict(n=0, MAE=None, RMSE=None, R2=None)
    return dict(n=len(y), MAE=float(mean_absolute_error(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=float(r2_score(y, p)) if len(y)>1 and np.var(y)>0 else None)


class Quantity:
    def fit(self, x, y, name, config):
        self.name = name
        self.poly = PolynomialFeatures(3 if name == 'polinomio_3' else 2, include_bias=False) if name in ('polinomio_2', 'polinomio_3', 'armonica_polinomio_2') else None
        if self.poly is not None and self.poly.fit(x).n_output_features_ > 10000:
            raise ValueError('Grado polinómico demasiado grande; use ACP o reduzca predictores.')
        z = self.poly.transform(x) if self.poly is not None else x
        self.scale = StandardScaler().fit(z)
        z = self.scale.transform(z)
        if name.startswith('bayes_'):
            self.model = BayesianPoissonMLP(layers=int(name[6]), width=13, epochs=config['neural_epochs'], seed=config['seed']).fit(z, y)
        elif name == 'red_neuronal':
            self.model = MLPRegressor(hidden_layer_sizes=(20,30,30,40), max_iter=config['neural_epochs'], random_state=config['seed'], early_stopping=False).fit(z, np.log1p(y))
        else:
            self.model = Ridge(alpha=1., solver='lsqr').fit(z, np.log1p(y) if name == 'log1p_lineal' else y)
        return self

    def predict(self, x):
        z = self.scale.transform(self.poly.transform(x) if self.poly is not None else x)
        if self.name.startswith('bayes_'):
            return self.model.posterior(z, samples=50)['media']
        p = self.model.predict(z)
        return np.expm1(np.clip(p, -20, 20)) if self.name in ('log1p_lineal', 'red_neuronal') else p


def train(frame, contract, config=None, policies=None, simulation=None):
    """Devuelve un paquete candidato sin sustituir modelos históricos ni usar test para elegir."""
    config = dict({'trees':100, 'gp_population':200, 'gp_generations':30, 'neural_epochs':100, 'seed':42}, **(config or {}))
    contract.validate()
    frame, provenance = complete_inputs(frame, contract, policies, simulation)
    frame, x, y, eligible = prepare(frame, contract, training=True)
    input_hash=hashlib.sha256(pd.util.hash_pandas_object(frame.astype(str),index=False).values.tobytes()).hexdigest()
    input_count=len(frame)
    reference=frame[list(contract.keys)].assign(Fecha_muestreo_exacta=frame._fecha).copy()
    for col in contract.views['simulacion']:
        reference[col]=frame[col].to_numpy()
        # Un cero autorizado o un valor manual no se convierte en simulación real.
        assumed=provenance[['cero','usuario']].apply(lambda r: col in r['cero'].split('; ') or col in r['usuario'].split('; '),axis=1)
        reference.loc[assumed,col]=np.nan
    frame, x, y = frame.loc[eligible].reset_index(drop=True), x.loc[eligible].reset_index(drop=True), y[eligible]
    times = np.sort(frame._fecha.unique())
    if len(times) < 10:
        raise ValueError('Se requieren al menos 10 fechas elegibles para separar 60/20/20 y producir probabilidades causales de entrenamiento.')
    a, b = int(.6*len(times)), int(.8*len(times))
    tr, va, te = frame._fecha.lt(times[a]).to_numpy(), frame._fecha.between(times[a], times[b], inclusive='left').to_numpy(), frame._fecha.ge(times[b]).to_numpy()
    binary = (y > 0).astype(int)
    if (y[tr] > 0).sum() < 10 or len(np.unique(binary[tr])) < 2:
        raise ValueError('Entrenamiento requiere ambas clases y al menos 10 observaciones positivas.')
    # Expansión temporal: cada fecha OOF sólo ve fechas anteriores, incluyendo preprocesado.
    oof = {fam: np.full((len(frame), 3), np.nan) for fam in FAMILIES[:3]}
    for start in range(2, a, 2):
        fit = frame._fecha.lt(times[start]).to_numpy()
        hold = frame._fecha.isin(times[start:min(start+2, a)]).to_numpy()
        if len(np.unique(binary[fit])) < 2:
            continue
        models = fit_experts(x.loc[fit], binary[fit], contract, config, config['seed']+start)
        pred = probabilities(models, x.loc[hold])
        for fam in pred:
            oof[fam][hold] = pred[fam]
    mask = tr & np.isfinite(oof['RF']).all(axis=1)
    if len(np.unique(binary[mask])) < 2 or (y[mask] > 0).sum() < 10:
        raise ValueError('La parte OOF causal requiere ambas clases y al menos 10 positivos. Amplíe el historial inicial.')
    entries = []
    for fam in FAMILIES[:3]:
        loss = [np.mean((oof[fam][mask, j]-binary[mask])**2) for j in range(3)]
        entries.append((fam, int(np.argmin(loss))))
    oof['MIXTA'] = mixture(oof, entries)
    models = fit_experts(x.loc[tr], binary[tr], contract, config, config['seed'])
    prob = probabilities(models, x)
    prob['MIXTA'] = mixture(prob, entries)
    detectors, binary_rows = {}, []
    for fam in FAMILIES:
        det = InverseBayes().fit(probability_tensor(oof[fam][mask]), binary[mask], context(frame.loc[mask]))
        p = det.predict_proba(probability_tensor(prob[fam]), context(frame))[:, 1]
        threshold = .5
        if len(np.unique(binary[va])) == 2:
            candidates = np.linspace(.025, .975, 39)
            threshold = float(min(candidates, key=lambda t: np.mean(p[va & (binary==0)] >= t)+np.mean(p[va & (binary==1)] < t)))
        detectors[fam] = dict(model=det, threshold=threshold)
        for part, ix in [('validacion', va), ('prueba', te)]:
            binary_rows.append(dict(familia=fam, conjunto=part, umbral=threshold, **binary_scores(binary[ix], p[ix], threshold)))
    # Estimador de cantidad común e independiente del detector elegido.
    # Usa covariables y nueve probabilidades expertas OOF; nunca la incidencia actual.
    extras = ['roya_t_1','roya_t_2'] if contract.temporal else []
    nums = contract.numeric + extras + ['mes_sen','mes_cos','dia_sen','dia_cos']
    positive = mask & (y > 0)
    quantity_input = Inputs().fit(x.loc[positive], nums, contract.categorical, contract.representation)
    z = quantity_input.transform(x)
    probabilities_all = np.column_stack([prob[f] for f in FAMILIES[:3]])
    z_all = np.column_stack([z, probabilities_all])
    z_train = np.column_stack([z[positive], *[oof[f][positive] for f in FAMILIES[:3]]])
    regressors, quantity_rows, cascade_rows = {}, [], []
    evaluation=frame.loc[va|te,['_fecha']].reset_index(drop=True).rename(columns={'_fecha':'fecha'})
    evaluation['conjunto']=np.where(va[va|te],'validacion','prueba')
    evaluation['real']=y[va|te]
    for fam in FAMILIES:
        evaluation['probabilidad_'+fam]=detectors[fam]['model'].predict_proba(probability_tensor(prob[fam][va|te]),context(frame.loc[va|te]))[:,1]
    notices = []
    for name in REGRESSORS:
        # Armónicas incluyen dos órdenes de Fourier de mes y día explícitos.
        harmonic = name in ('armonica','armonica_polinomio_2')
        train_z = np.column_stack([z_train, harmonics(x.loc[positive])]) if harmonic else z_train
        all_z = np.column_stack([z_all, harmonics(x)]) if harmonic else z_all
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reg = Quantity().fit(train_z, y[positive], name, config)
        notices.extend(dict(modelo=name, aviso=str(w.message)) for w in caught)
        regressors[name] = reg
        estimate = np.clip(reg.predict(all_z), 0, contract.max_level)
        evaluation['nivel_'+name]=estimate[va|te]
        for part, ix in [('validacion', va), ('prueba', te)]:
            quantity_rows.append(dict(modelo=name, conjunto=part, **scores(y[ix & (y>0)], estimate[ix & (y>0)])))
            for fam in FAMILIES:
                det = detectors[fam]
                p = det['model'].predict_proba(probability_tensor(prob[fam][ix]), context(frame.loc[ix]))[:, 1]
                cascade_rows.append(dict(detector=fam, modelo=name, conjunto=part, **scores(y[ix], np.where(p>=det['threshold'], estimate[ix], 0))))
    val = [r for r in binary_rows if r['conjunto']=='validacion' and r['sensibilidad'] is not None and r['especificidad'] is not None]
    selected_detector = min(val, key=lambda r: 2-r['sensibilidad']-r['especificidad'])['familia'] if val else None
    valq = [r for r in quantity_rows if r['conjunto']=='validacion' and r['MAE'] is not None]
    selected_quantity = min(valq, key=lambda r: r['MAE'])['modelo'] if valq else None
    ranges = {c: [float(x.loc[tr,c].min()), float(x.loc[tr,c].max())] for c in contract.numeric if x.loc[tr,c].notna().any()}
    audit = dict(formato=1, contrato=asdict(contract), config=config, filas=len(frame), positivos_cantidad=int(positive.sum()),
                 negativos_cantidad=0, particion=[dict(conjunto=n, filas=int(ix.sum()), fechas=int(frame.loc[ix,'_fecha'].nunique()), positivos=int(binary[ix].sum()), desde=str(frame.loc[ix,'_fecha'].min()), hasta=str(frame.loc[ix,'_fecha'].max())) for n, ix in [('train',tr),('validacion',va),('prueba',te)]],
                 protocolo='60/20/20 por fechas completas. OOF expansivo en train. Preprocesado sólo train. Cantidad sólo positivos OOF. Umbrales y selección sólo validación; test no elige. Sin aumento automático.',
                 horizonte='Un paso con antecedentes reales' if contract.temporal else 'Diagnóstico en fecha con covariables contemporáneas; no acredita pronóstico anticipado',
                 probabilidades='Posteriores aproximados; no se afirma calibración. Mezcla por menor Brier OOF train.',
                 seleccion_detector=selected_detector, seleccion_cantidad=selected_quantity, advertencias=notices,
                 promocion_operativa=False, version_sklearn=__import__('sklearn').__version__,
                 huella_datos=input_hash, filas_entrada=input_count)
    audit['sustituciones']={c:dict(filas=int(provenance[c].ne('').sum()),variables=sorted(set('; '.join(provenance[c]).split('; '))-{''})) for c in provenance}
    audit['reglas_entrenamiento']=policies or {}
    audit['predictores_constantes']=[c for c in x if x.loc[tr,c].nunique(dropna=True)<=1]
    return dict(kind='roya_portable_v1', contract=contract, models=models, mixture=entries, detectors=detectors,
                quantity_input=quantity_input, regressors=regressors, audit=audit, ranges=ranges,
                sites=set(map(tuple, frame.loc[tr,list(contract.keys)].to_numpy())),
                matrices=binary_rows, quantity_metrics=quantity_rows, cascade_metrics=cascade_rows, simulation_reference=reference,
                evaluation=evaluation)


def predict(bundle, frame, detector=None, regressor=None, history=None, policies=None, simulation=None, history_policies=None):
    if bundle.get('kind') != 'roya_portable_v1':
        raise ValueError('El paquete no es un modelo portátil compatible. Los artefactos históricos no se migran automáticamente.')
    contract = bundle['contract']
    detector = detector or bundle['audit']['seleccion_detector']
    regressor = regressor or bundle['audit']['seleccion_cantidad']
    if detector not in FAMILIES or regressor not in bundle['regressors']:
        raise ValueError('Elija detector y nivel explícitamente: la validación puede carecer de clases para seleccionarlos automáticamente.')
    source = simulation if simulation is not None else bundle.get('simulation_reference')
    frame=canonical(frame)
    if contract.temporal:
        for col in sum((list(contract.views[v]) for v in VIEWS), []):
            if col not in frame:frame[col]=np.nan
    frame, provenance = complete_inputs(frame, contract, policies, source,
                                       fields=contract.views['topografia'] if contract.temporal else None)
    history_assumed=False
    if history is not None:
        history, hp = complete_inputs(history, contract, history_policies, source)
        history_assumed=bool(hp.ne('').any().any())
    frame, x, _, eligible = prepare(frame, contract, history=history)
    out = frame[list(contract.keys)].copy()
    out.insert(0,'fila_archivo',np.arange(len(frame))+2)
    out['fecha'] = frame._fecha
    out['modelo_presencia'] = detector
    out['modelo_nivel'] = regressor
    out['probabilidad_roya'] = np.nan
    out['presencia_roya'] = pd.array([pd.NA]*len(frame), dtype='boolean')
    out['nivel_roya'] = np.nan
    out['estado'] = np.where(eligible, 'calculado_experimental', 'historia_insuficiente_t_1_t_2')
    out['sitio_nuevo'] = [tuple(row) not in bundle['sites'] for row in frame[list(contract.keys)].to_numpy()]
    out['fuera_rango_entrenamiento'] = False
    out['datos_imputados'] = x[contract.numeric].isna().any(axis=1)
    out['historial_con_sustituciones'] = history_assumed
    for name in provenance:
        out['variables_'+name]=provenance[name].to_numpy()
    for col, (lo, hi) in bundle['ranges'].items():
        out['fuera_rango_entrenamiento'] |= x[col].lt(lo) | x[col].gt(hi)
    out['categoria_nueva'] = False
    pre = bundle['quantity_input']
    for col, levels in zip(pre.categorical, pre.encoder.categories_ if pre.encoder is not None else []):
        out['categoria_nueva'] |= ~x[col].isin(levels) & eligible
    if eligible.any():
        probs = probabilities(bundle['models'], x.loc[eligible])
        probs['MIXTA'] = mixture(probs, bundle['mixture'])
        det = bundle['detectors'][detector]
        p = det['model'].predict_proba(probability_tensor(probs[detector]), context(frame.loc[eligible]))[:,1]
        decision = p >= det['threshold']
        out.loc[eligible,'probabilidad_roya'] = p
        out.loc[eligible,'presencia_roya'] = decision
        out.loc[eligible,'nivel_roya'] = 0.
        positive = np.flatnonzero(eligible)[decision]
        if len(positive):
            z = bundle['quantity_input'].transform(x.iloc[positive])
            z = np.column_stack([z, *[probs[f][decision] for f in FAMILIES[:3]]])
            if regressor in ('armonica','armonica_polinomio_2'):
                z = np.column_stack([z, harmonics(x.iloc[positive])])
            estimated = bundle['regressors'][regressor].predict(z)
            if not np.isfinite(estimated).all():
                raise ValueError('El estimador produjo valores no finitos. Revise la versión y las entradas fuera de rango.')
            out.loc[positive,'nivel_roya'] = np.clip(estimated, 0, contract.max_level)
    out['clase_roya'] = pd.array(np.ceil(out.nivel_roya/3), dtype='Int64')
    inconsistent = out.presencia_roya.fillna(False) & out.nivel_roya.le(0)
    out.loc[inconsistent, 'clase_roya'] = pd.NA
    out.loc[inconsistent, 'estado'] = 'revisar_presencia_positiva_con_nivel_cero'
    return out


def harmonics(x):
    """Dos primeros armónicos de mes y día, explícitos además de las covariables."""
    parts=[]
    for kind in ('mes','dia'):
        s,c=x[kind+'_sen'].to_numpy(),x[kind+'_cos'].to_numpy()
        parts.extend([s,c,2*s*c,c*c-s*s])
    return np.column_stack(parts)


def save(bundle, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    joblib.dump(bundle, destination/'modelo.joblib', compress=3)
    (destination/'protocolo.json').write_text(json.dumps(bundle['audit'], ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for name, key in [('matrices','matrices'),('cantidad','quantity_metrics'),('cascadas','cascade_metrics')]:
        pd.DataFrame(bundle[key]).to_csv(destination/(name+'.csv'), index=False)
    if 'evaluation' in bundle:
        bundle['evaluation'].to_csv(destination/'evaluacion_real_predicha.csv',index=False)
        from roya.informe_base import build
        build(bundle,destination)
    return destination


def load_model(path):
    """Sólo paquetes locales de confianza: joblib puede ejecutar código al cargar."""
    bundle = joblib.load(Path(path)/'modelo.joblib')
    if bundle.get('kind') != 'roya_portable_v1':
        raise ValueError('Formato de modelo incompatible.')
    return bundle
