"""Spark preparation and CPU AutoML for early diabetes readmission."""
from pathlib import Path
import json
import os
import sys
import importlib.metadata as metadata
import pandas as pd
import numpy as np
from src.preprocessing import canonicalize_missing

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

def prepare():
    from pyspark.sql import SparkSession, functions as F, Window
    os.environ.setdefault('JAVA_HOME', r'C:\Program Files\Microsoft\jdk-11.0.16.101-hotspot')
    os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
    os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
    spark = (SparkSession.builder.master('local[2]').appName('diabetes-readmission')
             .config('spark.driver.memory', '1g').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    raw = spark.read.option('header', True).option('inferSchema', True).csv(str(ROOT/'data/raw/diabetic_data.csv'))
    original = raw.count()
    clean = raw.replace('?', None).dropDuplicates()
    numeric = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications',
               'number_outpatient','number_emergency','number_inpatient','number_diagnoses']
    for col in numeric:
        clean = clean.withColumn(col, F.col(col).cast('double'))
    clean = (clean.withColumn('total_prior_visits', F.col('number_outpatient')+F.col('number_emergency')+F.col('number_inpatient'))
             .withColumn('medications_per_day', F.col('num_medications')/F.col('time_in_hospital'))
             .withColumn('labs_per_day', F.col('num_lab_procedures')/F.col('time_in_hospital'))
             .withColumn('early_readmission', (F.col('readmitted') == '<30').cast('int')))
    clean.printSchema()
    clean.show(10, truncate=False)
    # One earliest recorded encounter per patient: no patient can cross folds or holdout.
    unique = clean.withColumn('_rn', F.row_number().over(Window.partitionBy('patient_nbr').orderBy('encounter_id'))).where('_rn=1').drop('_rn')
    keep = numeric + ['patient_nbr','total_prior_visits','medications_per_day','labs_per_day',
        'race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
        'max_glu_serum','A1Cresult','insulin','change','diabetesMed','early_readmission']
    pdf = unique.select(keep).toPandas()
    # Stable global order makes the seeded holdout independent of Spark partitions.
    pdf = pdf.sort_values('patient_nbr').reset_index(drop=True)
    # Administrative codes are categories, not ordered magnitudes.
    for col in ['admission_type_id','discharge_disposition_id','admission_source_id']:
        pdf[col] = pdf[col].astype(str)
    pdf = pdf.replace({None: np.nan})
    (ROOT/'data/processed').mkdir(exist_ok=True)
    pdf.to_parquet(ROOT/'data/processed/readmission.parquet', index=False)
    counts = {'raw_rows':original, 'deduplicated_encounters':clean.count(), 'unique_patients':len(pdf)}
    spark.stop()
    print(counts)
    return pdf, counts

def train(pdf, counts):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
    from pycaret.classification import ClassificationExperiment
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    train_full, test = train_test_split(pdf, test_size=.2, random_state=SEED, stratify=pdf.early_readmission)
    train_data, _ = train_test_split(train_full, train_size=min(18000, len(train_full)-1), random_state=SEED, stratify=train_full.early_readmission)
    assert not pdf.patient_nbr.duplicated().any()
    overlap = len(set(train_full.patient_nbr) & set(test.patient_nbr))
    assert overlap == 0
    counts.update(training_pool=len(train_full), automl_rows=len(train_data), holdout_rows=len(test),
        training_positives=int(train_data.early_readmission.sum()),
        holdout_positives=int(test.early_readmission.sum()), patient_overlap=overlap)
    train_data = train_data.drop(columns='patient_nbr').reset_index(drop=True)
    test = test.drop(columns='patient_nbr').reset_index(drop=True)
    versions = {name:metadata.version(name) for name in ['pycaret','pyspark','scikit-learn','pandas','mlflow']}
    mlflow.set_tracking_uri('sqlite:///' + str(ROOT/'mlflow.db').replace('\\','/'))
    mlflow.set_experiment('diabetes-readmission-variant-fourteen')
    results = []
    for label, models in [('linear-trees',['lr','dt','rf']), ('boosting-discriminant',['ada','gbc','lda'])]:
        exp = ClassificationExperiment()
        # PyCaret's internal holdout is drawn from training only; external test remains sealed.
        exp.setup(data=train_data, target='early_readmission', session_id=SEED,
            fold_strategy='stratifiedkfold', fold=3, fold_shuffle=True,
            normalize=True, n_jobs=2, verbose=False, html=False, log_experiment=False,
            custom_pipeline=[('canonicalize_missing', FunctionTransformer(canonicalize_missing,
                validate=False, feature_names_out='one-to-one'))], custom_pipeline_position=0,
            system_log=False)
        with mlflow.start_run(run_name=label) as run:
            best = exp.compare_models(include=models, sort='AUC', fold=3, n_select=1,
                budget_time=2, errors='raise', verbose=False)
            board = exp.pull()
            assert len(board) >= 3, 'Time budget did not complete all three candidates'
            print(f'Cross-validation leaderboard: {label}\n{board.to_string()}')
            board.to_csv(ROOT/f'reports/leaderboard_{label}.csv')
            # Finalization includes only supplied train_data, never external test.
            final = exp.finalize_model(best)
            X = test.drop(columns='early_readmission')
            prob = final.predict_proba(X)[:,1]
            pred = final.predict(X)
            metrics = dict(roc_auc=roc_auc_score(test.early_readmission,prob),
                accuracy=accuracy_score(test.early_readmission,pred),
                f1=f1_score(test.early_readmission,pred,zero_division=0),
                pr_auc=average_precision_score(test.early_readmission,prob))
            exp.save_model(final, str(ROOT/f'models/{label}'), verbose=False)
            mlflow.log_params(dict(models=','.join(models), budget_minutes=2, folds=3,
                fold_strategy='stratifiedkfold', seed=SEED, library='PyCaret',
                selected_estimator=type(best).__name__, **counts))
            mlflow.log_metrics(metrics)
            mlflow.log_dict(versions,'versions.json')
            mlflow.log_dict(counts,'dataset_counts.json')
            mlflow.log_artifact(str(ROOT/f'reports/leaderboard_{label}.csv'))
            mlflow.sklearn.log_model(final, 'model', signature=infer_signature(X, prob),
                code_paths=[str(ROOT/'src')],
                pip_requirements=[f'{name}=={version}' for name, version in versions.items()])
            results.append(dict(run=label,run_id=run.info.run_id,selected=type(best).__name__,cv_auc=float(board.iloc[0]['AUC']),**metrics))
    result = {'counts':counts,'versions':versions,'runs':results,
              'selected_by_cv':max(results,key=lambda r:r['cv_auc'])['run']}
    (ROOT/'reports/results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

def reload_check(result):
    from pycaret.classification import load_model
    pdf = pd.read_parquet(ROOT/'data/processed/readmission.parquet')
    model = load_model(str(ROOT/'models'/result['selected_by_cv']),verbose=False)
    probabilities = model.predict_proba(pdf.drop(columns=['early_readmission','patient_nbr']).head(10))
    assert probabilities.shape == (10,2) and np.isfinite(probabilities).all()
    print('Reloaded full preprocessing pipeline:', probabilities.shape)
    return probabilities

if __name__ == '__main__':
    data, counts = prepare()
    result = train(data, counts)
    reload_check(result)
