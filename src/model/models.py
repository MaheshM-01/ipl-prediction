from __future__ import annotations

import os,sys,joblib,mlflow
import mlflow.sklearn
from typing import Any,Dict,Tuple,List
from pathlib import Path
from omegaconf import DictConfig,OmegaConf


import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
import warnings,joblib
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold,cross_validate,TimeSeriesSplit
from sklearn.metrics import make_scorer,accuracy_score, mean_absolute_percentage_error,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,auc,precision_recall_curve,average_precision_score,classification_report,confusion_matrix,mean_squared_error,r2_score,mean_absolute_error
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, StackingRegressor

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
if not MLFLOW_TRACKING_URI:
    default_mlruns_dir = Path(__file__).resolve().parents[2] / "mlruns"
    default_mlruns_dir.mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_URI = f"file:{default_mlruns_dir.as_posix()}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("IPL prediction system")



class Model1:
    def __init__(self,C:float=0.0100,max_iter:int=1000,random_state:int=42):
        self.C=C
        self.max_iter=max_iter
        self.random_state=random_state
        self.pipeline=None

        self.num_cols=[
            'team1_win_rate', 'team2_win_rate', 'team1_recent_form', 'team2_recent_form',
            'h2h_win_rate', 'venue_team1_win_rate', 'venue_bat_first_win_rate',
            'venue_toss_win_rate', 'win_rate_diff', 'net_venue_win_rate', 'recent_form_diff'
        ]
        self.cat_cols = ['venue', 'team1', 'team2', 'toss_decision']
        self.binary_cols = []
        self.all_features = self.num_cols + self.cat_cols + self.binary_cols
        self.latest_metrics={}


    def build_preprocessor(self)->ColumnTransformer:

        num_pipe=Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
        ])
        cat_pipe=Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
        ])
        binary_pipe=Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='most_frequent'))
        ])

        preprocessor=ColumnTransformer(transformers=[
            ('num',num_pipe,self.num_cols),
            ('cat',cat_pipe,self.cat_cols),
            ('binary',binary_pipe,self.binary_cols)
        ],remainder='drop')


        return preprocessor
    
    def train(self,train_df:pd.DataFrame,test_df:pd.DataFrame,target_col:str='y')->Dict[str,float]:
        x_train=train_df[self.all_features]
        y_train=train_df[target_col].astype(int)
        x_test=test_df[self.all_features]
        y_test=test_df[target_col].astype(int)

        print(f' x_train,y_train,x_test,y_test shapes: {x_train.shape},{y_train.shape},{x_test.shape},{y_test.shape}')

        preprocessor=self.build_preprocessor()
        self.pipeline=Pipeline(steps=[
            ('preprocessor',preprocessor),
            ('classifier',LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver='lbfgs',
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

        self.pipeline.fit(x_train,y_train)

        train_proba=self.pipeline.predict_proba(x_train)[:,1]
        train_auc=roc_auc_score(y_train,train_proba)

        test_preds=self.pipeline.predict(x_test)
        test_proba=self.pipeline.predict_proba(x_test)[:,1]

        test_auc=roc_auc_score(y_test,test_proba)
        test_acc=accuracy_score(y_test,test_preds)
        test_f1=f1_score(y_test,test_preds)

        print(f'Train ROC AUC: {train_auc:.4f}')
        print(f'Test ROC AUC : {test_auc:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Test F1 Score: {test_f1:.4f}\n')
        print('Classification Report:')
        print(classification_report(y_test, test_preds))

        cm=confusion_matrix(y_test,test_preds)

        print(f'Confusion Matrix Check:\ntn={cm[0,0]} fp={cm[0,1]}\nfn={cm[1,0]} tp={cm[1,1]}')
        
        return {"test_auc": test_auc, "test_accuracy": test_acc, "test_f1": test_f1}
    
    def save_model(self,filepath:str):
        if self.pipeline is None:
            raise ValueError('model has not been trained yet')
        os.makedirs(os.path.dirname(filepath),exist_ok=True)

        model_payload={
            'pipeline':self.pipeline,
            'feature_cols':self.all_features,
            'num_cols':self.num_cols,
            'cat_cols':self.cat_cols,
            'binary_cols':self.binary_cols

        }

        with mlflow.start_run(run_name='model1_outcome_classifier'):
            mlflow.log_param('Model_type','LogisticRegression')
            mlflow.log_param('Regularization_C',str(self.C))
            mlflow.log_metric('test_accuracy',float(self.latest_metrics.get('test_accuracy',0.0)))
            mlflow.log_metric('test_auc',float(self.latest_metrics.get('test_auc',0.0)))
            mlflow.sklearn.log_model(sk_model=self.pipeline,artifact_path='model_classifier_artifact')


        joblib.dump(model_payload,filepath)
        print(f'model 1 pipeline bundle successfully saved to {filepath}')


class Model2:

    def __init__(self)-> None:
        self.num_cols=[
            'pp_crr', 'pp_dot_pct', 'pp_boundary_pct',
            'mid_crr', 'mid_run_per_wkt',
            'venue_avg_score', 'batting_team_season_avg',
            'bowling_team_pp_econ', 'batting_team_rolling', 'h2h_avg'
            
            ]
        self.cat_cols=['batting_team', 'bowling_team', 'venue']
        self.pass_cols=['pp_runs', 'pp_wickets', 'pp_fours', 'pp_sixes',
            'pp_boundaries', 'pp_runs_per_wicket',
            'mid_runs', 'mid_wickets',
            'season_yr', 'is_impact_era', 'is_t20_boom_era', 'is_pre2022']

        self.all_features=self.num_cols+self.cat_cols+self.pass_cols
        self.pipeline=None
        self.feature_cols=None
        self.x_train=None
        self.y_train=None
        self.tscv=TimeSeriesSplit(n_splits=5)

        self.test_mae_score=0.0

    
    def build_preprocessor(self)->ColumnTransformer:
        num_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        pass_pipe = Pipeline(steps=[
            ('pass', SimpleImputer(strategy='constant', fill_value=0)) # Prevent passthrough leakage issues
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('cat', cat_pipe, self.cat_cols),
            ('num', num_pipe, self.num_cols),
            ('pass', pass_pipe, self.pass_cols)
        ], remainder='drop')

        return preprocessor

    def cv_mae(self,pipeline:Pipeline)->float:
        scores=[]
        for tr,val in self.tscv.split(self.x_train):
            x_tr,y_tr=self.x_train.iloc[tr],self.y_train.iloc[tr]
            x_val,y_val=self.x_train.iloc[val],self.y_train.iloc[val]

            current_pipe=Pipeline(steps=list(pipeline.named_steps.items()))
            current_pipe.fit(x_tr,y_tr)
            preds=current_pipe.predict(x_val)
            scores.append(mean_absolute_error(y_val,preds))

        return float(np.mean(scores))

    def xgb_objective(self,trial:optuna.Trial)->float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        pipe = Pipeline([
            ('preprocessor', self.build_preprocessor()),
            ('model', XGBRegressor(**params))
        ])
        return self.cv_mae(pipe)
    
    def lgbm_objective(self, trial: optuna.Trial) -> float:
        """Optuna Objective function tracking LightGBM parameter adjustments."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 40),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 2.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        pipe = Pipeline([
            ('preprocessor', self.build_preprocessor()),
            ('model', LGBMRegressor(**params))
        ])
        return self.cv_mae(pipe)
    

    def tune_hyperparameters(self)-> Tuple[Dict[str,Any],Dict[str,Any]]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print('tuning xgboost hyperparameters')
        xgb_study=optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        xgb_study.optimize(self.xgb_objective,n_trials=50,show_progress_bar=True)
        best_xgb_p=xgb_study.best_params.copy()
        if 'lr' in best_xgb_p: best_xgb_p['learning_rate'] = best_xgb_p.pop('lr')
        best_xgb_p.update({'random_state':42,'n_jobs':-1,'verbosity':0})
        print(f'best xgb cv mae: {xgb_study.best_value:.2f}')

        print('\n training lightgbm hyperparameters')
        lgbm_study=optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        lgbm_study.optimize(self.lgbm_objective,n_trials=50,show_progress_bar=True)
        best_lgbm_p=lgbm_study.best_params.copy()
        if 'lr' in best_lgbm_p: best_lgbm_p['learning_rate']=best_lgbm_p.pop('lr')
        best_lgbm_p.update({'random_state':42,'n_jobs':-1,'verbose':-1})
        print(f' best lgbm cv mae :{lgbm_study.best_value:.2f}')

        return best_xgb_p,best_lgbm_p
    
    def train_pipeline(self,train_df:pd.DataFrame,test_df: pd.DataFrame,target_col: str='y')-> Tuple[pd.DataFrame,List[str]]:

        self.x_train=train_df[self.all_features]
        self.y_train=train_df[target_col].astype(float)
        x_test=test_df[self.all_features]
        y_test=test_df[target_col].astype(float)

        best_xgb_p,best_lgbm_p=self.tune_hyperparameters()

        tuned_xgb=XGBRegressor(**best_xgb_p)
        tuned_lgbm=LGBMRegressor(**best_lgbm_p)
        tuned_rf=RandomForestRegressor(
            n_estimators=300,max_depth=4,min_samples_leaf=10,max_features=0.6,random_state=42,n_jobs=-1
        )

        stacking_cv=KFold(n_splits=5,shuffle=False)
        stacker=StackingRegressor(
            estimators=[
                ('xgb',tuned_xgb),
                ('lgbm',tuned_lgbm),
                ('rf',tuned_rf)
            ],
            final_estimator=Ridge(alpha=10.0),
            cv=stacking_cv,
            n_jobs=-1,

        )

        self.pipeline=Pipeline(steps=[
            ('preprocessor',self.build_preprocessor()),
            ('stacker',stacker)
        ])
        print(f' \n fitting data onto the final multi-model stacking ensemble')
        self.pipeline.fit(self.x_train,self.y_train)

        train_preds=self.pipeline.predict(self.x_train)

        print(f'train mae: {mean_absolute_error(self.y_train,train_preds):.2f}')

        y_pred=self.pipeline.predict(x_test)
        test_mae=mean_absolute_error(y_test,y_pred)
        self.test_mae_score=float(test_mae)
        
        test_rmse=mean_squared_error(y_test,y_pred)**0.5
        test_r2=r2_score(y_test,y_pred)
        test_mape=mean_absolute_percentage_error(y_test,y_pred)* 100.0

        print('\n============================================================')
        print(' FINAL ENSEMBLE TESTING PERFORMANCE METRICS')
        print('============================================================')
        print(f'Test Set MAE   : {test_mae:.2f} runs')
        print(f'Test Set RMSE  : {test_rmse:.2f} runs')
        print(f'Test Set R² Score: {test_r2:.4f}')
        print(f'Test Set MAPE  : {test_mape:.2f}%')


        within_10 = (np.abs(y_test - y_pred) <= 10).mean() * 100.0
        within_15 = (np.abs(y_test - y_pred) <= 15).mean() * 100.0
        within_20 = (np.abs(y_test - y_pred) <= 20).mean() * 100.0


        print(f'Predictions within ±10 run boundary: {within_10:.1f}%')
        print(f'Predictions within ±15 run boundary: {within_15:.1f}%')
        print(f'Predictions within ±20 run boundary: {within_20:.1f}%')

        model2_predictions_df=test_df.copy()
        model2_predictions_df['predicted_y']=y_pred
        model2_predictions_df['absolute_error']=np.abs(y_test-y_pred)

        return model2_predictions_df,self.all_features
    
    def save_model(self,filepath:str):
        if self.pipeline is None:
            raise ValueError('model has not been trained yet')
        os.makedirs(os.path.dirname(filepath),exist_ok=True)

        model_payload={
            'pipeline':self.pipeline,
            'feature_cols':self.all_features,
            'num_cols':self.num_cols,
            'cat_cols':self.cat_cols,
            'pass_cols':self.pass_cols
        }
        with mlflow.start_run(run_name='model2_firstInnings_regressor'):
            mlflow.log_param('model_type','stacking_regressor')
            mlflow.log_metric('test_mae',float(self.test_mae_score))
            mlflow.sklearn.log_model(sk_model=self.pipeline,artifact_path='model2_stacker_artifact')


        joblib.dump(model_payload,filepath)
        print(f'model 2 pipeline bundle successfully saved to {filepath}')
        
              
            
class Model3:
    def __init__(self):
        self.cat_cols=['batting_team', 
        'bowling_team', 
        'venue']
        self.num_cols=['target_runs', 
        'pp_runs', 
        'pp_wickets', 
        'pp_boundaries',
        'pp_crr', 
        'rrr_after_pp', 
        'pressure_index',
        'venue_chase_avg', 
        'team_recent_chase_form']
        self.pass_cols=['is_impact_era', 
        'is_t20_boom_era',
        'season_yr' ]

        self.all_features=self.cat_cols+self.num_cols+self.pass_cols
        self.pipeline=None
        self.feature_cols=list(self.all_features)
        self.x_train=None
        self.y_train=None
        self.tscv=TimeSeriesSplit(n_splits=5)
        self.winning_model=None
        self.holdout_mae_score=0.0
        self.best_model_name=''

    def build_preprocessor(self)->ColumnTransformer:
        preprocessor=ColumnTransformer(
        transformers=[
            ('cat',Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
            ]),self.cat_cols),
            ('num',Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scaling',StandardScaler())
            ]),self.num_cols),
            ('pass','passthrough',self.pass_cols),

        ],remainder='drop')
        return preprocessor
    
    def cv_mae(self,pipeline:Pipeline)->float:
        scores=[]
        for tr,val in self.tscv.split(self.x_train):
            x_tr,y_tr=self.x_train.iloc[tr],self.y_train.iloc[tr]
            x_val,y_val=self.x_train.iloc[val],self.y_train.iloc[val]

            current_pipe=Pipeline(steps=list(pipeline.named_steps.items()))
            current_pipe.fit(x_tr,y_tr)
            preds=current_pipe.predict(x_val)
            scores.append(mean_absolute_error(y_val,preds))
        return float(np.mean(scores))
    
    def xgb_objective(self,trail:optuna.Trial)->float:
        params=dict(
            n_estimators=trail.suggest_int('n_estimators',100,600),
            max_depth=trail.suggest_int('max_depth',2,6),
            learning_rate=trail.suggest_float('lr',0.005,0.1,log=True),
            subsample=trail.suggest_float('subsample',0.5,0.9),
            colsample_bytree=trail.suggest_float('colsample_bytree',0.4,0.9),
            min_child_weight=trail.suggest_int('min_child_weight',10,50),
            reg_alpha=trail.suggest_float('reg_alpha',1.0,10.0),
            reg_lambda=trail.suggest_float('reg_lambda',1.0,15.0),
            gamma=trail.suggest_float('gamma',0.0,5.0),
            random_state=42,n_jobs=-1,verbosity=0,
        )

        pipe=Pipeline([('preprocessor',self.build_preprocessor()),
                   ('model',XGBRegressor(**params))])
        return self.cv_mae(pipe)
    
    def lgbm_objective(self,trial:optuna.Trial)->float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 40),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 2.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        pipe=Pipeline([
            ('preprocessor',self.build_preprocessor()),
            ('model',LGBMRegressor(**params))
        ])
        return self.cv_mae(pipe)
    
    def hyperparameter_tuning(self)->Tuple[Dict[str,Any],Dict[str,Any]]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f' tuning the xgb')
        xgb_study=optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        xgb_study.optimize(self.xgb_objective,n_trials=50,show_progress_bar=True)
        best_xgb_p=xgb_study.best_params.copy()
        if 'lr' in best_xgb_p: best_xgb_p['learning_rate']=best_xgb_p.pop('lr')
        best_xgb_p.update({'random_state':42,'n_jobs':-1,'verbosity':0})
        xgb_cv_mae=float(xgb_study.best_value)
        print(f' best xgb cv mae: {xgb_cv_mae:.2f}')

        print(f'\n tuning the lgbm')
        lgbm_study=optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        lgbm_study.optimize(self.lgbm_objective,n_trials=50,show_progress_bar=True)
        best_lgbm_p=lgbm_study.best_params.copy()
        if 'lr' in best_lgbm_p: best_lgbm_p['learning_rate']=best_lgbm_p.pop('lr')
        best_lgbm_p.update({'random_state':42,'n_jobs':-1,'verbose':-1})
        lgbm_cv_mae=float(lgbm_study.best_value)
        print(f' best lgbm cv mae: {lgbm_cv_mae:.2f}')

        return best_xgb_p, xgb_cv_mae, best_lgbm_p, lgbm_cv_mae
    


    # Inside src/models/models.py -> Class Model3

    def train_pipeline(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = 'final_chase_score') -> Tuple[pd.DataFrame, List[str]]:
        self.feature_cols = list(self.all_features)
        self.x_train = train_df[self.all_features]
        self.y_train = train_df[target_col].astype(float)
        x_test = test_df[self.all_features]
        y_test = test_df[target_col].astype(float)

        # 1. Gather optimized parameter dictionaries and their tuned CV MAE scores
        best_xgb_p, xgb_cv_mae, best_lgbm_p, lgbm_cv_mae = self.hyperparameter_tuning()

        # 2. Instantiate the fine-tuned base estimators
        tuned_xgb = XGBRegressor(**best_xgb_p)
        tuned_lgbm = LGBMRegressor(**best_lgbm_p)
        tuned_rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)

        # 3. Calculate Stacking Ensemble performance for competition
        print("\n Checking Stacking Ensemble score for Model 3...")
        stacking_cv = KFold(n_splits=5, shuffle=False)
        stacker = StackingRegressor(
            estimators=[('xgb', tuned_xgb), ('lgbm', tuned_lgbm), ('rf', tuned_rf)],
            final_estimator=Ridge(alpha=5.0),
            cv=stacking_cv,
            n_jobs=-1
        )
        
        # Test ensemble score using historical windows splits
        stacker_pipe = Pipeline([('preprocessor', self.build_preprocessor()), ('model', stacker)])
        # Evaluate cross validation negative MAE scores
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(stacker_pipe, self.x_train, self.y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
        stacker_cv_mae = float(np.abs(scores.mean()))
        print(f" Best Stacking Ensemble CV MAE: {stacker_cv_mae:.2f}")

        # -----------------------------------------------------------------
        # AUTOMATED SELECTION MATRIX (Selecting the absolute champion!)
        # -----------------------------------------------------------------
        score_matrix = {
            "xgb": xgb_cv_mae,
            "lgbm": lgbm_cv_mae,
            "stacker": stacker_cv_mae
        }
        
        # Find the architecture name with the absolute minimum error
        self.best_model_name = min(score_matrix, key=score_matrix.get)
        print(f"\n ARCHITECTURE SELECTED FOR MODEL 3: [{self.best_model_name.upper()}] with MAE: {score_matrix[self.best_model_name]:.2f}")

        # 4. Map the winning candidate to the final training pipeline instance
        if self.best_model_name == "xgb":
            champion_estimator = tuned_xgb
        elif self.best_model_name == "lgbm":
            champion_estimator = tuned_lgbm
        else:
            champion_estimator = stacker

        self.pipeline = Pipeline([
            ('preprocessor', self.build_preprocessor()),
            ('model', champion_estimator)
        ])

        # 5. Fit the ultimate winner onto the whole dataset
        print(f" Fitting the entire training context onto final {self.best_model_name.upper()} pipeline...")
        self.pipeline.fit(self.x_train, self.y_train)

        # 6. Evaluate test holdout set predictions (2024 Season)
        preds = self.pipeline.predict(x_test)
        test_df = test_df.copy()
        test_df['preds_m3'] = preds

        # Calculate standard production regression evaluation metrics
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, preds)
        self.holdout_mae=float(mae)

        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

      
        print(f" model 3 metrics ({self.best_model_name.upper()})")
        print()
        print(f" Test Set Mean Absolute Error (MAE) : {mae:.2f} runs")
        print(f" Test Root Mean Squared Error (RMSE): {rmse:.2f} runs")
        print(f" Test Set R-Squared Score (R²)       : {r2:.4f}")
        print(f" Test Mean Absolute Pct Error (MAPE): {mape:.2f}%")
        print()

        # Return features lists for production mapping tracking pipelines bundles
        return test_df, self.all_features
    
 
    

    def save_model(self,filepath:str):
        if self.pipeline is None:
            raise RuntimeError('run train_pipeline() first')
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        resolved_feature_cols = self.feature_cols if self.feature_cols is not None else list(self.all_features)
        payload={
            'pipeline':self.pipeline,
            'feature_cols':resolved_feature_cols,
            'num_cols':self.num_cols,
            'cat_cols':self.cat_cols,
            'pass_cols':self.pass_cols,
            'winning_model': self.best_model_name
        }

        with mlflow.start_run(run_name='model3_chase_score_prediction'):
            mlflow.log_param('winning_algorithm',self.best_model_name)
            mlflow.log_metric('holdout_mae',float(self.holdout_mae))
            mlflow.sklearn.log_model(sk_model=self.pipeline,artifact_path='model3_chase_predictor')
            
        joblib.dump(payload,filepath)
        print(f' model 3 pipeline bundle successfully saved to {filepath}')

        
        
