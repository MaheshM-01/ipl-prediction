from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import time

from src.monitoring import IPLMonitor

app=FastAPI(
    title='IPL multi model prediction ',
    description='Unified api microserve exposing inference channels',
    version='2.0.0')

m1_path=os.path.join('models','model1_winner_classifier.pkl')
m2_path=os.path.join('models','model2_first_innings_stacker.pkl')
m3_path=os.path.join('models','model3_chase_predictor.pkl')

m1_bundle=joblib.load(m1_path)
m2_bundle=joblib.load(m2_path)
m3_bundle=joblib.load(m3_path)
monitor=IPLMonitor()

DEFAULT_CATEGORICALS = {
    "venue": "Unknown",
    "team1": "Unknown",
    "team2": "Unknown",
    "toss_decision": "Unknown",
    "batting_team": "Unknown",
    "bowling_team": "Unknown",
}


def align_features(input_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    aligned = input_df.copy()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = DEFAULT_CATEGORICALS.get(col, 0.0)
    return aligned[feature_cols]

class FullMatchStream(BaseModel):
    pp_runs: float
    pp_wickets:float
    mid_crr: float
    venue_avg_score: float
    batting_team_rolling: float
    h2h_avg: float
    team1_is_home: int=1
    team2_is_home: int=0
    venue_team1_win_rate: float=0.55

@app.get('/health')
def health_ckeck():
    try:
        monitor.update_health(status='all_systems_operatonal',models_loaded=['m1','m2','m3'])
    except Exception:
        pass
    return {
        'status': 'all_systems_operatonal',
        'models_loaded': ['m1','m2','m3']
    }

@app.post('/predict/all_tracker')
async def evaluate_full_match_lifecycle(payload: FullMatchStream):
    start_ts=time.perf_counter()
    raw_input_data={}
    try:
        raw_input_data=payload.model_dump()
        input_df=pd.DataFrame([raw_input_data])
        m1_features=m1_bundle['feature_cols']
        input_df['net_venue_win_rate']=input_df['venue_team1_win_rate']-0.5
        m1_matrix=align_features(input_df,m1_features)

        m1_pred=m1_bundle['pipeline'].predict(m1_matrix)[0]
        m1_proba=m1_bundle['pipeline'].predict_proba(m1_matrix)[:,1][0]


        m2_features=m2_bundle['feature_cols']
        m2_matrix=align_features(input_df,m2_features)
        m2_pred=m2_bundle['pipeline'].predict(m2_matrix)[0]

        m3_features=m3_bundle['feature_cols']
        m3_matrix=align_features(input_df,m3_features)
        m3_pred=m3_bundle['pipeline'].predict(m3_matrix)[0]

        response_payload={
            'status': 'success',
            'model_1_outcome': {
                'predicted_winner': 'team 1 ' if m1_pred==1 else "team 2",
                'team1_win_probability_pct': round(float(m1_proba)*100,2)

            },
            'model_2_innings1': {
                'projected_first_innings_runs': round(float(m2_pred),1)

            },
            'model_3_chase': {
                'projected_final_chase_score': round(float(m3_pred),1),
                'champion_architecture': m3_bundle.get('winning_model','stacker').upper()

            }
        }
        latency_ms=(time.perf_counter()-start_ts)*1000.0
        try:
            monitor.log_prediction(
                request_payload=raw_input_data,
                response_payload=response_payload,
                latency_ms=latency_ms,
                status_code=200
            )
        except Exception:
            pass
        return response_payload
    
    except Exception as e:
        try:
            monitor.log_error(
                error_message=str(e),
                request_payload=raw_input_data,
                stage='predict_all_tracker'
            )
        except Exception:
            pass
        raise HTTPException (status_code=500,detail=f'core inference failure :{str(e)}')
