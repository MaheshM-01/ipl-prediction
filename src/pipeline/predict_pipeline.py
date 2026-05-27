from __future__ import annotations
import os,sys
import pandas as pd
import numpy as np
import joblib
from omegaconf import DictConfig
from typing import Dict,Any
from src.features.features import Merging,FeatureEngineeringModel1,FeatureEngineeringModel2,FeatureEngineeringModel3

class IPLInferencePipeline:
    def __init__(self,cfg:DictConfig)-> None:
        self.cfg=cfg
        self.merger=Merging(cfg=self.cfg)
        self.fe1=FeatureEngineeringModel1(cfg=self.cfg)
        self.fe2=FeatureEngineeringModel2(cfg=self.cfg)
        self.fe3=FeatureEngineeringModel3(cfg=self.cfg)

        self.m1_bundle=joblib.load(os.path.join(self.cfg.paths.models_dir,'model1_winner_classifier.pkl'))
        self.m2_bundle=joblib.load(os.path.join(self.cfg.paths.models_dir,'model2_first_innings_stacker.pkl'))
        self.m3_bundle=joblib.load(os.path.join(self.cfg.paths.models_dir,'model3_chase_predictor.pkl'))


        self.hist_matches=pd.read_csv(os.path.join(self.cfg.paths.processed_data,'matches_clean.csv'))
        self.hist_merged=pd.read_csv(os.path.join(self.cfg.paths.processed_data,'merged_clean.csv'))

    @staticmethod
    def _resolve_feature_cols(bundle: Dict[str, Any], model_name: str) -> list[str]:
        feature_cols = bundle.get('feature_cols')
        if isinstance(feature_cols, (list, tuple)) and len(feature_cols) > 0:
            return list(feature_cols)

        # Backward-compatible fallback for older bundles where feature_cols was not persisted.
        fallback_cols: list[str] = []
        for key in ('num_cols', 'cat_cols', 'binary_cols', 'pass_cols'):
            cols = bundle.get(key)
            if isinstance(cols, (list, tuple)) and len(cols) > 0:
                fallback_cols.extend(list(cols))

        if len(fallback_cols) == 0:
            raise ValueError(f'{model_name} bundle does not contain usable feature columns.')

        # Keep order stable while removing duplicates.
        seen = set()
        ordered_cols = [c for c in fallback_cols if not (c in seen or seen.add(c))]
        return ordered_cols

    @staticmethod
    def _select_feature_frame(df: pd.DataFrame, feature_cols: list[str], model_name: str) -> pd.DataFrame:
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f'{model_name} missing feature columns: {missing_cols}')
        return df[feature_cols]

    def predict_live_stream(self,raw_unseen_matches: pd.DataFrame,raw_unseen_deliveries:pd.DataFrame)-> Dict[str,Any]:
        print(f' predicting on the live stream')

        matches_clean,merged_clean,second_inn_clean=self.merger.merge_datasets(raw_unseen_matches,raw_unseen_deliveries)
        live_match_ids=matches_clean['id'].unique() if 'id' in matches_clean.columns else  matches_clean['match_id'].unique()

        
        #score model 1 track
        comb_m1=pd.concat([self.hist_matches,matches_clean],ignore_index=True)
        comb_m1_eng=self.fe1.build_features(comb_m1)
        live_m1=comb_m1_eng[comb_m1_eng['id'].isin(live_match_ids)].copy()
        live_m1['net_venue_win_rate']=live_m1['venue_team1_win_rate']-0.5
        live_m1['team1_is_home'],live_m1['team2_is_home']=0, 0

        m1_feature_cols=self._resolve_feature_cols(self.m1_bundle,'model1')
        m1_input=self._select_feature_frame(live_m1,m1_feature_cols,'model1')
        m1_preds=self.m1_bundle['pipeline'].predict(m1_input)
        m1_probas=self.m1_bundle['pipeline'].predict_proba(m1_input)[:,1]

        #score model 2 track

        comb_m2=pd.concat([self.hist_merged,merged_clean],ignore_index=True)
        df2_st=self.fe2.create_features2(comb_m2)
        _=self.fe2.pp_features2(df2_st)
        _=self.fe2.model2_features2(df2_st)
        _=self.fe2.model2_2featurers(df2_st)
        _=self.fe2.build_features2(df2_st)

        comb_m2_fn,_=self.fe2.model2_df(df2_st)
        live_m2=comb_m2_fn[comb_m2_fn['match_id'].isin(live_match_ids)].copy()

        m2_feature_cols=self._resolve_feature_cols(self.m2_bundle,'model2')
        m2_input=self._select_feature_frame(live_m2,m2_feature_cols,'model2')
        m2_preds=self.m2_bundle['pipeline'].predict(m2_input)


        #score model 3 track

        hist_s2=self.hist_merged[self.hist_merged['inning']==2]
        comb_m3=pd.concat([hist_s2,second_inn_clean],ignore_index=True)
        df3_st=self.fe3.create_features3(comb_m3)
        _=self.fe3.match_features3(df3_st)
        comb_m3_fn,_,_,_=self.fe3.build_features3(df3_st)
        live_m3=comb_m3_fn[comb_m3_fn['match_id'].isin(live_match_ids)].copy()

        m3_feature_cols=self._resolve_feature_cols(self.m3_bundle,'model3')
        m3_input=self._select_feature_frame(live_m3,m3_feature_cols,'model3')
        m3_preds=self.m3_bundle['pipeline'].predict(m3_input)

        return {
            'match_ids': live_match_ids.tolist(),
            'winning_model_predictions': ['team 1' if p==1 else "team 2" for p in m1_preds],
            'team1_win_probabilities': m1_probas.tolist(),
            'projected_innings1_runs': m2_preds.tolist(),
            'projected_chase_scores': m3_preds.tolist()
        }
    
