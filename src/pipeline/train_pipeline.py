

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from src.features.features import Merging,FeatureEngineeringModel1,FeatureEngineeringModel2,FeatureEngineeringModel3
from src.model.models import Model1,Model2,Model3


class IPLTrainingPipeline:
    def __init__(self,cfg:DictConfig)->None:
        self.cfg=cfg

        self.merger=Merging(cfg=self.cfg)
        self.fe1=FeatureEngineeringModel1(cfg=self.cfg)
        self.fe2=FeatureEngineeringModel2(cfg=self.cfg)
        self.fe3=FeatureEngineeringModel3(cfg=self.cfg)

        self.model1=Model1()
        self.model2=Model2()
        self.model3=Model3()

    def run(self)->None:
        print(f' training pipeline - inintiating historical update')

        df_matches_raw=pd.read_csv(self.cfg.paths.raw_matches)
        df_deliveries_raw=pd.read_csv(self.cfg.paths.raw_deliveries)

        matches_clean,merged_clean,second_inn_clean=self.merger.merge_datasets(
            df_matches_raw,df_deliveries_raw

        )

        matches_clean.to_csv(os.path.join(self.cfg.paths.processed_data,'matches_clean.csv'),index=False)
        merged_clean.to_csv(os.path.join(self.cfg.paths.processed_data,'merged_clean.csv'),index=False)
        second_inn_clean.to_csv(os.path.join(self.cfg.paths.processed_data,'second_inn_clean.csv'),index=False)

        train_season=[2019,2020,2021,2022,2023]
        test_season=[2024]

        print(f' feaure engineering-1')
        dataset_1=self.fe1.season_feature1(matches_clean)
        dataset_1=self.fe1.build_features(dataset_1)
        
        dataset_1['team2_is_home']=0
        dataset_1['team2_is_home']=0
        dataset_1=dataset_1[dataset_1['winner'].notna()].copy()
        dataset_1['y']=(dataset_1['winner']==dataset_1['team1']).astype(int)
        dataset_1['net_venue_win_rate']=dataset_1['venue_team1_win_rate']-0.5

        train_m1=dataset_1[dataset_1['season_yr'].isin(train_season)].copy()
        test_m1=dataset_1[dataset_1['season_yr'].isin(test_season)].copy()

        self.model1.train(train_df=train_m1,test_df=test_m1,target_col='y')
        self.model1.save_model(os.path.join(self.cfg.paths.models_dir,'model1_winner_classifier.pkl'))


        print(f' feature engineering -2')
        df2_state=self.fe2.create_features2(merged_clean)
        _=self.fe2.pp_features2(df2_state)
        _=self.fe2.model2_features2(df2_state)
        _=self.fe2.model2_2featurers(df2_state)
        _=self.fe2.build_features2(df2_state)
        dataset_2,_=self.fe2.model2_df(df2_state)

        train_m2=dataset_2[dataset_2['season_yr'].isin(train_season)]
        test_m2=dataset_2[dataset_2['season_yr'].isin(test_season)]

        print(f'optimizing model 2 hyperparameters')

        _, _=self.model2.train_pipeline(train_df=train_m2,test_df=test_m2,target_col='y')

        self.model2.save_model(os.path.join(self.cfg.paths.models_dir,'model2_first_innings_stacker.pkl'))


        print(f' feature engineering -3')
        df3_state=self.fe3.create_features3(second_inn_clean)
        _=self.fe3.match_features3(df3_state)
        dataset_3,_,_,_ =self.fe3.build_features3(df3_state)

        train_m3=dataset_3[dataset_3['season_yr'].isin(train_season)].copy()
        test_m3=dataset_3[dataset_3['season_yr'].isin(test_season)].copy()

        print(f'running the model 3 training pipeline and selecting the best')
        _, _=self.model3.train_pipeline(train_df=train_m3,test_df=test_m3,target_col='final_chase_score')
        self.model3.save_model(os.path.join(self.cfg.paths.models_dir,'model3_chase_predictor.pkl'))

        print(f' training pipeline completed successfully')
