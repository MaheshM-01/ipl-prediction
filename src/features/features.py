
from __future__ import annotations
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from typing import Tuple,List,Any,Dict


class Merging:
    def __init__(self,cfg:DictConfig)->None:
        self.config=cfg

        self.team_name_mapping={
            'Delhi Daredevils'             : 'DC',
            'Delhi Capitals'               : 'DC',
            'Deccan Chargers'              : 'SRH',
            'Sunrisers Hyderabad'          : 'SRH',
            'Kings XI Punjab'              : 'PBKS',
            'Punjab Kings'                 : 'PBKS',
            'Royal Challengers Bangalore'  : 'RCB',
            'Royal Challengers Bengaluru'  : 'RCB',
            'Rising Pune Supergiant'       : 'RPS',
            'Rising Pune Supergiants'      : 'RPS',
            'Gujarat Lions'                : 'GL',
            'Gujarat Titans'               : 'GT',
            'Lucknow Super Giants'         : 'LSG',
            'Mumbai Indians'               : 'MI',
            'Chennai Super Kings'          : 'CSK',
            'Kolkata Knight Riders'        : 'KKR',
            'Rajasthan Royals'             : 'RR',
            'Kochi Tuskers Kerala'         : 'KTK',
            'Pune Warriors'                : 'PWI',
            'Pune Warriors India'          : 'PWI',

        }



    def merge_datasets(self,matches_raw:pd.DataFrame,deliveries_raw:pd.DataFrame,)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:

        matches=matches_raw.copy()
        deliveries=deliveries_raw.copy()

        matches['date']=pd.to_datetime(matches['date'])

        matches=matches[matches['method'].isna()].copy()

        matches=matches[matches['result'].isin(['runs','wickets'])].copy()

        matches=matches[matches['super_over']=='N'].copy()

        for col in ['team1','team2','toss_winner','winner']:
            if col in matches.columns:
                matches[col]=matches[col].replace(self.team_name_mapping)


                
        drop_cols=[c for c in ['umpire1', 'umpire2', 'city', 'player_of_match', 'method', 'super_over', 'match_type', 'target_overs']if c in matches.columns]
        matches.drop(columns=drop_cols,inplace=True)

        deliveries['over']=deliveries['over']+1

        for col in ['batting_team','bowling_team']:
            if col in deliveries.columns:
                deliveries[col]=deliveries[col].replace(self.team_name_mapping)

        deliveries=deliveries[deliveries['inning'].isin([1,2])].copy()

        valid_ids=set(matches['id'].unique())
        deliveries=deliveries[deliveries['match_id'].isin(valid_ids)].copy()

        match_cols=[
            'id', 'season', 'date', 'venue', 'team1', 'team2', 
            'toss_winner', 'toss_decision', 'winner', 'result', 'result_margin', 'target_runs'
        ]

        merged=deliveries.merge(matches[match_cols],left_on='match_id',right_on='id',how='inner')
        merged.drop(columns=['id'],inplace=True)


        key_cols=['match_id','winner','season','venue','batting_team']

        assert merged[key_cols].isnull().sum().sum()==0,"asserting error : nulls found in key columns"
        assert merged['over'].min()==1 and merged['over'].max()==20,'assertion error: over boundaries broken'
        assert sorted(merged['inning'].unique())==[1,2],'assertion error: innings index mismatch'
        assert set(merged['is_wicket'].unique())=={0,1},'assertion error: invalid wicket tracking keys'
        assert merged['match_id'].nunique()==len(matches),'assertion error: unique match mappings'


        merged['batting_team_wins']=(merged['batting_team']==merged['winner']).astype(int)

        second_innings_clean=merged[merged['inning']==2].copy()


        print(f' merging layer successful : clean matches {matches.shape},merged clean deliveries {merged.shape}')
        return matches, merged,second_innings_clean
    


class FeatureEngineeringModel1:
    def __init__(self,cfg: DictConfig)-> None:
        self.config=cfg

    def season_feature1(self,df1:pd.DataFrame)-> pd.DataFrame:
        df1=df1.copy()
        df1['date']=pd.to_datetime(df1['date'],errors='coerce')
        df1['season_yr']=df1['date'].dt.year.astype('Int64')
        df1=df1.sort_values(by=['date']).reset_index(drop=True)
        return df1
    
    def rolling_win_rate1(self,team:str,before_date:pd.Timestamp,df_all1:pd.DataFrame,window:int=10)->float:
        mask=(
            ((df_all1['team1']==team) | (df_all1['team2']==team))
            &(df_all1['date']< before_date)

        )

        past=df_all1[mask].sort_values('date').tail(window)
        if len(past)==0:
            return 0.5
        wins=(past['winner']==team).sum()

        return round(wins/len(past),4)
    
    def h2h_win_rate1(self,team1:str,team2:str,before_date:pd.Timestamp,df_all1:pd.DataFrame,window:int=20,)->float:
        mask=(
            (
                ((df_all1['team1']==team1) & (df_all1['team2']==team2))
                | ((df_all1['team1']==team2) & (df_all1['team2']==team1))

            )
            & (df_all1['date'] < before_date)
        )

        past=df_all1[mask].sort_values('date').tail(window)
        if len(past)==0:
            return 0.5
        
        wins=(past['winner']==team1).sum()

        return round(wins/len(past),4)
    
    def recent_form1(self,team:str,before_date:pd.Timestamp,df_all1:pd.DataFrame,window:int=5)->float:
        mask=(
            ((df_all1['team1']==team) | (df_all1['team2']==team))
            &(df_all1['date']< before_date)


        )

        past=df_all1[mask].sort_values('date').tail(window)

        if len(past)==0:
            wins=(past['winner']==team).sum()
            return 0.5
        wins=(past['winner']==team).sum()
        return round(wins/len(past),4)
    
    def venue_team_win_rate1(self,team:str,venue:str,before_date:pd.Timestamp,df_all1:pd.DataFrame,window:int=10)->float:
        mask=(
            ((df_all1['team1']==team) | (df_all1['team2']==team))
            & (df_all1['venue']==venue)
            &(df_all1['date']< before_date)
        )

        past=df_all1[mask].sort_values('date').tail(window)
        if len(past)<3:
            return 0.5
        wins=(past['winner']==team).sum()

        return round(wins/len(past),4)
    
    def venue_bat_win_rate1(self,venue:str,before_date:pd.Timestamp,df_all1:pd.DataFrame)->float:
        mask=(
            (df_all1['venue']==venue)
            &(df_all1['date']<before_date)
            &(df_all1['winner'].notna())
        )
        past=df_all1[mask]
        if len(past)==0:
            return 0.5
        
        bat_first_wins1=(
            ((past['toss_decision']=='bat') & (past['toss_winner']==past['winner']))
            | ((past['toss_decision']=='field') & (past['toss_winner'] !=past['winner']))

        ).sum()
        return round(bat_first_wins1/len(past),4)
    
    def venue_toss_win_rate1(self,venue: str, before_date:pd.Timestamp,df_all1:pd.DataFrame)->float:
        mask = (
            (df_all1["venue"] == venue)
            & (df_all1["date"] < before_date)
            & (df_all1["winner"].notna())
        )
        past = df_all1[mask]
        if len(past) == 0:
            return 0.5
        toss_winner_won = (past["toss_winner"] == past["winner"]).sum()
        return round(toss_winner_won / len(past), 4)
    def build_features(self, df_all1: pd.DataFrame) -> pd.DataFrame:
        df_all1 = df_all1.copy()
        df_all1["date"] = pd.to_datetime(df_all1["date"], errors="coerce")
        df_all1 = df_all1.sort_values("date").reset_index(drop=True)

        df_all1["season_yr"] = df_all1["date"].dt.year.astype("Int64")
        df_all1["team1_win_rate"] = df_all1.apply(
            lambda r: self.rolling_win_rate1(r["team1"], r["date"], df_all1, window=10), axis=1
        )
        df_all1["team2_win_rate"] = df_all1.apply(
            lambda r: self.rolling_win_rate1(r["team2"], r["date"], df_all1, window=10), axis=1
        )
        df_all1["team1_recent_form"] = df_all1.apply(
            lambda r: self.recent_form1(r["team1"], r["date"], df_all1, window=5), axis=1
        )
        df_all1["team2_recent_form"] = df_all1.apply(
            lambda r: self.recent_form1(r["team2"], r["date"], df_all1, window=5), axis=1
        )
        df_all1["h2h_win_rate"] = df_all1.apply(
            lambda r: self.h2h_win_rate1(r["team1"], r["team2"], r["date"], df_all1, window=20),
            axis=1,
        )
        df_all1["venue_team1_win_rate"] = df_all1.apply(
            lambda r: self.venue_team_win_rate1(
                r["team1"], r["venue"], r["date"], df_all1, window=10
            ),
            axis=1,
        )
        df_all1["venue_bat_first_win_rate"] = df_all1.apply(
            lambda r: self.venue_bat_win_rate1(r["venue"], r["date"], df_all1), axis=1
        )
        df_all1["venue_toss_win_rate"] = df_all1.apply(
            lambda r: self.venue_toss_win_rate1(r["venue"], r["date"], df_all1), axis=1
        )
        df_all1["win_rate_diff"] = df_all1["team1_win_rate"] - df_all1["team2_win_rate"]
        df_all1["recent_form_diff"] = df_all1["team1_recent_form"] - df_all1["team2_recent_form"]
        return df_all1
        




class FeatureEngineeringModel2:

    def __init__(self,cfg:DictConfig)->None:
        self.config=cfg
    
    def create_features2(self,df2:pd.DataFrame)-> pd.DataFrame:
        df2=df2.copy()
        df2['date']=pd.to_datetime(df2['date'],errors='coerce')
        df2['season_yr']=df2['date'].dt.year.astype('Int64')
        df2=df2.sort_values(by=['date']).reset_index(drop=True)
        return df2
    def pp_features2(self,df2:pd.DataFrame)->pd.DataFrame:
        self.first_inn=(df2[df2['inning']==1].copy().sort_values(['match_id','over','ball']).reset_index(drop=True))
        self.powerplay=self.first_inn[self.first_inn['over']<=6].copy()
        pp_agg=self.powerplay.groupby('match_id').agg(
            pp_runs        = ('total_runs',   'sum'),
            pp_wickets     = ('is_wicket',    'sum'),
            pp_fours       = ('batsman_runs', lambda x: (x == 4).sum()),
            pp_sixes       = ('batsman_runs', lambda x: (x == 6).sum()),
            pp_dot_balls   = ('batsman_runs', lambda x: (x == 0).sum()),
            pp_legal_balls = ('extras_type',  lambda x: x.isna().sum()),
            batting_team   = ('batting_team', 'first'),
            bowling_team   = ('bowling_team', 'first'),
            venue          = ('venue',        'first'),
            season_yr      = ('season_yr',    'first'),
            date           = ('date',         'first'),
        
        ).reset_index()
        pp_agg['pp_crr']            = (pp_agg['pp_runs'] / 6).round(3)
        pp_agg['pp_boundaries']     = pp_agg['pp_fours'] + pp_agg['pp_sixes']
        pp_agg['pp_dot_pct']        = (pp_agg['pp_dot_balls']  / pp_agg['pp_legal_balls'].clip(lower=1)).round(3)
        pp_agg['pp_boundary_pct']   = (pp_agg['pp_boundaries'] / pp_agg['pp_legal_balls'].clip(lower=1)).round(3)
        pp_agg['pp_runs_per_wicket']= (pp_agg['pp_runs'] / (pp_agg['pp_wickets'] + 1)).round(3)

        print(f'Powerplay feature matrix: {pp_agg.shape}')
        pp_agg[['pp_runs','pp_wickets','pp_crr','pp_dot_pct','pp_boundary_pct','pp_runs_per_wicket']].describe().round(2)
        self.pp_agg = pp_agg
        return self.pp_agg
    
    def model2_features2(self,df2:pd.DataFrame)->pd.DataFrame:
        if not hasattr(self, 'first_inn') or not hasattr(self, 'pp_agg'):
            self.pp_features2(df2)

        middle = self.first_inn[(self.first_inn['over'] >= 7) & (self.first_inn['over'] <= 15)].copy()

        mid_agg = middle.groupby('match_id').agg(
            mid_runs    = ('total_runs', 'sum'),
            mid_wickets = ('is_wicket',  'sum'),
            mid_balls   = ('extras_type', lambda x: x.isna().sum()),
        ).reset_index()

        mid_agg['mid_crr']        = (mid_agg['mid_runs'] / 9).round(3)   # 9 overs
        mid_agg['mid_run_per_wkt']= (mid_agg['mid_runs'] / (mid_agg['mid_wickets'] + 1)).round(3)

        # Merge with powerplay
        self.pp_agg = self.pp_agg.merge(
            mid_agg[['match_id','mid_runs','mid_wickets','mid_crr','mid_run_per_wkt']],
            on='match_id', how='left'
        )

        # Fill matches where middle overs data is missing (e.g. rain interruptions)
        for col in ['mid_runs','mid_wickets','mid_crr','mid_run_per_wkt']:
            self.pp_agg[col] = self.pp_agg[col].fillna(self.pp_agg[col].median())

        print(f'After middle-overs merge: {self.pp_agg.shape}')
        self.pp_agg[['mid_runs','mid_wickets','mid_crr','mid_run_per_wkt']].describe().round(2)
        return self.pp_agg
    
    def model2_2featurers(self,df2:pd.DataFrame)->pd.DataFrame:
        if not hasattr(self, 'pp_agg') or not hasattr(self, 'powerplay'):
            self.pp_features2(df2)

        train_season=[2019,2020,2021,2022,2023]

        self.match_totals=(df2.groupby(['match_id','venue','season_yr','date','batting_team','bowling_team'])['total_runs'].sum().reset_index().rename(columns={'total_runs':'match_total'}))

        train_match_totals=self.match_totals[self.match_totals['season_yr'].isin(train_season)]
        venue_avg_map=train_match_totals.groupby('venue')['match_total'].mean().round(2)
        self.global_avg=train_match_totals['match_total'].mean()
        self.pp_agg['venue_avg_score']=self.pp_agg['venue'].map(venue_avg_map).fillna(self.global_avg)

        self.mt_sorted=self.match_totals.sort_values('date')
        self.pp_sorted=(
            self.powerplay.groupby(['match_id','bowling_team','season_yr','date'])['total_runs'].sum().reset_index().assign(pp_economy_val=lambda d: d['total_runs']/6).sort_values('date')
        )

        return self.pp_agg , self.mt_sorted, self.pp_sorted
    
    def batting_team_season_avg(self,team,season_yr,before_date:pd.Timestamp)->float:
        past=self.mt_sorted.loc[
            (self.mt_sorted['batting_team']==team)&
            (self.mt_sorted['season_yr']==season_yr)&
            (self.mt_sorted['date']< before_date)
        ]
        return round(past['match_total'].mean(),2) if len(past)>=2 else self.global_avg
    
    def bowling_team_pp_econ(self,team,season_yr,before_date:pd.Timestamp)->float:
        past=self.pp_sorted.loc[
            (self.pp_sorted['bowling_team']==team)&
            (self.pp_sorted['season_yr']==season_yr)&
            (self.pp_sorted['date']< before_date)
        ]
        return round(past['pp_economy_val'].mean(),3) if len(past)>=2 else 8.5
    
    def batting_team_rolling(self,team,before_date:pd.Timestamp)->float:
        past=self.mt_sorted.loc[
            (self.mt_sorted['batting_team']==team) &
            (self.mt_sorted['date'] < before_date)
        ].tail(3)

        return round(past['match_total'].mean(),2) if len(past)>=1 else self.global_avg
    
    def h2h_avg(self,bat_team,bowl_team,before_date:pd.Timestamp)->float:
        past=self.mt_sorted.loc[
            (self.mt_sorted['batting_team']==bat_team)&
            (self.mt_sorted['bowling_team']==bowl_team)&
            (self.mt_sorted['date']< before_date)
        ]

        return round(past['match_total'].mean(),2) if len(past) >= 2 else self.global_avg
    

    
    def build_features2(self,df2:pd.DataFrame)->pd.DataFrame:
        if not hasattr(self, 'pp_agg') or not hasattr(self, 'mt_sorted') or not hasattr(self, 'pp_sorted'):
            self.model2_2featurers(df2)

        self.pp_agg['batting_team_season_avg']=self.pp_agg.apply(lambda r: self.batting_team_season_avg(r['batting_team'],r['season_yr'],r['date']),axis=1)

        self.pp_agg['bowling_team_pp_econ']=self.pp_agg.apply(lambda r: self.bowling_team_pp_econ(r['bowling_team'],r['season_yr'],r['date']),axis=1)

        self.pp_agg['batting_team_rolling']=self.pp_agg.apply(lambda r: self.batting_team_rolling(r['batting_team'],r['date']),axis=1)
        self.pp_agg['h2h_avg']=self.pp_agg.apply(lambda r: self.h2h_avg(r['batting_team'],r['bowling_team'],r['date']),axis=1)
     

        print(f'final feature matrix: {self.pp_agg.describe().round(2)}')
        return self.pp_agg
    
    def model2_df(self,df2:pd.DataFrame)->pd.DataFrame:
        if not hasattr(self, 'match_totals'):
            self.model2_2featurers(df2)
        if not hasattr(self, 'pp_agg') or 'h2h_avg' not in self.pp_agg.columns:
            self.build_features2(df2)

        y_df=self.match_totals[['match_id','match_total']].rename(columns={'match_total':'y'})
        model2_df=self.pp_agg.merge(y_df,on='match_id',how='left')

        model2_df['is_impact_era']=(model2_df['season_yr']>=2023).astype(int)
        model2_df['is_t20_boom_era']=(model2_df['season_yr']>=2021).astype(int)
        model2_df['is_pre2022']=(model2_df['season_yr']< 2022).astype(int)

        num_fill=model2_df.select_dtypes(include=np.number).columns
        model2_df[num_fill]=model2_df[num_fill].fillna(model2_df[num_fill].median())

        print(f'model_df shape : {model2_df.shape}')
        print(f'null check : {model2_df.isnull().sum().sum()}')
        print(f' season : {sorted(model2_df.season_yr.unique())}')
        print(f'\n target stats')
        print(model2_df['y'].describe().round(1))

        feature_cols=[
            'pp_runs', 'pp_wickets', 'pp_crr',
            'pp_fours', 'pp_sixes', 'pp_boundaries',
            'pp_dot_pct', 'pp_boundary_pct', 'pp_runs_per_wicket',
            # Middle overs (NEW)
            'mid_runs', 'mid_wickets', 'mid_crr', 'mid_run_per_wkt',
            # Date-gated contextual
            'venue_avg_score', 'batting_team_season_avg',
            'bowling_team_pp_econ',
            'batting_team_rolling',
            'h2h_avg',                # NEW
            # Categorical
            'batting_team', 'bowling_team', 'venue',
            # Era flags
            'season_yr', 'is_impact_era', 'is_t20_boom_era', 'is_pre2022',

        ]

        return model2_df, feature_cols
    


class FeatureEngineeringModel3:

    def __init__(self,cfg:DictConfig)->None:
        self.config=cfg

    def create_features3(self,df3:pd.DataFrame)->pd.DataFrame:
        df3=df3.copy()
        df3['date']=pd.to_datetime(df3['date'],errors='coerce')
        df3['season_yr']=df3['date'].dt.year.astype('Int64')
        df3=df3.sort_values(by=['date']).reset_index(drop=True)
        return df3
    
    def match_features3(self,df3:pd.DataFrame)->pd.DataFrame:
        match_totals=df3.groupby('match_id')['total_runs'].sum().reset_index()
        match_totals.rename(columns={'total_runs': 'final_chase_score'},inplace=True)

        match_df=df3[['match_id','date','season_yr','venue','batting_team','bowling_team','target_runs','batting_team_wins']].drop_duplicates()
        chase_df=pd.merge(match_df,match_totals,on='match_id',how='inner')

        pp_df=df3[df3['over']<=6]
        pp_stats=pp_df.groupby('match_id').agg(
            pp_runs=('total_runs','sum'),
            pp_wickets=('is_wicket','sum')
        ).reset_index()

        pp_df['is_boundary']=pp_df['batsman_runs'].apply(lambda x: 1 if x in [4,6] else 0)
        pp_boundaries = pp_df.groupby('match_id')['is_boundary'].sum().reset_index().rename(columns={'is_boundary':'pp_boundaries'})

        
        chase_df = pd.merge(chase_df, pp_stats, on='match_id', how='inner')
        chase_df = pd.merge(chase_df, pp_boundaries, on='match_id', how='inner')

       
        chase_df.dropna(subset=['pp_runs'], inplace=True)

        
        chase_df['pp_crr'] = round((chase_df['pp_runs']/6), 2)
        chase_df['runs_needed_after_pp'] = chase_df['target_runs'] - chase_df['pp_runs']
        chase_df['rrr_after_pp'] = round((chase_df['runs_needed_after_pp']/14), 2)
        chase_df['pressure_index'] = chase_df['rrr_after_pp'] * (chase_df['pp_wickets']+1)

        chase_df[['target_runs','pp_runs','pp_wickets','rrr_after_pp','pressure_index']].head()
        
        chase_df['is_impact_era']=chase_df['season_yr'].apply(lambda x: 1 if x >= 2023 else 0)
        chase_df['is_t20_boom_era']=chase_df['season_yr'].apply(lambda x: 1 if x>=2018 else 0)
        self.chase_df = chase_df
        return self.chase_df


    def get_venue_chase_avg(self,venue,match_df,df):
        past_matches=df[(df['venue']==venue)& (df['date']<match_df)]
        if len(past_matches)<3:
            return df[df['date']<match_df]['final_chase_score'].mean()
        return round(past_matches['final_chase_score'].mean(),2)
    
    def get_team_recent_form(self,team,match_date,df):
        past_matches=df[(df['batting_team']==team) & (df['date']<match_date)].sort_values(by='date').tail(5)
        if len(past_matches)<2:
            return df[df['date']< match_date]['final_chase_score'].mean()
        return round(past_matches['final_chase_score'].mean(),2)
    
    def build_features3(self,df3:pd.DataFrame)->pd.DataFrame:
        if not hasattr(self, 'chase_df'):
            self.match_features3(df3)

        self.chase_df['venue_chase_avg']=self.chase_df.apply(lambda x: self.get_venue_chase_avg(x['venue'],x['date'],self.chase_df),axis=1)
        self.chase_df['team_recent_chase_form']=self.chase_df.apply(lambda x: self.get_team_recent_form(x['batting_team'],x['date'],self.chase_df),axis=1)
            
        global_mean=self.chase_df['final_chase_score'].mean()

        self.chase_df['venue_chase_avg'] = self.chase_df['venue_chase_avg'].fillna(global_mean)
        self.chase_df['team_recent_chase_form'] = self.chase_df['team_recent_chase_form'].fillna(global_mean)

        self.chase_df=self.chase_df.sort_values('date').reset_index(drop=True)
        train_season=[2019,2020,2021,2022,2023]
        test_season=[2024]
        features_cols=['target_runs', 'is_impact_era', 'is_t20_boom_era',
    # Powerplay Dynamics
            'pp_runs', 'pp_wickets', 'pp_boundaries',
            # Chase Pressure Dynamics
            'pp_crr', 'rrr_after_pp', 'pressure_index',
            # Historical Context
            'venue_chase_avg', 'team_recent_chase_form',
            # Categorical
            'batting_team', 'bowling_team', 'venue',
            # Split feature
            'season_yr']
        return self.chase_df,features_cols,train_season,test_season
    
