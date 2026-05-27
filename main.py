import os,sys
import pandas as pd
from config.config import ConfigLoader
from src.pipeline.train_pipeline import IPLTrainingPipeline
from src.pipeline.predict_pipeline import IPLInferencePipeline

def run_system_lifecycle():
    cfg=ConfigLoader().load()
    trainer=IPLTrainingPipeline(cfg=cfg)
    trainer.run()

    # test inference flow
    print(f'\n executing live mock streaming prediction')
    live_matches_mock=pd.read_csv(cfg.paths.raw_matches).tail(2)
    live_deliveries_mock=pd.read_csv(cfg.paths.raw_deliveries)

    live_deliveries_mock=live_deliveries_mock[live_deliveries_mock['match_id'].isin(live_matches_mock['id'])]

    inference_engine=IPLInferencePipeline(cfg=cfg)
    predictions=inference_engine.predict_live_stream(live_matches_mock,live_deliveries_mock)

    print(f'\n predictions on the live stream')

    print()
    print("Predicted Matches Winners Classes  :", predictions["winning_model_predictions"])
    print("Predicted Team 1 Win Probabilities :", predictions["team1_win_probabilities"])
    print("Projected Innings 1 Base Score Max :", predictions["projected_innings1_runs"])
    print("Projected Innings 2 Final Chases   :", predictions["projected_chase_scores"])


if __name__ == "__main__":
    run_system_lifecycle()
