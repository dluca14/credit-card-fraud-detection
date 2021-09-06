import logging
import os

import joblib
import pandas as pd
from flask import abort

from build_model.create_model import run_all, get_predictors_target

logger = logging.getLogger('is_fraud')


class TransactionModel:
    transaction_model = None
    path = None

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "model")
        self.transaction_model = None

        try:
            logger.info(self.path)
        except Exception as ex:
            logger.error(f"Error: {type(ex)} {ex}")
            abort(500)

    @staticmethod
    def get_columns_predictors():
        predictors, target = get_predictors_target()
        return predictors

    def load_resource(self):
        logger.info(self.path)
        model = joblib.load(os.path.join(self.path, "model_light_gbm.pkl"))
        return model

    @staticmethod
    def retrain():
        logger.info("retrain model")
        try:
            model_id, validation_score = run_all(run_optimization=False, run_test=False)
            return {"retrain": "OK", "model_id": model_id, "validation_score": validation_score}
        except Exception as ex:
            return {"retrain": "KO"}

    def predict(self, parameters):
        try:
            logger.info(parameters)
            test_df = pd.io.json.json_normalize(parameters)
            test_df.columns = self.get_columns_predictors()
            if not self.transaction_model:
                self.transaction_model = self.load_resource()
            predicted = self.transaction_model.predict(test_df.values)[0]
            logger.info(f"Predicted value: {predicted}")
            return "True" if predicted else "False"
        except Exception as ex:
            logger.error(f"Error: {type(ex)} {ex}")
            abort(500)
