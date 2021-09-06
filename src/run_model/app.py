
import logging.config
import sys
sys.path.append("..")

from flask import Flask, jsonify, request, abort

from run_model.transaction_classifier import TransactionModel
from errors.errors_blueprint import errors_blueprint
from storage.mongo_db import insert_transaction, write_inference, get_models_validation_score, write_manual

app = Flask(__name__)
app.register_blueprint(errors_blueprint)


logger = logging.getLogger('is_fraud')


@app.route('/status', methods=['GET'])
def status():
    logger.info("Request received")
    get_models_validation_score()
    return jsonify({'status': "OK"})


@app.route('/is_fraud', methods=['POST'])
def inference():
    try:
        predictors = request.json['predictors']
        logger.info(predictors)
        transaction_id = insert_transaction(predictors)
        transaction_model = TransactionModel()
        category_prediction = transaction_model.predict(predictors)
        logger.info("Response sent")
        inference_data = {"is_fraud": category_prediction}
        write_inference(transaction_id, inference_data)
        return jsonify(inference_data)
    except Exception as ex:
        logger.error(f"Error: {type(ex)} {ex}")
        abort(500)


@app.route('/manual', methods=['POST'])
def manual():
    transaction_id = request.json["transaction_id"]
    operator_decision = {"is_fraud": request.json["is_fraud"]}
    transaction_data = write_manual(transaction_id, operator_decision)
    return jsonify(transaction_data)
    return jsonify


@app.route('/retrain', methods=['GET'])
def retrain():
    try:
        transaction_model = TransactionModel()
        train_status = transaction_model.retrain()
        return jsonify(train_status)

    except Exception as ex:
        logger.error(f"Error: {type(ex)} {ex}")
        abort(500)


if __name__ == '__main__':
    app.run()
