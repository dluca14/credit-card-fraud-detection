from pymongo import MongoClient
import datetime
import hashlib

mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["gpreda"] # set your own database, on your local MongoDB env
mongo_collection = mongo_db["credit_card"]


def insert_transaction(transaction_data):

    datetime_now = datetime.datetime.utcnow().isoformat()
    transaction_id = hashlib.md5(f"{datetime_now}{transaction_data['Amount']}".encode("utf-8")).hexdigest()

    mongo_dict = {"_id": f"/transactions/{transaction_id}/data.json",
                  "file": {"transaction_id": transaction_id,
                           "transaction_data": transaction_data,
                           },
                  "content_type": "application/json",
                  "created_timestamp": datetime.datetime.utcnow().isoformat()
              }
    try:
        mongo_collection.insert_one(mongo_dict)
    except Exception as ex:
        print("Exception MongoDB:", ex)
        pass

    return transaction_id


def write_inference(transaction_id, inference_data):
    """
    Write the inferemce in MongoDB
    @param transaction_id - transaction id
    @param inference_data - inference data
    """
    mongo_dict = {"_id": f"/transactions/{transaction_id}/inference.json",
                  "inference": inference_data,
                  "content_type": "application/json",
                  "created_timestamp": datetime.datetime.utcnow().isoformat(),
                  "updated_timestamp": datetime.datetime.utcnow().isoformat()
              }
    try:
        mongo_collection.insert_one(mongo_dict)
    except Exception as ex:
        print("Exception MongoDB:", ex)
        pass


def write_one_manual(transaction_id, operator_decision):
    """
    Write a new manual overwritten inference
    """
    mongo_dict = {"_id": f"/transactions/{transaction_id}/manual.json",
                  "manual": operator_decision,
                  "content_type": "application/json",
                  "created_timestamp": datetime.datetime.utcnow().isoformat(),
                  "updated_timestamp": datetime.datetime.utcnow().isoformat()
                  }
    try:
        mongo_collection.insert_one(mongo_dict)
        return {"status": "OK", "manual": operator_decision, "transaction_id": transaction_id}
    except Exception as ex:
        print("Exception MongoDB:", ex)


def replace_write_manual(result, transaction_id, operator_decision):
    """
    Replace a manual overwritten inference with a new overwritten
    """
    if result:
        created_timestamp = result["created_timestamp"]
        mongo_dict = {"_id": f"/transactions/{transaction_id}/manual.json",
                      "manual": operator_decision,
                      "content_type": "application/json",
                      "created_timestamp": created_timestamp,
                      "updated_timestamp": datetime.datetime.utcnow().isoformat()
                      }
        try:
            mongo_collection.replace_one(result, mongo_dict)
            return {"status": "OK", "manual": operator_decision, "transaction_id": transaction_id}
        except Exception as ex:
            print("Exception MongoDB:", ex)
            pass


def write_manual(transaction_id, operator_decision):
    """
    Overwrite the inference value in MongoDB
    @param transaction_id - transaction id
    @param operator_decision - manual overwrite of initial inference
    """
    filter_expression = {"_id": {"$regex": f"/transactions/{transaction_id}/manual.json"}}
    try:
        mongo_query = filter_expression
        results = mongo_collection.find(mongo_query)
        for result in results:
            if result:
                return replace_write_manual(result, transaction_id, operator_decision)
            else:
                return write_one_manual(transaction_id, operator_decision)
        else:
            return write_one_manual(transaction_id, operator_decision)
    except Exception as ex:
        print("Exception MongoDB:", ex)
        pass

    return {"status": "KO"}


def register_model(model_data, model_parameters, training_parameters, validation_data):
    """
    Write the model description to MongoDB
    @param model_data - data used to create the model
    @param model_parameters - model parameters
    @param training_parameters - training parameters
    @param validation_data - validation data
    """
    datetime_now = datetime.datetime.utcnow().isoformat()
    model_id = hashlib.md5(f"{datetime_now}{model_data['train_rows']}".encode("utf-8")).hexdigest()

    mongo_dict = {"_id": f"/models/{model_id}/data.json",
                  "model_id": model_id,
                  "model_data": model_data,
                  "model_parameters": model_parameters,
                  "training_parameters": training_parameters,
                  "validation_data": validation_data,
                  "content_type": "application/json",
                  "created_timestamp": datetime.datetime.utcnow().isoformat(),
                  "updated_timestamp": datetime.datetime.utcnow().isoformat()
              }
    try:
        mongo_collection.insert_one(mongo_dict)
        return model_id
    except Exception as ex:
        print("Exception MongoDB:", ex)
        return None
        pass


def get_models():
    filter_expression = {"_id": {"$regex": "models"}}
    try:
        validation_scores = []
        mongo_query = filter_expression
        results = mongo_collection.find(mongo_query)
        if results:
            return results
    except Exception as ex:
        print("Exception MongoDB:", ex)
        pass


def get_models_validation_score():
    filter_expression = {"_id": {"$regex": "models"}}
    try:
        validation_scores = []
        mongo_query = filter_expression
        results = mongo_collection.find(mongo_query)
        if results:
            for result in results:
                print(result["validation_data"]["validation_score_all"])
                validation_scores.append({"model_id": result["model_id"],
                                          "validation_score": result["validation_data"]["validation_score_all"]})
            return validation_scores
    except Exception as ex:
        print("Exception MongoDB:", ex)
        pass


def get_transaction_by_id(transaction_id):
    """
    Get transaction by id
    @param transaction_id - transaction id
    @returns the transaction information for a certain transaction

    """
    try:
        mongo_query = {"_id": f"/transactions/{transaction_id}/data.json"}
        results = mongo_collection.find(mongo_query)
        if results:
            for result in results:
                return result
    except Exception as ex:
        print("Exception MongoDB:", ex)
        pass


def get_inference(transaction_id):
    """
     Returns the run_model for a certain transaction, identified by transaction_id
    @param transaction_id - transaction id
    @return run_model data
    """
    try:
        mongo_query = {"_id": f"/transactions/{transaction_id}/run_model.json"}
        results = mongo_collection.find(mongo_query)
        if results:
            for result in results:
                return result
    except Exception as ex:
        print("Exception MongoDB:", ex)
        pass


def get_transactions():
    filter_expression = {"_id": {"$regex": "transactions"}}
    try:
        mongo_query = filter_expression
        results = mongo_collection.find(mongo_query)
        if results:
            return results
    except Exception as ex:
        print("Exception MongoDB:", ex)
        pass
