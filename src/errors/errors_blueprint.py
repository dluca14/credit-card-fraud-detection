'''
Application error handlers.
'''

from flask import Blueprint, jsonify

errors_blueprint = Blueprint('errors', __name__)


@errors_blueprint.app_errorhandler(400)
def handle_malformed_request_error(error):
    return build_response_status_code(400, False, 'MalformedRequestError',
                                      'Description is missing from your request.')


@errors_blueprint.app_errorhandler(401)
def handle_authorization_error(error):
    return build_response_status_code(401, False, 'AuthorizationError',
                                      'Request missing proper authorization.')


@errors_blueprint.app_errorhandler(404)
def handle_missing_service_error(error):
    return build_response_status_code(404, False, 'MissingServiceError',
                                      'The service you are trying to access is missing.')


@errors_blueprint.app_errorhandler(405)
def handle_method_not_allowed_error(error):
    return build_response_status_code(405, False, 'MethodNotAllowedError',
                                      'The method is not allowed for the requested URL.')


@errors_blueprint.app_errorhandler(500)
def handle_internal_server_error(error):
    return build_response_status_code(500, False, 'InternalServerError',
                                      'An internal error occurred. Please retry.')


def build_response_status_code(status_code, success, error_type, error_message):
    response = {
        'success': success,
        'error': {
            'type': error_type,
            'message': error_message
        }
    }
    return jsonify(response), status_code
