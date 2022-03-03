from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import jsonschema
from jsonschema import validate
import numpy

coefficients = [15474.641917266252, 40434.828554699525, 41070.89890345359, 40333.321557077004, 18848.58522068252, 16208.389893117124, 22243.339231275633, 24105.90611545494, 4164.772926590216, 1545.66504043396, 7713.376000750904, 4867.980804174962, 3660.479537770913, 39414.38519263655, -11723.44287074937, 7510.605980978513, 19310.262715758203, 4948.1351321853645]
min_max_scaler = MinMaxScaler()
min_max_scaler.scale_ = [0.25, 0.25, 1,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
    1.,   1.,   1., 1.,   1. ,   1. ,  1.  ]
min_max_scaler.min_ = [-0.25, -0.25,  0.,    0.,    0.,    0.,    0.,    0.,    0.,
    0.,    0.,    0.,  0.,    0.,    0,    0. ,   0.,    0.  ]

script_schema_1 = {
    "type": "object",
    "properties": {
        "performance": {"type": "number"},
        "seniority": {"type": "number"},
        "gender": {"type": "string"},
        "education": {"type": "string"},
        "job": {"type": "string"}     
    },
}

def validate_input(array):
    for x in array:
        if(x < 0):
            return False
    return True

def h(params, sample):
    """
    This evaluates a generic linear function h(x) with current parameters.  h stands for hypothesis
	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		sample (lst) a list containing the values of a sample 
	Returns:
		Evaluation of h(x)
	"""
    acum = 0
    if any(isinstance(i, numpy.ndarray or list) for i in sample):
        sample = sample[0]
    for i in range(len(params)):
        acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
    return acum

def validateJson(jsonData, schema):
    try:
        validate(instance=jsonData, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        return False
    return True

def get_prediction(input):
    """This function receives the data from the requests, parses it as a dictionary and gets the prediction
    Args:
        input (json) The request body
    
    Returns:
        The prediction for the input value
    """
    try:
        data = json.loads(input)
    except ValueError as err:
        return -1
    if not (validateJson(data, script_schema_1)):
        return -1
    data = json.loads(input)
    data = data.items()
    input_data = [data[0], data[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if data[2] == 'Female':
        input_data[2] = 1
    else:
        input_data[3] = 1
    
    if data[3] == 'High School':
        input_data[4] = 1
    elif data[3] == 'College':
        input_data[5] = 1
    elif data[3] == 'Masters':
        input_data[6] = 1
    elif data[3] == 'PhD':
        input_data[7] = 1
        
    if data[4] == 'Data Scientist':
        input_data[8] = 1
    elif data[4] == 'Driver':
        input_data[9] = 1
    elif data[4] == 'Financial Analyst':
        input_data[10] = 1
    elif data[4] == 'Graphic Designer':
        input_data[11] = 1
    elif data[4] == 'IT':
        input_data[12] = 1
    elif data[4] == 'Manager':
        input_data[13] = 1
    elif data[4] == 'Marketing Associate':
        input_data[14] = 1
    elif data[4] == 'Sales Associate':
        input_data[15] = 1
    elif data[4] == 'Software Engineer':
        input_data[16] = 1
    elif data[4] == 'Warehouse Associate':
        input_data[17] = 1
    an_array = np.array(list(input_data.values()))
    if not validate_input(an_array):
        return -1
    norm = min_max_scaler.transform([an_array])
    pred = h(coefficients, norm)
    return pred

def main(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    data = request.data
    norm = get_prediction(data)
    output = {
        'success': False
    }
    if norm != -1:
        output = {
            'success': True,
            'prediction': norm
        }
    return json.dumps(output)
