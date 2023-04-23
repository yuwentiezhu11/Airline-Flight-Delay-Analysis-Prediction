from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# demo function


def model(airline, airportfrom, airportto, weekdays, time, length):
    data = {'Airline': [airline], 'AirportFrom': [airportfrom], 'AirportTo': [
        airportto], 'weekdays': [weekdays], 'Time': [time], 'Length': [length]}
    dataFrame = pd.DataFrame(data)
    model = joblib.load('./model.pkl')
    result = model.predict(dataFrame)
    return result[0]

# demo fetch data from flask backend - Members API Route


# @app.route("/demo")
# def demo():
#     return {"members": ["Member1", "Member2", "Member3"]}

# POST Route to handle data from React frontend


@app.route("/predict", methods=["GET", "POST"])
def predict():

    # if request.method=="post":
    print("receive data from React:")
    # must convert ImmutableMultiDict to dictionary
    sub_dict = request.form.to_dict()
    print(sub_dict)
    parameter1 = sub_dict["parameter1"]
    parameter2 = sub_dict["parameter2"]
    parameter3 = sub_dict["parameter3"]
    parameter4 = sub_dict["parameter4"]
    parameter5 = sub_dict["parameter5"]
    parameter6 = sub_dict["parameter6"]
    print(type(parameter1))
    if parameter1 == "-1":
        return jsonify({
            "status": "Prediction Failed",
            "result": "Please select \'Airline\'"
        })
    elif parameter2 == "-1":
        return jsonify({
            "status": "Prediction Failed",
            "result": "Please select \'AirportFrom\'"
        })
    elif parameter3 == "-1":
        return jsonify({
            "status": "Prediction Failed",
            "result": "Please select \'AirportTo\'"
        })
    elif parameter4 == "-1":
        return jsonify({
            "status": "Prediction Failed",
            "result": "Please select \'Day of Week\'"
        })
    elif parameter5 == "":
        return jsonify({
            "status": "Prediction Failed",
            "result": "Please input \'Flight Time\'"
        })
    elif parameter6 == "":
        return jsonify({
            "status": "Prediction Failed",
            "result": "Please input \'Flight Length\'"
        })
    print("parameter1: "+parameter1)
    print("parameter2: "+parameter2)
    print("parameter3: "+parameter3)
    print("parameter4: "+parameter4)
    print("parameter5: "+parameter5)
    print("parameter6: "+parameter6)
    result = model(int(parameter1), int(parameter2), int(
        parameter3), int(parameter4), int(parameter5), int(parameter6))
    # type: numpy.int64
    # numpy.int64 -> int
    result = result.item()
    if result == 0:
        result = 'The flight will be on time'
    else:
        result = 'The flight will be delayed'
    return jsonify({
        "status": "Prediction Successful",
        "result": result
    })


if __name__ == "__main__":
    # macos avoid ports 5000 and 7000
    app.run(port=8000, debug=True)
