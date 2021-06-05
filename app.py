from flask import Flask, render_template, redirect, request, jsonify
from flask.templating import render_template_string

# Import sqlalchemy libraries to connect with the Postgres database.
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

import os
import time

#################################################
# Database Setup
#################################################
# PotgreSQL connection requirements
# from config import userid
# from config import password
userid = os.environ['userid']
password = os.environ['password']
rds_connection_string = f"{userid}:{password}@awspostgres.ctkgxnaawxx6.ap-southeast-2.rds.amazonaws.com/AWSPostgres"
engine = create_engine(f'postgresql://{rds_connection_string}')

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)

# Save reference to the table
Failures = Base.classes.hvac_failures
Speed_values = Base.classes.speed

#################################################
# Flask Setup
#################################################
app = Flask(__name__) 


#################################################
# Flask Routes
#################################################

# Set route - displays landing page
@app.route("/")
def index():
    return render_template("dashboard.html")

# Set route - displays a dashboard page
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Set route - displays a blank page
@app.route("/blank")
def blank():
    return render_template("blank.html")

# Set route - displays a basic table
@app.route("/basic-table")
def table():
    return render_template("basic-table.html")

# Set route - displays a site map
@app.route("/site-map")
def sitemap():
    return render_template("map-google.html")


#################################################
# Flask Routes - API's
#################################################

# A route to return all of the available failures
@app.route("/api/v1/failures")
def failures():
    
    # Create our session (link) from Python to the DB
    session = Session(engine)

    """Return a list of all HVAC failures"""
    # Create our session (link) from Python to the DB
    session = Session(engine)

    # Check for Parameters
    if 'id' in request.args:
        results = session.query(Failures.primary_element, Failures.start_time, Failures.end_time, Failures.duration, Failures.difference, Failures.comments, Failures.failure).limit(5).all()
    else:
        results = session.query(Failures.primary_element, Failures.start_time, Failures.end_time, Failures.duration, Failures.difference, Failures.comments, Failures.failure).all()

    session.close()

    # Create a dictionary from the row data and append to a list of all_failures
    all_failures = []
    for primary_element, start_time, end_time, duration, difference, comments, failure in results:
        failure_dict = {}
        failure_dict["primary_element"] = primary_element
        failure_dict["start_time"] = start_time
        failure_dict["end_time"] = end_time
        failure_dict["duration"] = duration
        failure_dict["difference"] = float(difference)
        failure_dict["comments"] = comments
        failure_dict["failure"] = failure
        all_failures.append(failure_dict)

    return jsonify(all_failures)

# A route to return the latest speed values for a fin-fan
@app.route("/api/v1/speed")
def speed_function():
    
    # Create our session (link) from Python to the DB
    session = Session(engine)

    """Return a list of speed values"""
    # Create our session (link) from Python to the DB
    session = Session(engine)

    # Check for Parameters
    if 'id' in request.args:
        results = session.query(Speed_values.equipment_id, Speed_values.speed).filter_by(Speed_values.equipment_id==id).all()
    else:
        results = session.query(Speed_values.equipment_id, Speed_values.speed).all()

    session.close()

    # Create a dictionary from the row data and append to a list of all_failures
    speed_vals = []
    for equipment_id, speed in results:
        speed_dict = {}
        speed_dict["equipment_id"] = equipment_id
        speed_dict["speed"] = speed
        speed_vals.append(speed_dict)

    return jsonify(speed_vals)



if __name__ == "__main__":
    app.run(debug=True)
