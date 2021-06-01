from flask import Flask, render_template, redirect, request, jsonify
from flask.templating import render_template_string

# Import sqlalchemy libraries to connect with the Postgres database.
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

import pprint
import time

#################################################
# Database Setup
#################################################
# PotgreSQL connection requirements
from config import userid
from config import password
rds_connection_string = f"{userid}:{password}@awspostgres.ctkgxnaawxx6.ap-southeast-2.rds.amazonaws.com/AWSPostgres"
engine = create_engine(f'postgresql://{rds_connection_string}')

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)

# Save reference to the table
Patients = Base.classes.patients
# Station = Base.classes.station

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

# Set route - displays a blank page
@app.route("/blank")
def blank():
    return render_template("blank.html")


# A route to return all of the available ..........
@app.route('/api/v1/resources/patients/all', methods=['GET'])
def api_patients_all():
#     docs = []
#     # read records from Mongo, remove the _id field, convert to JSON and allow for CORS
#     for doc in mongo.db.country_codes.find():
#         doc.pop('_id') 
#         docs.append(doc)
    session = Session(engine)
    first_row = session.query(Patients).first()
    
    # response = jsonify(docs)
#     response.headers.add('Access-Control-Allow-Origin', '*')
    return jsonify(json_list = first_row.__dict__)
    


if __name__ == "__main__":
    app.run(debug=True)
