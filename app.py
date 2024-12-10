import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib  # or pickle
import uvicorn
import warnings, os, csv
from datetime import datetime
from utils.logger import logger

# Disable warnings
warnings.filterwarnings("ignore")

# Setup FastAPI app
app = FastAPI()

logger.info(f"Starting FastAPI on port")
model = joblib.load("model.pkl")

# Ensure the 'output' directory exists for saving results
if not os.path.exists('output'):
    os.makedirs('output')


# Filepath for results CSV
results_file = 'output/results.csv'

# Check if the CSV file exists and write headers if it's a new file
if not os.path.exists(results_file):
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp", "Age", "SleepDuration", "PhysicalActivityLevel", "StressLevel", 
            "HeartRate", "DailySteps", "BP", "Gender", "BMICategory", "SleepDisorderStatus", 
            "Prediction"
        ])




class UserData(BaseModel):
    Age: int
    SleepDuration: int
    PhysicalActivityLevel: int
    StressLevel: int
    HeartRate: int
    DailySteps: int
    BP: str  # Blood pressure as systolic/diastolic (string)
    gender: str  # 'male' or 'female'
    BMICategory: int  # 0 for Normal, 1 for Obesity
    SleepDisorderStatus: int  # 0 for No sleep disorder, 1 for Sleep disorder

# Preprocessing function to handle transformations (Blood Pressure, Gender, BMICategory, SleepDisorderStatus)
def preprocess_input(data):
    """
    Process the incoming data:
    - Convert gender to a numeric value (0 for male, 1 for female).
    - Convert BP string to systolic and diastolic pressure.
    - Return processed features ready for prediction.
    """
    # Convert gender to numeric
    if data.gender.lower() == "male":
        Male = 1
        Female = 0
    elif data.gender.lower() == "female":
        Male = 0
        Female = 1
    else:
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'.")
    
    # Handle BMI: 0 for Normal, 1 for Obesity
    if data.BMICategory == 1:
        BMI_overweight = 1
        BMI_normal = 0
    elif data.BMICategory == 0:
        BMI_overweight = 0
        BMI_normal = 1
    else:
        raise HTTPException(status_code=400, detail="BMICategory must be 0 (Normal) or 1 (Obesity).")

    # Handle BP: "systolic/diastolic" -> systolic and diastolic pressure
    try:
        systolic_pressure, diastolic_pressure = map(int, data.BP.split('/'))
    except ValueError:
        raise HTTPException(status_code=400, detail="BP should be in the format 'systolic/diastolic' (e.g., '120/80')")

    # Extract values for the other fields
    age = data.Age
    sleep_duration = data.SleepDuration
    physical_activity_level = data.PhysicalActivityLevel
    stress_level = data.StressLevel
    heart_rate = data.HeartRate
    daily_steps = data.DailySteps
    sleep_disorder_status = data.SleepDisorderStatus

    return [
        age, sleep_duration, physical_activity_level, stress_level, 
        heart_rate, daily_steps, systolic_pressure, diastolic_pressure, 
        Female, Male, BMI_overweight, BMI_normal, sleep_disorder_status
    ]


@app.post("/predict")
def predict(user_data: UserData):
    try:
        logger.info(f"read user data")
        # Print the user data to see what we're receiving
        print("Received data:", user_data.dict())

        # Preprocess the input data
        logger.info(f"Preprocess the data")
        processed_data = preprocess_input(user_data)

        prediction_input = [processed_data]
        print(prediction_input)

        logger.info(f"prediction data")
        prediction = model.predict(prediction_input)
        print("Predicted sleep quality:", prediction)

        response = {
            "status": "200 OK",
            "prediction": prediction.tolist()
        }


        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, user_data.Age, user_data.SleepDuration, user_data.PhysicalActivityLevel, 
                user_data.StressLevel, user_data.HeartRate, user_data.DailySteps, user_data.BP, 
                user_data.gender, user_data.BMICategory, user_data.SleepDisorderStatus, 
                prediction[0]])

        if isinstance(prediction[0], int):  # Classification result
            response["prediction_label"] = "Good Sleep" if prediction[0] == 1 else "Poor Sleep"


        elif isinstance(prediction[0], float):  # Regression result
            response["predicted_sleep_quality_score"] = round(prediction[0], 2)

        return response


    except KeyError as ke:
        error_msg = f"KeyError: The key '{str(ke)}' is missing in the input data."
        print(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        error_msg = str(e)  # Capture the error message
        print(f"Unexpected error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sleep Quality Prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
