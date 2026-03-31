import pandas as pd
import joblib
import re

def rate_into_categorial ():

    rate = patient_data.loc[0]["Blood Pressure"]
    pressure = re.split (r'/', rate)
    systolic = int (pressure[0])
    diastolic = int (pressure[1])
    
    if systolic < 120 and diastolic < 80:
        blood_pressure_category = 'normal'
    elif systolic <= 129 and systolic >= 120 and diastolic < 80:
        blood_pressure_category = 'elevated'
    elif (systolic <= 139 and systolic >= 130) or (diastolic <= 89 and diastolic >= 80):
        blood_pressure_category = 'stage 1 hypertension'
    elif systolic >= 140 or diastolic >= 90:
        blood_pressure_category = 'stage 2 hypertension'
    elif systolic >= 180 or diastolic >= 120: 
        blood_pressure_category = 'severe hypertension'
    else:
        blood_pressure_category = 'measurement error'
    
    patient_data.at[0, "Blood Pressure"] = blood_pressure_category

def one_hot_encoding ():
    for pair in One_Hot_Encoding:
        feature = pair[0]
        unique_category = pair[1]

        for category in unique_category[0]:
            patient_data[category] = 1 if patient_data.loc[0][feature] == category else 0
        del patient_data[feature]

def ordered_target_encoding ():
    for pair in Ordered_Target_Encoding:
        feature = pair[0]
        option_count, overall_mean_i, alpha, n_i, classifier = pair[1]
        current_category = patient_data.loc[0][feature]

        for target in classifier:
            name_new_feature = feature + '_' + target
            catboost_encode = (option_count[current_category] + overall_mean_i  * alpha) / (n_i + alpha)    
            patient_data[name_new_feature] = catboost_encode
        del patient_data[feature]

def get_patient_input():
    print("\n--- New Patient Inference ---")
    
    print("\n1. Choose Gender:")
    print("   [1] Male")
    print("   [2] Female")
    gender_choice = int(input("Enter choice (1 or 2): "))
    if gender_choice == 1:
        gender = "Male" 
    elif gender_choice == 0:
        gender = "Female"
    else:
        raise ValueError ("Error: Please enter number in the list !")
    
    age = int(input("\n2. Enter Age: "))
    if age < 0:
        raise ValueError ("Error: Age must be an positive integer !")
    
    print ("\n3. Choose Occupation:")
    occupations = ['Doctor', 'Nurse', 'Engineer', 'Software Engineer', 'Teacher']
    for i, occ in enumerate(occupations):
        print(f"   [{i + 1}] {occ}")
    index = int(input("Enter choice: ")) - 1
    if index < 0 or index >= len (occupations):
        raise ValueError ("Error: Please enter number in the list !")
    occ_choice = occupations[index]

    sleep_dur = float(input("\n3. Enter Sleep Duration (e.g., 7.5): "))
    if sleep_dur < 0:
        raise ValueError ("Error: Sleep Duration must be an positive float and hour unit !")

    quality = int(input("\n4. Quality of Sleep (1-10): "))
    if quality < 0 or quality > 10:
        raise ValueError ("Error: Please choose quality number in correct range !")

    activity = int(input("\n5. Physical Activity Level (Minutes/Day): "))
    if activity < 0 or activity > 1440:
        raise ValueError ("Error: Activity Duration must be positive integer and no more than 1440 mins a day !")

    stress = int(input("\n6. Stress Level (1-10): "))
    if  stress < 0 or  stress > 10:
        raise ValueError ("Error: Please choose Stress Level in correct range !")


    print("\n7. Choose BMI Category:")
    print("   [1] Normal")
    print("   [2] Normal Weight")
    print("   [3] Overweight")
    print("   [4] Obese")
    bmi_choice = int(input("Enter choice (1, 2, 3 or 4): "))
    bmi_map = {1: "Normal", 2: "Normal Weight", 3: "Overweight", 4: "Obese"} 
    if bmi_choice < 0 or bmi_choice > 4:
        raise ValueError ("Error: Please enter number in the list !")
    bmi = bmi_map.get(bmi_choice, 0)
    
    print("\n8. Blood Pressure:")
    systolic = int(input("   Enter Systolic (top number, e.g., 120): "))
    diastolic = int(input("   Enter Diastolic (bottom number, e.g., 80): "))
    if systolic < 0 or diastolic < 0:
        raise ValueError ("Error: Systolic and Diastolic must be an positive integer !")

    heart_rate = int(input("\n9. Enter Heart Rate (BPM): "))
    if heart_rate < 0:
        raise ValueError ("Error: Heart Rate must be an positive integer !")

    steps = int(input("\n10. Enter Daily Steps: "))
    if steps < 0:
        raise ValueError ("Error: Steps must be an positive integer !")

    final_features = {
        "Gender": gender, 
        "Age": age,
        "Occupation": occ_choice,
        "Sleep Duration": sleep_dur,
        "Qaulity of Sleep": quality, 
        "Physical Activity Level": activity, 
        "Stress Level": stress,
        "BMI Category": bmi,
        "Blood Pressure": str (systolic) + '/' + str (diastolic),
        "Heart Rate": heart_rate,
        "Daily Steps": steps
    }
    
    return final_features

def collect_patient_data ():
    try:
        patient_data = pd.DataFrame ([get_patient_input ()])
        
        print("\n--- Processing Data ---")
        print ("Model is processing...")
        return patient_data

    except ValueError as e:
        if "invalid literal" in str (e):
            print ("Error: Please do not leave it blank !")
        else:
            print(e)
        return None
    
def loading_file ():
    try:
        Gradient_Boosting_Classifier = joblib.load ("Gradient Boosting Classifier.pkl")
        One_Hot_Encoding = joblib.load ("One-Hot Encoding.pkl")
        Ordered_Target_Encoding = joblib.load ("Ordered Target Encoding.pkl")
        Model_Features = joblib.load ("Model Features.pkl")
        return [Gradient_Boosting_Classifier, One_Hot_Encoding, Ordered_Target_Encoding, Model_Features]

    except FileNotFoundError:
        print ("Error: Model files not found. Check file paths again !")
        return None

def make_prediction (patient_data):
    patient_data = patient_data.reindex (columns = Model_Features, fill_value = 0)
    prediction = Gradient_Boosting_Classifier.predict (patient_data)
    print (f"\n Result: {prediction[0]}")

patient_data = collect_patient_data ()
Gradient_Boosting_Classifier, One_Hot_Encoding, Ordered_Target_Encoding, Model_Features = loading_file ()

rate_into_categorial ()
one_hot_encoding ()
ordered_target_encoding ()
make_prediction (patient_data)





# cd "/Users/chikhoado/Desktop/PROJECTS/Sleeping Disorder"
# python3 -m venv .venv
# source .venv/bin/activate
# pip install pandas scikit-learn numpy seaborn
# python "/Users/chikhoado/Desktop/PROJECTS/Sleeping Disorder/Inference.py"