import pandas as pd
import datetime
import json

URGENT_GROUP = {'mel', 'bcc', 'akiec'}
NORMAL_GROUP = {'nv', 'bkl', 'df', 'vasc'}

DISEASE_DESCRIPTIONS = {
    'mel': "Melanoma (Dangerous): A serious form of skin cancer, requires immediate check-up by a doctor.",
    'bcc': "Basal Cell Carcinoma (Dangerous): A common type of skin cancer, often slow-growing, needs early examination.",
    'akiec': "Actinic Keratosis (Pre-cancerous/Dangerous): Pre-cancerous skin lesion caused by sun exposure, can develop into cancer.",
    'nv': "Nevus (Benign): Common moles, mostly benign, but changes should be monitored.",
    'bkl': "Benign Keratosis (Benign): Common benign skin lesion in older adults, not cancerous.",
    'df': "Dermatofibroma (Benign): Small, benign skin nodule, usually requires no treatment.",
    'vasc': "Vascular Lesion (Benign): Issues related to skin blood vessels, usually benign (e.g., cherry angioma)."
}

INITIAL_QUEUE_STATE = {
    "urgent_queue": [],
    "normal_queue": [],
    "completed_queue": [],
    "next_stt": 101
}

def add_patient_to_queue(patient_id: str, prediction_label: str, queue_state: dict) -> tuple[str, dict]:
    updated_state = queue_state.copy()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if prediction_label in URGENT_GROUP:
        updated_state["urgent_queue"].append({
            'patient_id': patient_id, 'prediction': prediction_label, 'timestamp': timestamp
        })
        message = "PRIORITY CONSULTATION REQUIRED \nPlease wait for further instructions."
    elif prediction_label in NORMAL_GROUP:
        stt = updated_state["next_stt"]
        updated_state["normal_queue"].append({
            'patient_id': patient_id, 'prediction': prediction_label, 'stt': stt, 'timestamp': timestamp
        })
        updated_state["next_stt"] += 1
        message = f"Your queue number is: {stt}\nPlease wait for your turn."
    else:
        message = "Lesion type could not be determined. Please contact staff."

    return message, updated_state

def get_doctor_dashboard(queue_state: dict) -> pd.DataFrame:
    urgent_patients = []
    for patient in queue_state["urgent_queue"]:
        urgent_patients.append({
            "Status/QueueNo": "PRIORITY",
            "Patient ID": patient['patient_id'],
            "Prediction": patient['prediction'].upper(),
            "Check-in Time": patient['timestamp']
        })

    normal_patients = []
    for patient in queue_state["normal_queue"]:
        normal_patients.append({
            "Status/QueueNo": f"No. {patient['stt']}",
            "Patient ID": patient['patient_id'],
            "Prediction": patient['prediction'].upper(),
            "Check-in Time": patient['timestamp']
        })

    all_patients = urgent_patients + sorted(normal_patients, key=lambda x: int(x["Status/QueueNo"].split(" ")[1]))

    if not all_patients:
        return pd.DataFrame(columns=["Status/QueueNo", "Patient ID", "Prediction", "Check-in Time"])

    dashboard_df = pd.DataFrame(all_patients)
    return dashboard_df

def mark_patient_completed(patient_id: str, queue_state: dict) -> dict:
    updated_state = queue_state.copy()
    updated_state["urgent_queue"] = list(updated_state["urgent_queue"])
    updated_state["normal_queue"] = list(updated_state["normal_queue"])
    updated_state["completed_queue"] = list(updated_state["completed_queue"])

    patient_found = None
    queue_to_modify = None
    index_to_remove = -1

    for i, patient in enumerate(updated_state["urgent_queue"]):
        if patient['patient_id'] == patient_id:
            patient_found = patient
            queue_to_modify = updated_state["urgent_queue"]
            index_to_remove = i
            break

    if not patient_found:
        for i, patient in enumerate(updated_state["normal_queue"]):
            if patient['patient_id'] == patient_id:
                patient_found = patient
                queue_to_modify = updated_state["normal_queue"]
                index_to_remove = i
                break

    if patient_found and queue_to_modify is not None and index_to_remove != -1:
        del queue_to_modify[index_to_remove]
        patient_found['completed_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_state["completed_queue"].append(patient_found)
        print(f"Moved patient {patient_id} to completed list.")
    elif not patient_found:
        print(f"Patient with ID: {patient_id} not found in waiting queues.")

    return updated_state

def get_completed_patients_dashboard(queue_state: dict) -> pd.DataFrame:
    completed_patients_data = []
    for patient in list(queue_state["completed_queue"]):
        status = "PRIORITY" if patient.get('stt') is None else f"No. {patient.get('stt')}"
        completed_patients_data.append({
            "Status/QueueNo": status,
            "Patient ID": patient.get('patient_id', 'N/A'),
            "Prediction": patient.get('prediction', 'N/A').upper(),
            "Check-in Time": patient.get('timestamp', 'N/A'),
            "Completion Time": patient.get('completed_time', 'N/A')
        })

    if not completed_patients_data:
        return pd.DataFrame(columns=["Status/QueueNo", "Patient ID", "Prediction", "Check-in Time", "Completion Time"])

    dashboard_df = pd.DataFrame(completed_patients_data)
    try:
        dashboard_df["Completion Time"] = pd.to_datetime(dashboard_df["Completion Time"])
        dashboard_df = dashboard_df.sort_values(by="Completion Time", ascending=False)
        dashboard_df["Completion Time"] = dashboard_df["Completion Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error sorting completed dashboard: {e}")

    return dashboard_df

if __name__ == '__main__':
    current_state = INITIAL_QUEUE_STATE.copy()
    _, current_state = add_patient_to_queue("PT001", "nv", current_state)
    print(f"PT001 ('nv'): {_}")
    _, current_state = add_patient_to_queue("PT002", "mel", current_state)
    print(f"PT002 ('mel'): {_}")
    _, current_state = add_patient_to_queue("PT003", "bcc", current_state)
    print(f"PT003 ('bcc'): {_}")
    _, current_state = add_patient_to_queue("PT004", "bkl", current_state)
    print(f"PT004 ('bkl'): {_}")

    print("\nCurrent queue state:")
    print(json.dumps(current_state, indent=2))

    print("\nDoctor Dashboard (Initial):")
    dashboard_waiting = get_doctor_dashboard(current_state)
    print(dashboard_waiting.to_string(index=False))

    print("\n--- Marking PT002 and PT001 as completed ---")
    current_state = mark_patient_completed("PT002", current_state)
    current_state = mark_patient_completed("PT001", current_state)

    print("\nDoctor Dashboard (After completing PT002, PT001):")
    dashboard_waiting_updated = get_doctor_dashboard(current_state)
    print(dashboard_waiting_updated.to_string(index=False))

    print("\nCompleted Patients List:")
    dashboard_completed = get_completed_patients_dashboard(current_state)
    print(dashboard_completed.to_string(index=False))