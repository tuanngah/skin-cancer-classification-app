import gradio as gr
import os
import sys
import datetime
import pandas as pd
import traceback

project_root = os.path.abspath(os.getcwd())
sys.path.append(project_root)

try:
    from src.app.predictor import Predictor
    from src.app.triage_manager import (
        add_patient_to_queue,
        get_doctor_dashboard,
        mark_patient_completed,
        get_completed_patients_dashboard,
        INITIAL_QUEUE_STATE,
        DISEASE_DESCRIPTIONS
    )
except ImportError as e:
    print("ERROR: Could not import modules from 'src/app'.")
    print(f"Original error: {e}")
    sys.exit(1)

print("Initializing predictor (loading ONNX model)...")
try:
    predictor = Predictor(use_gpu=False)
    print("Predictor is ready.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load ONNX model. {e}")
    predictor = None

def main_check_in_flow(image_input, current_queue_state):
    if predictor is None:
        return "Error: Model not loaded successfully.", "", "", current_queue_state
    if image_input is None:
        return "Error: Please upload an image.", "", "", current_queue_state
    try:
        patient_id = f"PT_{datetime.datetime.now().strftime('%H%M%S%f')[:-3]}" # Changed prefix to PT
        predictions = predictor.predict(image_input)
        predicted_label = max(predictions, key=predictions.get)
        prediction_prob = predictions[predicted_label]
        # Translate descriptions or keep them if intended for Vietnamese users
        description = DISEASE_DESCRIPTIONS.get(predicted_label, "No description available.")
        prediction_text = f"{predicted_label.upper()} ({prediction_prob:.1%})"
        triage_message, updated_queue_state = add_patient_to_queue(
            patient_id, predicted_label, current_queue_state
        )
        return triage_message, prediction_text, description, updated_queue_state
    except Exception as e:
        print(f"Error in main_check_in_flow: {e}")
        traceback.print_exc()
        return f"An error occurred: {e}", "", "", current_queue_state

def refresh_dashboards(current_queue_state):
    try:
        waiting_df = get_doctor_dashboard(current_queue_state)
        completed_df = get_completed_patients_dashboard(current_queue_state)
        # Ensure correct column names expected by Gradio DataFrame headers
        waiting_df.columns = ["Status/QueueNo", "Patient ID", "Prediction", "Check-in Time"]
        completed_df.columns = ["Status/QueueNo", "Patient ID", "Prediction", "Check-in Time", "Completion Time"]
        return waiting_df, completed_df
    except Exception as e:
        print(f"Error in refresh_dashboards: {e}")
        empty_waiting = pd.DataFrame(columns=["Status/QueueNo", "Patient ID", "Prediction", "Check-in Time"])
        empty_completed = pd.DataFrame(columns=["Status/QueueNo", "Patient ID", "Prediction", "Check-in Time", "Completion Time"])
        return empty_waiting, empty_completed

def handle_select_patient(evt: gr.SelectData, current_queue_state: dict):
    selected_patient_id = None
    selected_display_text = "No patient selected."
    if evt.index is not None and len(evt.index) > 0:
        try:
            # Need to get the currently displayed DataFrame to map index to ID
            waiting_df, _ = refresh_dashboards(current_queue_state)
            selected_row_index = evt.index[0]
            if selected_row_index < len(waiting_df):
                selected_patient_id = waiting_df.iloc[selected_row_index]["Patient ID"]
                selected_display_text = f"Selected: {selected_patient_id}"
                print(f"Selected Patient ID: {selected_patient_id}")
            else:
                 selected_display_text = "Selection index out of bounds."
                 print(selected_display_text)
        except Exception as e:
            print(f"Error getting selected patient ID: {e}")
            selected_display_text = f"Selection Error: {e}"
    return selected_patient_id, selected_display_text

def handle_mark_completed_button(selected_patient_id: str, current_queue_state: dict):
    print(f"\n--- Mark Complete Button Clicked ---")
    print(f"Selected ID from state: {selected_patient_id}")
    display_message = "Updated. Select next patient." # Default success message

    if selected_patient_id is None:
        print("No patient selected to mark as complete.")
        display_message = "Error: Please select a patient first!"
        # Return unchanged state and the error message
        return current_queue_state, None, display_message

    try:
        updated_queue_state = mark_patient_completed(selected_patient_id, current_queue_state)
        print(f"Updated Queue State (after mark): {updated_queue_state}")
        print("--- Mark Complete Button Finished ---\n")
        # Return updated state, reset selected ID, and success message
        return updated_queue_state, None, display_message
    except Exception as e:
        print(f"Error in handle_mark_completed_button: {e}")
        traceback.print_exc()
        # Keep selected ID if error occurred and show error message
        display_message = f"Processing Error: {e}"
        return current_queue_state, selected_patient_id, display_message

with gr.Blocks(theme=gr.themes.Soft(), title="Dermatology Triage System") as interface:
    gr.Markdown("# Intelligent Dermatology Triage System 🩺")
    gr.Markdown("Upload a skin lesion image for prediction and priority classification.")

    queue_state = gr.State(value=INITIAL_QUEUE_STATE.copy())
    selected_patient_id_state = gr.State(value=None)

    with gr.Tabs():
        with gr.TabItem("Patient Check-in"):
            with gr.Row():
                with gr.Column(scale=1):
                    patient_image_input = gr.Image(type="pil", label="1. Upload Skin Lesion Image", height=300)
                    submit_button = gr.Button("Submit for Classification", variant="primary")
                    gr.Markdown("---")
                    gr.Markdown("**DISCLAIMER:** Results are for reference only and do not replace a doctor's diagnosis.")
                with gr.Column(scale=1):
                    checkin_output_message = gr.Textbox(label="Triage Result", placeholder="Triage status will appear here...", lines=2, interactive=False)
                    predicted_label_output = gr.Textbox(label="Preliminary Diagnosis", placeholder="Predicted disease name...", interactive=False)
                    description_output = gr.Textbox(label="Description", placeholder="Short description of the predicted disease...", lines=4, interactive=False)

        with gr.TabItem("Doctor Dashboard"):
            gr.Markdown("#### Waiting Patient List (Priority First)")
            doctor_dashboard_output = gr.DataFrame(
                headers=["Status/QueueNo", "Patient ID", "Prediction", "Check-in Time"],
                datatype=["str", "str", "str", "str"],
                label="Patient Queue",
                interactive=True,
            )
            selected_patient_display = gr.Textbox(label="Selected Patient (ID)", interactive=False)
            mark_complete_button = gr.Button("Mark as Seen", variant="primary")

            gr.Markdown("---")
            gr.Markdown("#### Patients Seen History")
            completed_dashboard_output = gr.DataFrame(
                headers=["Status/QueueNo", "Patient ID", "Prediction", "Check-in Time", "Completion Time"],
                datatype=["str", "str", "str", "str", "str"],
                label="Seen History",
                interactive=False,
            )

    submit_button.click(
        fn=main_check_in_flow,
        inputs=[patient_image_input, queue_state],
        outputs=[checkin_output_message, predicted_label_output, description_output, queue_state]
    )

    interface.load(
        fn=refresh_dashboards,
        inputs=[queue_state],
        outputs=[doctor_dashboard_output, completed_dashboard_output]
    )

    doctor_dashboard_output.select(
        fn=handle_select_patient,
        inputs=[queue_state],
        outputs=[selected_patient_id_state, selected_patient_display]
    )

    mark_complete_button.click(
        fn=handle_mark_completed_button,
        inputs=[selected_patient_id_state, queue_state],
        outputs=[queue_state, selected_patient_id_state, selected_patient_display]
    )

    queue_state.change(
        fn=refresh_dashboards,
        inputs=[queue_state],
        outputs=[doctor_dashboard_output, completed_dashboard_output]
    )

if __name__ == "__main__":
    interface.launch(share=False, debug=True)