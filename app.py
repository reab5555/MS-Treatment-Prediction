import gradio as gr
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("ms_treatment_model.joblib")

# Define the prediction function
def predict_treatment_response_and_treatment(age, ms_type, edss, lesion, wb):
    # Encode ms_type
    ms_type_map = {"RR": 0, "PPMS": 1, "APMS": 2}
    ms_type_encoded = ms_type_map[ms_type]
    
    # Create a feature array
    features = np.array([[age, ms_type_encoded, edss, lesion, wb]])
    
    # Predict the response and probability
    response_prediction = model.predict(features)
    response_probability = model.predict_proba(features)
    response_confidence = int((response_probability[0][1] if response_prediction[0] == 1 else response_probability[0][0]) * 100)
    
    if response_prediction[0] == 0:
        # Return only the response prediction and confidence if the response is "No"
        return (
            f"No ({response_confidence}%)",
            "N/A"
        )
    
    # Predict the treatment
    treatment_prediction = model.predict(features)  # Adjust if necessary
    treatment_probability = model.predict_proba(features)  # Adjust if necessary
    treatment_confidence = int(np.max(treatment_probability[0]) * 100)
    
    # Map treatment prediction back to categorical label
    treatment_map = {
        0: "Tecfidera", 
        1: "Tysabri", 
        2: "Avonex", 
        3: "Lemtrada", 
        4: "Vumerity", 
        5: "Mavenclad", 
        6: "Ocrevus"
    }
    treatment_label = treatment_map[treatment_prediction[0]]
    
    # Return the predictions and confidence levels
    return (
        f"Yes ({response_confidence}%)",
        f"{treatment_label} ({treatment_confidence}%)"
    )

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_treatment_response_and_treatment,
    inputs=[
        gr.Slider(minimum=17, maximum=80, label="Age"),
        gr.Dropdown(choices=["RR", "PPMS", "APMS"], label="MS Type"),
        gr.Slider(minimum=1, maximum=8, step=0.5, label="EDSS"),
        gr.Slider(minimum=0, maximum=80, label="Lesion"),
        gr.Slider(minimum=1000, maximum=2000, label="WB")
    ],
    outputs=[
        gr.Textbox(label="Treatment Response and Confidence"),
        gr.Textbox(label="Predicted Treatment and Confidence")
    ],
    title="Multiple Sclerosis Treatment Predictor (Demo)",
    description=(
        "Enter the following details to predict the treatment response probability and suggested treatment.\n\n"
        "- **Age**: Numeric, between 17 and 80.\n"
        "- **MS Type**: Type of Multiple Sclerosis (RR, PPMS, APMS).\n"
        "- **EDSS**: Expanded Disability Status Scale, values between 1 and 8.\n"
        "- **Lesion**: Number of lesions observed in MRI, values between 0 and 80.\n"
        "- **WB**: Whole brain volume, values between 1000 and 2000.\n\n"
        "Note: The results and predictions must be taken with caution as the model prediction abilities and the data it trained on are limited."
    )
)

# Launch the Gradio app
iface.launch()
