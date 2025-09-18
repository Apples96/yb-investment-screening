import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from src.model.model import MNISTModel
from src.db.database import get_prediction_stats, get_prediction_history, log_prediction, update_true_label
import uuid

# Initialize the model
@st.cache_resource
def load_model():
    model = MNISTModel()
    model.load_state_dict(torch.load('src/model/saved/mnist_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_digit(canvas_image):
    if canvas_image is None:
        st.error("Please draw a digit first!")
        return None, None

    try:
        # Convert the numpy image (with RGBA channels) to grayscale PIL Image
        img = Image.fromarray(canvas_image.astype('uint8'), mode="RGBA").convert('L')
        # Resize to MNIST dimensions (28x28)
        img = img.resize((28, 28))
        # Convert to numpy array and then a tensor
        img_array = np.array(img)
        # MNIST has white digits on black background, so invert colors if needed
        # If your drawing is black on white, you need this inversion
        img_array = 255 - img_array
        # Show the processed image for debugging
        st.image(img_array, caption="Processed Image (After Inversion)", width=100)
       
        # Normalize using the same values from training
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = (img_tensor - 0.1307) / 0.3081

        # Make prediction
        model = load_model()
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item() * 100

        return prediction, confidence

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def main():
    st.title("MNIST Digit Recognizer")

    # Initialize session state
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    if 'current_confidence' not in st.session_state:
        st.session_state.current_confidence = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    def reset_canvas():
        st.session_state.canvas_key = str(uuid.uuid4())
        st.session_state.prediction_made = False
        st.session_state.current_prediction = None
        st.session_state.current_confidence = None
        st.session_state.current_image = None
        st.session_state.submitted = False

    # Display statistics
    stats = get_prediction_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions", stats['total_predictions'])
    with col2:
        st.metric("Model Accuracy", f"{stats['accuracy']:.1f}%")

    st.write("Draw a digit (0-9) in the canvas below and click 'Predict'")

    # Create a drawable canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # White background
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key
    )

    # Store current image in session state
    if canvas_result.image_data is not None:
        st.session_state.current_image = canvas_result.image_data.copy()

    # Prediction section
    col1, col2 = st.columns([1, 3])
    
    with col1:
        predict_button = st.button("Predict")
    
    with col2:
        clear_button = st.button("Clear Canvas", on_click=reset_canvas)

    if predict_button and not st.session_state.prediction_made:
        prediction, confidence = predict_digit(st.session_state.current_image)
        if prediction is not None:
            # Store predictions in session state
            st.session_state.prediction_made = True
            st.session_state.current_prediction = prediction
            st.session_state.current_confidence = confidence
            
            # Log prediction without true label (only once)
            prediction_id = log_prediction(prediction, confidence, image_data=st.session_state.current_image)
            st.session_state.prediction_id = prediction_id
            print("Inserted prediction with ID:", prediction_id)

            st.rerun()

    # Show prediction results and true label input
    if st.session_state.prediction_made and not st.session_state.submitted:
        st.write(f"Prediction: {st.session_state.current_prediction}")
        st.write(f"Confidence: {st.session_state.current_confidence:.2f}%")
        
        # Allow user to input true label
        true_label = st.number_input("Enter the correct digit (optional):", 
                                   min_value=0, max_value=9, step=1)
        
        if st.button("Submit True Label"):
            st.write("Submitting true label...")
            # Only update with true label, don't log another prediction
            if not st.session_state.submitted:
                update_success = update_true_label(st.session_state.prediction_id, true_label)
                if update_success:
                    st.success("True label submitted and updated!")
                else:
                    st.error("Error updating true label.")
                st.session_state.submitted = True
                reset_canvas()
                st.rerun()

    with st.expander("Prediction History"):
        history = get_prediction_history()
        if history:
            # Optionally, convert to a DataFrame for nicer display
            import pandas as pd
            df_history = pd.DataFrame(history, columns=["ID", "Predicted", "Confidence", "True Label", "Timestamp"])
            st.dataframe(df_history)
        else:
            st.write("No predictions logged yet.")

if __name__ == "__main__":
    main()