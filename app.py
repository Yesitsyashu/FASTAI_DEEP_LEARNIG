import gradio as gr
from fastai.vision.all import *

# Load the exported FastAI model
learn = load_learner('fruit_classifier.pkl')

# Define a prediction function
def classify_image(img):
    pred, pred_idx, probs = learn.predict(img)
    return {str(pred): float(probs[pred_idx])}

# Set up the Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=2),
    title="Fruit Freshness Classifier",
    description="Upload an image of a fruit to check if it is fresh or rotten."
)

# Launch the app
iface.launch()
