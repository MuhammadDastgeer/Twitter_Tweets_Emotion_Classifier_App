import streamlit as st
import pickle
import joblib

st.set_page_config(page_title="Twitter Tweets Emotion Classifier", layout="centered")

st.title("Twitter Tweets Emotion Classifier")
st.write("Enter text and the app will tell you the predicted emotion (Sadness, Anger, Love, Surprise, Fear, Joy).")

# Always load from local file (no upload)
model_path = "text_classifier_with_vectorizer.pkl"

def load_model(path):
    try:
        with open(path,"rb") as f:
            m = pickle.load(f)
        return m, "pickle"
    except Exception:
        try:
            m = joblib.load(path)
            return m, "joblib"
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return None, None

model, mtype = load_model(model_path)
if model is None:
    st.stop()

st.write("Loaded model type:", mtype)

# Label mapping for emotions
label_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

text = st.text_area("Input text", height=200)
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter text to classify.")
    else:
        if mtype in ("pickle","joblib"):
            try:
                if hasattr(model, "predict"):
                    y = model.predict([text])
                    pred_label = label_map.get(int(y[0]), str(y[0]))
                    st.success(f"Predicted emotion: {pred_label}")
                elif isinstance(model, dict) and "model" in model:
                    clf = model["model"]
                    vect = model.get("vectorizer")
                    X = vect.transform([text]) if vect is not None else [text]
                    y = clf.predict(X)
                    pred_label = label_map.get(int(y[0]), str(y[0]))
                    st.success(f"Predicted emotion: {pred_label}")
                else:
                    st.write("Loaded object:", type(model))
                    st.write("Try saving a sklearn-compatible estimator or a dict with keys 'model' and optional 'vectorizer'.")
            except Exception as e:
                st.error(f"Inference failed: {e}")
        else:
            st.error("Unknown model type.")
