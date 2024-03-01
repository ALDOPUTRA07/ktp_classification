import streamlit as st

from streamlit_app.utils.ui import dataset_description, header_description, result
from streamlit_app.utils.utils import process

# fastapi endpoint
url = 'http://127.0.0.1:8000'
endpoint = '/predict'


# UI
header_description()

dataset_description()

st.divider()

image = st.file_uploader('insert image', type=['png', 'jpg'])  # image upload widget

if image is not None:
    button_classification = st.button('Classify image', type="primary")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Image")
        st.image(image)

    if button_classification:
        segments = process(image, url + endpoint)

        with col2:
            results = result(segments)
