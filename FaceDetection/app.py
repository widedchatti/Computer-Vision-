import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from predection import predict, image_processing
import matplotlib.pyplot as plt



# Create a div with the "app-container" class
st.markdown('<div class="app-container">', unsafe_allow_html=True)

# Title of the web app
st.title("Hello! Let's detect some faces :) ")

# Upload an image
uploaded_image = st.file_uploader("Upload an image ", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Check if the "Detect Face" button is clicked
    if st.button("Detect Face"):
        # Read the uploaded image
        image = Image.open(uploaded_image)

        # Call the prediction and apply it to the uploaded processed image
        with st.spinner("Detecting faces..."):
            predicted_labels, indices = predict(image_processing(image))

        fig, ax = plt.subplots()

        # Ensure that the aspect ratio matches the displayed image
        ax.set_aspect('equal')
        image_processed = image_processing(image)
        ax.imshow(np.array(image_processed), cmap="purple")
        ax.axis('off')
        Ni, Nj = (62, 47)
        indices = np.array(indices)

        for i, j in indices[predicted_labels == 1]:
            ax.add_patch(plt.Circle((j, i), Nj, Ni, edgecolor='green', alpha=0.3, lw=2, facecolor='none'))

        st.pyplot(fig)

# Close the div
st.markdown('</div>', unsafe_allow_html=True)
