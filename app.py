import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from train_model import Generator

# ———— Load pretrained generator ————
latent_dim = 100
class_dim = 10

model = Generator(latent_dim, class_dim)
model.load_state_dict(torch.load('generator.pth', map_location='cpu'))
model.eval()

# ———— Streamlit UI ————
st.title("Handwritten Digit Generator")

digit = st.slider('Select Digit', 0, 9, 0)

if st.button('Generate 5 Images'):
    noise  = torch.randn(5, latent_dim)
    labels = torch.full((5,), digit, dtype=torch.long)

    with torch.no_grad():
        gen_imgs = model(noise, labels)  # (5,1,28,28)
    imgs = (gen_imgs + 1) / 2.0          # normalize to [0,1]

    for i in range(5):
        # Drop the channel, get a (28,28) float array in [0,1]
        gray_float = imgs[i, 0].cpu().numpy()
        
        # Convert to 8-bit grayscale
        gray_uint8 = (gray_float * 255).clip(0, 255).astype(np.uint8)
        
        # Display
        st.image(
            gray_uint8,
            caption=f"Generated digit: {digit}",
            width=56,
            channels="GRAY"
        )
