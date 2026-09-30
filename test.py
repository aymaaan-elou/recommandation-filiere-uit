import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title("Test d'installation ✅")
st.write("Si vous voyez ce message, tout est prêt !")
st.write(f"pandas : {pd.__version__}")
st.write(f"numpy : {np.__version__}")
st.write(f"sklearn : importé avec succès")