import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Système de Recommandation de Filière - UIT",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# DÉFINITION DES 18 FILIÈRES DE LA FACULTÉ DES SCIENCES DE KÉNITRA
# ============================================================
filieres = {
    'Intelligence Artificielle': {'maths': 0.9, 'physique': 0.5, 'chimie': 0.1, 'svt': 0.1, 'geologie': 0.0, 'francais': 0.5, 'anglais': 0.8},
    'Informatique Fondamentale': {'maths': 0.9, 'physique': 0.4, 'chimie': 0.1, 'svt': 0.1, 'geologie': 0.0, 'francais': 0.5, 'anglais': 0.7},
    'Mathématiques et Applications': {'maths': 0.95, 'physique': 0.5, 'chimie': 0.1, 'svt': 0.1, 'geologie': 0.0, 'francais': 0.5, 'anglais': 0.5},
    'Mathématiques et Informatique pour la Décision': {'maths': 0.9, 'physique': 0.5, 'chimie': 0.1, 'svt': 0.1, 'geologie': 0.0, 'francais': 0.6, 'anglais': 0.6},
    "Ingénierie Électronique et Traitement de l'Information": {'maths': 0.6, 'physique': 0.9, 'chimie': 0.3, 'svt': 0.0, 'geologie': 0.0, 'francais': 0.4, 'anglais': 0.6},
    'Physique Fondamentale et Applications': {'maths': 0.7, 'physique': 0.95, 'chimie': 0.4, 'svt': 0.1, 'geologie': 0.0, 'francais': 0.5, 'anglais': 0.5},
    'Ingénieries des Énergies Renouvelables': {'maths': 0.6, 'physique': 0.9, 'chimie': 0.5, 'svt': 0.1, 'geologie': 0.1, 'francais': 0.4, 'anglais': 0.6},
    'Radioprotection et Dosimétrie Médicale': {'maths': 0.5, 'physique': 0.9, 'chimie': 0.6, 'svt': 0.4, 'geologie': 0.0, 'francais': 0.5, 'anglais': 0.6},
    'Génie des Matériaux': {'maths': 0.5, 'physique': 0.7, 'chimie': 0.9, 'svt': 0.2, 'geologie': 0.1, 'francais': 0.4, 'anglais': 0.5},
    'Génie des Procédés Industriels': {'maths': 0.5, 'physique': 0.6, 'chimie': 0.9, 'svt': 0.2, 'geologie': 0.1, 'francais': 0.4, 'anglais': 0.5},
    'Chimie et Applications': {'maths': 0.3, 'physique': 0.5, 'chimie': 0.95, 'svt': 0.3, 'geologie': 0.1, 'francais': 0.5, 'anglais': 0.4},
    'Analyse et Qualité': {'maths': 0.4, 'physique': 0.4, 'chimie': 0.9, 'svt': 0.4, 'geologie': 0.2, 'francais': 0.6, 'anglais': 0.5},
    'Sciences Biomédicales et Santé': {'maths': 0.3, 'physique': 0.4, 'chimie': 0.7, 'svt': 0.95, 'geologie': 0.2, 'francais': 0.6, 'anglais': 0.6},
    'Biotechnologie et Productions Végétales': {'maths': 0.2, 'physique': 0.3, 'chimie': 0.6, 'svt': 0.9, 'geologie': 0.3, 'francais': 0.5, 'anglais': 0.4},
    'Biosciences': {'maths': 0.2, 'physique': 0.3, 'chimie': 0.5, 'svt': 0.95, 'geologie': 0.3, 'francais': 0.5, 'anglais': 0.4},
    "Sciences et Techniques de l'Eau en Agriculture": {'maths': 0.3, 'physique': 0.5, 'chimie': 0.6, 'svt': 0.8, 'geologie': 0.5, 'francais': 0.5, 'anglais': 0.4},
    "Génie Civil et Géosciences de l'Environnement": {'maths': 0.7, 'physique': 0.7, 'chimie': 0.3, 'svt': 0.4, 'geologie': 0.9, 'francais': 0.5, 'anglais': 0.5},
    'Géoresources': {'maths': 0.4, 'physique': 0.5, 'chimie': 0.4, 'svt': 0.3, 'geologie': 0.95, 'francais': 0.5, 'anglais': 0.5}
}

df_filieres = pd.DataFrame(filieres).T
filieres_cols = ['maths', 'physique', 'chimie', 'svt', 'geologie', 'francais', 'anglais']

# ============================================================
# FONCTION DE JUSTIFICATION
# ============================================================
def justifier_recommandation(notes, filiere):
    competences = df_filieres.loc[filiere]
    justifications = []

    if notes['note_maths'] >= 14 and competences['maths'] >= 0.7:
        justifications.append("vos bonnes notes en mathématiques")
    if notes['note_physique'] >= 14 and competences['physique'] >= 0.7:
        justifications.append("vos bonnes notes en physique")
    if notes['note_chimie'] >= 14 and competences['chimie'] >= 0.7:
        justifications.append("vos bonnes notes en chimie")
    if notes['note_svt'] >= 14 and competences['svt'] >= 0.7:
        justifications.append("votre intérêt pour les sciences de la vie")
    if notes['note_geologie'] >= 14 and competences['geologie'] >= 0.7:
        justifications.append("votre intérêt pour les géosciences")
    if notes['note_anglais'] >= 14 and competences['anglais'] >= 0.7:
        justifications.append("votre bon niveau en anglais")

    if justifications:
        return f"Recommandé pour {' et '.join(justifications)}."
    return "Correspond à votre profil général."

# ============================================================
# INTERFACE UTILISATEUR
# ============================================================
st.title("🎓 Système de Recommandation de Filière")
st.markdown("### Faculté des Sciences - Université Ibn Tofail, Kénitra")
st.markdown("---")

st.markdown("""
**Bienvenue !** Ce système vous aide à identifier la filière universitaire la plus adaptée à votre profil.
Remplissez le formulaire ci-dessous avec vos notes et vos préférences.
""")

# Formulaire de saisie
with st.form("profil_etudiant"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Vos notes (sur 20)")
        note_maths = st.slider("Mathématiques", 0.0, 20.0, 14.0, 0.5)
        note_physique = st.slider("Physique", 0.0, 20.0, 13.0, 0.5)
        note_chimie = st.slider("Chimie", 0.0, 20.0, 12.0, 0.5)
        note_svt = st.slider("SVT / Biologie", 0.0, 20.0, 12.0, 0.5)
        note_geologie = st.slider("Géologie", 0.0, 20.0, 10.0, 0.5)
        note_francais = st.slider("Français", 0.0, 20.0, 13.0, 0.5)
        note_anglais = st.slider("Anglais", 0.0, 20.0, 14.0, 0.5)

    with col2:
        st.subheader("🎯 Votre profil")
        serie_bac = st.selectbox(
            "Série du Baccalauréat",
            ["Sciences Maths", "PC (Physique-Chimie)", "SVT", "BCG (Bio-Chimie-Géologie)", "Économie"]
        )
        interet_principal = st.selectbox(
            "Centre d'intérêt principal",
            ["Programmation / Informatique", "Mathématiques", "Physique", "Chimie",
             "Biologie / Santé", "Géologie / Environnement", "Énergies", "Ingénierie"]
        )
        preference_carriere = st.selectbox(
            "Préférence de carrière",
            ["Ingénieur", "Chercheur", "Médecin / Santé", "Enseignant", "Technicien", "Entrepreneur"]
        )

    submitted = st.form_submit_button("🔍 Obtenir mes recommandations", use_container_width=True)

# ============================================================
# TRAITEMENT ET AFFICHAGE DES RÉSULTATS
# ============================================================
if submitted:
    # Créer le profil de l'étudiant
    notes = {
        'note_maths': note_maths,
        'note_physique': note_physique,
        'note_chimie': note_chimie,
        'note_svt': note_svt,
        'note_geologie': note_geologie,
        'note_francais': note_francais,
        'note_anglais': note_anglais
    }

    # ✅ CORRECTION : Normaliser en divisant par 20 (échelle 0-1)
    notes_normalisees = np.array(list(notes.values())) / 20.0

    # Créer le vecteur de profil
    profil = pd.Series(notes_normalisees, index=filieres_cols)

    # Calculer la similarité avec chaque filière
    filieres_vectors = df_filieres[filieres_cols].values
    similarites = cosine_similarity(profil.values.reshape(1, -1), filieres_vectors)[0]

    # ============================================================
    # BONUS SELON LA SÉRIE DU BAC
    # ============================================================
    bonus_serie = {
        'Sciences Maths': {
            'Intelligence Artificielle': 0.15,
            'Informatique Fondamentale': 0.15,
            'Mathématiques et Applications': 0.15,
            'Mathématiques et Informatique pour la Décision': 0.15
        },
        'PC (Physique-Chimie)': {
            "Ingénierie Électronique et Traitement de l'Information": 0.15,
            'Physique Fondamentale et Applications': 0.15,
            'Ingénieries des Énergies Renouvelables': 0.15,
            'Génie des Matériaux': 0.10,
            'Génie des Procédés Industriels': 0.10,
            'Chimie et Applications': 0.10
        },
        'SVT': {
            'Sciences Biomédicales et Santé': 0.15,
            'Biotechnologie et Productions Végétales': 0.15,
            'Biosciences': 0.15,
            "Sciences et Techniques de l'Eau en Agriculture": 0.10
        },
        'BCG (Bio-Chimie-Géologie)': {
            'Sciences Biomédicales et Santé': 0.15,
            'Géoresources': 0.15,
            "Génie Civil et Géosciences de l'Environnement": 0.15,
            'Biosciences': 0.10,
            'Analyse et Qualité': 0.10
        }
    }

    if serie_bac in bonus_serie:
        for i, filiere in enumerate(df_filieres.index):
            if filiere in bonus_serie[serie_bac]:
                similarites[i] += bonus_serie[serie_bac][filiere]

    # ============================================================
    # BONUS SELON LE CENTRE D'INTÉRÊT PRINCIPAL
    # ============================================================
    bonus_interet = {
        'Programmation / Informatique': {
            'Intelligence Artificielle': 0.15,
            'Informatique Fondamentale': 0.15,
            'Mathématiques et Informatique pour la Décision': 0.10
        },
        'Mathématiques': {
            'Mathématiques et Applications': 0.15,
            'Intelligence Artificielle': 0.10,
            'Informatique Fondamentale': 0.10
        },
        'Physique': {
            'Physique Fondamentale et Applications': 0.15,
            "Ingénierie Électronique et Traitement de l'Information": 0.10
        },
        'Chimie': {
            'Chimie et Applications': 0.15,
            'Génie des Matériaux': 0.10,
            'Génie des Procédés Industriels': 0.10
        },
        'Biologie / Santé': {
            'Sciences Biomédicales et Santé': 0.15,
            'Biosciences': 0.15,
            'Biotechnologie et Productions Végétales': 0.10
        },
        'Géologie / Environnement': {
            'Géoresources': 0.15,
            "Génie Civil et Géosciences de l'Environnement": 0.15
        },
        'Énergies': {
            'Ingénieries des Énergies Renouvelables': 0.15
        },
        'Ingénierie': {
            "Génie Civil et Géosciences de l'Environnement": 0.10,
            'Génie des Matériaux': 0.10,
            "Ingénierie Électronique et Traitement de l'Information": 0.10
        }
    }

    if interet_principal in bonus_interet:
        for i, filiere in enumerate(df_filieres.index):
            if filiere in bonus_interet[interet_principal]:
                similarites[i] += bonus_interet[interet_principal][filiere]

    # ============================================================
    # BONUS SELON LA PRÉFÉRENCE DE CARRIÈRE
    # ============================================================
    bonus_carriere = {
        'Ingénieur': [
            'Intelligence Artificielle',
            'Informatique Fondamentale',
            "Génie Civil et Géosciences de l'Environnement",
            "Ingénierie Électronique et Traitement de l'Information",
            'Génie des Matériaux'
        ],
        'Chercheur': [
            'Physique Fondamentale et Applications',
            'Chimie et Applications',
            'Biosciences',
            'Mathématiques et Applications'
        ],
        'Médecin / Santé': [
            'Sciences Biomédicales et Santé',
            'Radioprotection et Dosimétrie Médicale'
        ],
        'Enseignant': [
            'Mathématiques et Applications',
            'Physique Fondamentale et Applications',
            'Chimie et Applications',
            'Biosciences'
        ],
        'Technicien': [
            'Génie des Procédés Industriels',
            'Génie des Matériaux',
            "Ingénierie Électronique et Traitement de l'Information"
        ],
        'Entrepreneur': [
            'Intelligence Artificielle',
            'Génie des Procédés Industriels',
            'Génie des Matériaux'
        ]
    }

    if preference_carriere in bonus_carriere:
        for i, filiere in enumerate(df_filieres.index):
            if filiere in bonus_carriere[preference_carriere]:
                similarites[i] += 0.10

    # Normaliser les scores finaux entre 0 et 1
    similarites = np.clip(similarites, 0, 1)

    # Classer les filières
    resultats = sorted(zip(df_filieres.index, similarites), key=lambda x: x[1], reverse=True)

    # ============================================================
    # AFFICHAGE DES RÉSULTATS
    # ============================================================
    st.markdown("---")
    st.subheader("🎯 Vos 3 filières recommandées")

    for i, (filiere, score) in enumerate(resultats[:3]):
        with st.container():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"### {i+1}. {filiere}")
                st.markdown(f"*{justifier_recommandation(notes, filiere)}*")
            with col_b:
                st.metric("Score de compatibilité", f"{score*100:.1f}%")
            st.progress(float(score))
            st.markdown("")

    # Afficher toutes les filières
    with st.expander("📋 Voir le classement complet des 18 filières"):
        df_resultats = pd.DataFrame(resultats, columns=['Filière', 'Score'])
        df_resultats['Score (%)'] = (df_resultats['Score'] * 100).round(1)
        df_resultats = df_resultats[['Filière', 'Score (%)']]
        st.dataframe(df_resultats, use_container_width=True, hide_index=True)

    # Conseil personnalisé
    st.markdown("---")
    st.info(f"""
    💡 **Conseil personnalisé** : Avec votre profil, la filière **{resultats[0][0]}** semble la plus adaptée.
    Nous vous recommandons de consulter le descriptif détaillé de cette filière sur le site de la faculté
    et de contacter le responsable pédagogique pour plus d'informations.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Projet de système de recommandation - Faculté des Sciences de Kénitra<br>
    Université Ibn Tofail - 2025</small>
</div>
""", unsafe_allow_html=True)