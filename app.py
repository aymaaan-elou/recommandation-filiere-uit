import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Système de Recommandation de Filière - UIT",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# DÉFINITION DES 18 FILIÈRES AVEC LEURS PROFILS DE COMPÉTENCES
# (utilisés uniquement pour générer les données d'entraînement)
# ============================================================
PROFILS_FILIERES = {
    'Intelligence Artificielle':                              {'maths': 16, 'physique': 13, 'chimie': 10, 'svt': 10, 'geologie': 8,  'francais': 13, 'anglais': 16},
    'Informatique Fondamentale':                              {'maths': 16, 'physique': 14, 'chimie': 10, 'svt': 10, 'geologie': 8,  'francais': 13, 'anglais': 15},
    'Mathématiques et Applications':                          {'maths': 18, 'physique': 14, 'chimie': 10, 'svt': 10, 'geologie': 8,  'francais': 13, 'anglais': 13},
    'Mathématiques et Informatique pour la Décision':         {'maths': 17, 'physique': 14, 'chimie': 10, 'svt': 10, 'geologie': 8,  'francais': 14, 'anglais': 14},
    "Ingénierie Électronique et Traitement de l'Information": {'maths': 13, 'physique': 17, 'chimie': 12, 'svt': 9,  'geologie': 8,  'francais': 12, 'anglais': 14},
    'Physique Fondamentale et Applications':                  {'maths': 15, 'physique': 18, 'chimie': 13, 'svt': 10, 'geologie': 8,  'francais': 13, 'anglais': 13},
    'Ingénieries des Énergies Renouvelables':                 {'maths': 13, 'physique': 17, 'chimie': 14, 'svt': 10, 'geologie': 10, 'francais': 12, 'anglais': 14},
    'Radioprotection et Dosimétrie Médicale':                 {'maths': 12, 'physique': 17, 'chimie': 14, 'svt': 13, 'geologie': 8,  'francais': 13, 'anglais': 14},
    'Génie des Matériaux':                                    {'maths': 12, 'physique': 15, 'chimie': 17, 'svt': 11, 'geologie': 10, 'francais': 12, 'anglais': 13},
    'Génie des Procédés Industriels':                         {'maths': 12, 'physique': 14, 'chimie': 17, 'svt': 11, 'geologie': 10, 'francais': 12, 'anglais': 13},
    'Chimie et Applications':                                 {'maths': 10, 'physique': 13, 'chimie': 18, 'svt': 12, 'geologie': 10, 'francais': 13, 'anglais': 12},
    'Analyse et Qualité':                                     {'maths': 11, 'physique': 12, 'chimie': 17, 'svt': 13, 'geologie': 11, 'francais': 14, 'anglais': 13},
    'Sciences Biomédicales et Santé':                         {'maths': 10, 'physique': 12, 'chimie': 15, 'svt': 18, 'geologie': 11, 'francais': 14, 'anglais': 14},
    'Biotechnologie et Productions Végétales':                {'maths': 9,  'physique': 11, 'chimie': 14, 'svt': 17, 'geologie': 12, 'francais': 13, 'anglais': 12},
    'Biosciences':                                            {'maths': 9,  'physique': 11, 'chimie': 13, 'svt': 18, 'geologie': 12, 'francais': 13, 'anglais': 12},
    "Sciences et Techniques de l'Eau en Agriculture":         {'maths': 10, 'physique': 13, 'chimie': 14, 'svt': 16, 'geologie': 14, 'francais': 13, 'anglais': 12},
    "Génie Civil et Géosciences de l'Environnement":          {'maths': 15, 'physique': 15, 'chimie': 11, 'svt': 12, 'geologie': 18, 'francais': 13, 'anglais': 13},
    'Géoresources':                                           {'maths': 12, 'physique': 13, 'chimie': 12, 'svt': 11, 'geologie': 18, 'francais': 13, 'anglais': 13},
}

SERIES_BAC = ['Sciences Maths', 'PC (Physique-Chimie)', 'SVT', 'BCG (Bio-Chimie-Géologie)', 'Économie']
INTERETS = ['Programmation / Informatique', 'Mathématiques', 'Physique', 'Chimie',
            'Biologie / Santé', 'Géologie / Environnement', 'Énergies', 'Ingénierie']
CARRIERES = ['Ingénieur', 'Chercheur', 'Médecin / Santé', 'Enseignant', 'Technicien', 'Informaticien']

# ============================================================
# GÉNÉRATION DU DATASET D'ENTRAÎNEMENT
# ============================================================
@st.cache_data
def generer_dataset(n_par_filiere: int = 200, bruit: float = 1.5, seed: int = 42):
    """
    Génère un dataset synthétique de n_par_filiere * 18 étudiants.
    Chaque étudiant est généré en ajoutant du bruit gaussien
    autour du profil idéal de sa filière.
    """
    rng = np.random.default_rng(seed)
    lignes = []

    for filiere, profil in PROFILS_FILIERES.items():
        for _ in range(n_par_filiere):
            etudiant = {
                'note_maths':     np.clip(rng.normal(profil['maths'],     bruit), 0, 20),
                'note_physique':  np.clip(rng.normal(profil['physique'],  bruit), 0, 20),
                'note_chimie':    np.clip(rng.normal(profil['chimie'],    bruit), 0, 20),
                'note_svt':       np.clip(rng.normal(profil['svt'],       bruit), 0, 20),
                'note_geologie':  np.clip(rng.normal(profil['geologie'],  bruit), 0, 20),
                'note_francais':  np.clip(rng.normal(profil['francais'],  bruit), 0, 20),
                'note_anglais':   np.clip(rng.normal(profil['anglais'],   bruit), 0, 20),
                'serie_bac':      _serie_compatible(filiere, rng),
                'interet_principal': _interet_compatible(filiere, rng),
                'preference_carriere': _carriere_compatible(filiere, rng),
                'filiere':        filiere
            }
            lignes.append(etudiant)

    return pd.DataFrame(lignes)


def _serie_compatible(filiere, rng):
    """Retourne une série de bac plausible pour la filière."""
    correspondances = {
        'Intelligence Artificielle':                              ['Sciences Maths'],
        'Informatique Fondamentale':                              ['Sciences Maths'],
        'Mathématiques et Applications':                          ['Sciences Maths'],
        'Mathématiques et Informatique pour la Décision':         ['Sciences Maths'],
        "Ingénierie Électronique et Traitement de l'Information": ['PC (Physique-Chimie)', 'Sciences Maths'],
        'Physique Fondamentale et Applications':                  ['PC (Physique-Chimie)'],
        'Ingénieries des Énergies Renouvelables':                 ['PC (Physique-Chimie)'],
        'Radioprotection et Dosimétrie Médicale':                 ['PC (Physique-Chimie)'],
        'Génie des Matériaux':                                    ['PC (Physique-Chimie)'],
        'Génie des Procédés Industriels':                         ['PC (Physique-Chimie)'],
        'Chimie et Applications':                                 ['PC (Physique-Chimie)'],
        'Analyse et Qualité':                                     ['PC (Physique-Chimie)', 'BCG (Bio-Chimie-Géologie)'],
        'Sciences Biomédicales et Santé':                         ['SVT', 'BCG (Bio-Chimie-Géologie)'],
        'Biotechnologie et Productions Végétales':                ['SVT', 'BCG (Bio-Chimie-Géologie)'],
        'Biosciences':                                            ['SVT', 'BCG (Bio-Chimie-Géologie)'],
        "Sciences et Techniques de l'Eau en Agriculture":         ['SVT', 'BCG (Bio-Chimie-Géologie)'],
        "Génie Civil et Géosciences de l'Environnement":          ['BCG (Bio-Chimie-Géologie)', 'PC (Physique-Chimie)'],
        'Géoresources':                                           ['BCG (Bio-Chimie-Géologie)'],
    }
    return rng.choice(correspondances.get(filiere, SERIES_BAC))


def _interet_compatible(filiere, rng):
    correspondances = {
        'Intelligence Artificielle':                              ['Programmation / Informatique', 'Mathématiques'],
        'Informatique Fondamentale':                              ['Programmation / Informatique', 'Mathématiques'],
        'Mathématiques et Applications':                          ['Mathématiques'],
        'Mathématiques et Informatique pour la Décision':         ['Mathématiques', 'Programmation / Informatique'],
        "Ingénierie Électronique et Traitement de l'Information": ['Physique', 'Ingénierie'],
        'Physique Fondamentale et Applications':                  ['Physique'],
        'Ingénieries des Énergies Renouvelables':                 ['Énergies', 'Physique', 'Ingénierie'],
        'Radioprotection et Dosimétrie Médicale':                 ['Physique', 'Biologie / Santé'],
        'Génie des Matériaux':                                    ['Chimie', 'Ingénierie'],
        'Génie des Procédés Industriels':                         ['Chimie', 'Ingénierie'],
        'Chimie et Applications':                                 ['Chimie'],
        'Analyse et Qualité':                                     ['Chimie'],
        'Sciences Biomédicales et Santé':                         ['Biologie / Santé'],
        'Biotechnologie et Productions Végétales':                ['Biologie / Santé'],
        'Biosciences':                                            ['Biologie / Santé'],
        "Sciences et Techniques de l'Eau en Agriculture":         ['Biologie / Santé', 'Géologie / Environnement'],
        "Génie Civil et Géosciences de l'Environnement":          ['Géologie / Environnement', 'Ingénierie'],
        'Géoresources':                                           ['Géologie / Environnement'],
    }
    return rng.choice(correspondances.get(filiere, INTERETS))


def _carriere_compatible(filiere, rng):
    correspondances = {
        'Intelligence Artificielle':                              ['Ingénieur', 'Chercheur', 'Informaticien'],
        'Informatique Fondamentale':                              ['Ingénieur', 'Chercheur', 'Informaticien'],
        'Mathématiques et Applications':                          ['Chercheur', 'Enseignant'],
        'Mathématiques et Informatique pour la Décision':         ['Ingénieur', 'Chercheur','Enseignant'],
        "Ingénierie Électronique et Traitement de l'Information": ['Ingénieur', 'Technicien'],
        'Physique Fondamentale et Applications':                  ['Chercheur', 'Enseignant'],
        'Ingénieries des Énergies Renouvelables':                 ['Ingénieur', 'Technicien'],
        'Radioprotection et Dosimétrie Médicale':                 ['Médecin / Santé', 'Technicien'],
        'Génie des Matériaux':                                    ['Ingénieur', 'Technicien'],
        'Génie des Procédés Industriels':                         ['Ingénieur', 'Technicien'],
        'Chimie et Applications':                                 ['Chercheur', 'Enseignant'],
        'Analyse et Qualité':                                     ['Technicien', 'Chercheur'],
        'Sciences Biomédicales et Santé':                         ['Médecin / Santé', 'Chercheur'],
        'Biotechnologie et Productions Végétales':                ['Chercheur', 'Enseignant'],
        'Biosciences':                                            ['Chercheur', 'Enseignant', 'Médecin / Santé'],
        "Sciences et Techniques de l'Eau en Agriculture":         ['Chercheur', 'Technicien'],
        "Génie Civil et Géosciences de l'Environnement":          ['Ingénieur', 'Technicien'],
        'Géoresources':                                           ['Chercheur', 'Technicien'],
    }
    return rng.choice(correspondances.get(filiere, CARRIERES))


# ============================================================
# ENTRAÎNEMENT DU MODÈLE (mis en cache pour ne pas réentraîner)
# ============================================================
@st.cache_resource
def entrainer_modele():
    """Entraîne un Random Forest sur le dataset synthétique."""
    df = generer_dataset()

    # Encodage des colonnes catégorielles
    df_encoded = df.copy()
    colonnes_cat = ['serie_bac', 'interet_principal', 'preference_carriere']
    encodeurs = {}
    for col in colonnes_cat:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encodeurs[col] = le

    # Séparation features / cible
    features = ['note_maths', 'note_physique', 'note_chimie', 'note_svt',
                'note_geologie', 'note_francais', 'note_anglais',
                'serie_bac', 'interet_principal', 'preference_carriere']
    X = df_encoded[features]
    y = df_encoded['filiere']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entraînement du Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, encodeurs, accuracy, features


# ============================================================
# FONCTION DE JUSTIFICATION (basée sur les features importantes)
# ============================================================
def justifier_recommandation(notes, filiere):
    """Génère une justification en français basée sur les notes."""
    profil_ideal = PROFILS_FILIERES[filiere]
    justifications = []

    if notes['note_maths'] >= 14 and profil_ideal['maths'] >= 15:
        justifications.append("vos bonnes notes en mathématiques")
    if notes['note_physique'] >= 14 and profil_ideal['physique'] >= 15:
        justifications.append("vos bonnes notes en physique")
    if notes['note_chimie'] >= 14 and profil_ideal['chimie'] >= 15:
        justifications.append("vos bonnes notes en chimie")
    if notes['note_svt'] >= 14 and profil_ideal['svt'] >= 15:
        justifications.append("votre intérêt pour les sciences de la vie")
    if notes['note_geologie'] >= 14 and profil_ideal['geologie'] >= 15:
        justifications.append("votre intérêt pour les géosciences")
    if notes['note_anglais'] >= 14 and profil_ideal['anglais'] >= 14:
        justifications.append("votre bon niveau en anglais")

    if justifications:
        return f"Recommandé pour {' et '.join(justifications)}."
    return "Correspond à votre profil général."


# ============================================================
# INTERFACE UTILISATEUR
# ============================================================
st.title("🎓 Système de Recommandation de Filière - Machine Learning")
st.markdown("### Faculté des Sciences - Université Ibn Tofail, Kénitra")
st.markdown("---")

st.markdown("""
**Bienvenue !** Ce système utilise un modèle de **Machine Learning** (Random Forest)
entraîné sur des profils d'étudiants pour recommander la filière la plus adaptée à votre profil.
""")

# Entraînement du modèle (affiché une seule fois grâce au cache)
with st.spinner("⏳ Chargement du modèle de Machine Learning..."):
    model, encodeurs, accuracy, features = entrainer_modele()

# Afficher les infos du modèle dans la sidebar
with st.sidebar:
    st.header("ℹ️ À propos du modèle")
    st.metric("Précision (accuracy)", f"{accuracy*100:.1f}%")
    st.markdown(f"""
    - **Algorithme** : Random Forest
    - **Arbres** : 200
    - **Données d'entraînement** : 3 600 profils
    - **Filières** : 18
    - **Features** : {len(features)}
    """)
    st.markdown("---")
    st.caption("Projet Master IA - UIT 2025")

# Formulaire
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
        serie_bac = st.selectbox("Série du Baccalauréat", SERIES_BAC)
        interet_principal = st.selectbox("Centre d'intérêt principal", INTERETS)
        preference_carriere = st.selectbox("Préférence de carrière", CARRIERES)

    submitted = st.form_submit_button("🔍 Obtenir mes recommandations", use_container_width=True)

# ============================================================
# PRÉDICTION
# ============================================================
if submitted:
    # Construction du profil encodé
    profil = pd.DataFrame([{
        'note_maths': note_maths,
        'note_physique': note_physique,
        'note_chimie': note_chimie,
        'note_svt': note_svt,
        'note_geologie': note_geologie,
        'note_francais': note_francais,
        'note_anglais': note_anglais,
        'serie_bac': encodeurs['serie_bac'].transform([serie_bac])[0],
        'interet_principal': encodeurs['interet_principal'].transform([interet_principal])[0],
        'preference_carriere': encodeurs['preference_carriere'].transform([preference_carriere])[0]
    }])

    # Prédiction des probabilités
    probabilites = model.predict_proba(profil)[0]
    classes = model.classes_

    # Résultats triés
    resultats = sorted(zip(classes, probabilites), key=lambda x: x[1], reverse=True)

    # Notes sous forme de dict pour la justification
    notes = {
        'note_maths': note_maths,
        'note_physique': note_physique,
        'note_chimie': note_chimie,
        'note_svt': note_svt,
        'note_geologie': note_geologie,
        'note_francais': note_francais,
        'note_anglais': note_anglais
    }

    # ============================================================
    # AFFICHAGE
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
                st.metric("Probabilité", f"{score*100:.1f}%")
            st.progress(float(score))
            st.markdown("")

    # Classement complet
    with st.expander("📋 Voir le classement complet des 18 filières"):
        df_resultats = pd.DataFrame(resultats, columns=['Filière', 'Probabilité'])
        df_resultats['Probabilité (%)'] = (df_resultats['Probabilité'] * 100).round(2)
        df_resultats = df_resultats[['Filière', 'Probabilité (%)']]
        st.dataframe(df_resultats, use_container_width=True, hide_index=True)

    # Top 3 features importantes
    with st.expander("🧠 Quelles sont les matières les plus déterminantes ?"):
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importances.set_index('Feature'))

    # Conseil personnalisé
    st.markdown("---")
    st.info(f"""
    💡 **Conseil personnalisé** : Avec votre profil, la filière **{resultats[0][0]}** 
    a la plus forte probabilité ({resultats[0][1]*100:.1f}%). 
    Nous vous recommandons de consulter le descriptif détaillé de cette filière 
    sur le site de la faculté et de contacter le responsable pédagogique.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Projet de recommandation par Machine Learning - Faculté des Sciences de Kénitra<br>
    Université Ibn Tofail - 2025</small>
</div>
""", unsafe_allow_html=True)