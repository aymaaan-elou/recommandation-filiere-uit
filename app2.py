import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, roc_auc_score, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Prédiction du Risque d'Échec - UIT",
    page_icon="⚠️",
    layout="wide"
)

# ============================================================
# GÉNÉRATION DU DATASET D'ENTRAÎNEMENT
# ============================================================
@st.cache_data
def generer_dataset(n_etudiants=2000, taux_echec=0.15, seed=42):
    """
    Génère un dataset synthétique d'étudiants avec leur résultat final.
    Environ 15 % d'échecs pour simuler un déséquilibre réaliste.
    """
    rng = np.random.default_rng(seed)
    
    # Nombre d'échecs et de réussites
    n_echecs = int(n_etudiants * taux_echec)
    n_reussites = n_etudiants - n_echecs
    
    lignes = []
    
    # Générer les étudiants qui réussissent
    for _ in range(n_reussites):
        lignes.append({
            'moyenne_semestre_precedent': np.clip(rng.normal(13.5, 2.5), 6, 20),
            'note_controle_1':            np.clip(rng.normal(13, 3), 0, 20),
            'absences':                   max(0, int(rng.normal(4, 3))),
            'temps_trajet_min':           max(5, int(rng.normal(30, 20))),
            'boursier':                   rng.choice([0, 1], p=[0.7, 0.3]),
            'filiere':                    rng.choice(['SMI', 'SMC', 'SV', 'SVT', 'Géologie']),
            'echec':                      0
        })
    
    # Générer les étudiants qui échouent
    for _ in range(n_echecs):
        lignes.append({
            'moyenne_semestre_precedent': np.clip(rng.normal(9, 2), 0, 20),
            'note_controle_1':            np.clip(rng.normal(7, 3), 0, 20),
            'absences':                   max(0, int(rng.normal(15, 5))),
            'temps_trajet_min':           max(5, int(rng.normal(55, 30))),
            'boursier':                   rng.choice([0, 1], p=[0.6, 0.4]),
            'filiere':                    rng.choice(['SMI', 'SMC', 'SV', 'SVT', 'Géologie']),
            'echec':                      1
        })
    
    df = pd.DataFrame(lignes).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ============================================================
# ENTRAÎNEMENT DU MODÈLE
# ============================================================
@st.cache_resource
def entrainer_modele():
    """Entraîne un Random Forest avec gestion du déséquilibre."""
    df = generer_dataset()
    
    # Encodage de la filière
    le_filiere = LabelEncoder()
    df['filiere'] = le_filiere.fit_transform(df['filiere'])
    
    features = ['moyenne_semestre_precedent', 'note_controle_1', 'absences',
                'temps_trajet_min', 'boursier', 'filiere']
    
    X = df[features]
    y = df['echec']
    
    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE : rééquilibrage des classes
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Modèle avec pondération des classes
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)
    
    # Évaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'auc_roc':   roc_auc_score(y_test, y_proba)
    }
    
    # Validation croisée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    
    return model, le_filiere, metrics, cv_scores, features, X_test, y_test


# ============================================================
# INTERFACE
# ============================================================
st.title("⚠️ Prédiction du Risque d'Échec Académique")
st.markdown("### Faculté des Sciences - Université Ibn Tofail, Kénitra")
st.markdown("---")

st.markdown("""
**Bienvenue !** Ce système utilise un modèle de **Machine Learning** pour identifier
les étudiants à risque d'échec dès la 4ème semaine du semestre, afin de permettre
une intervention précoce.
""")

# Entraînement
with st.spinner("⏳ Entraînement du modèle..."):
    model, le_filiere, metrics, cv_scores, features, X_test, y_test = entrainer_modele()

# Sidebar
with st.sidebar:
    st.header("ℹ️ Performance du modèle")
    st.metric("F1-score", f"{metrics['f1']*100:.1f}%")
    st.metric("Recall (échecs détectés)", f"{metrics['recall']*100:.1f}%")
    st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    st.markdown("---")
    st.markdown(f"""
    - **Algorithme** : Random Forest
    - **Arbres** : 200
    - **Rééquilibrage** : SMOTE
    - **Validation croisée (F1)** : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}
    """)
    st.caption("Projet Master IA - UIT 2025")

# Formulaire
with st.form("profil_etudiant"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Données académiques")
        moyenne_prec = st.slider("Moyenne du semestre précédent", 0.0, 20.0, 12.0, 0.5)
        note_cc1 = st.slider("Note du premier contrôle continu", 0.0, 20.0, 11.0, 0.5)
        absences = st.slider("Nombre d'absences ce semestre", 0, 40, 5, 1)
    
    with col2:
        st.subheader("🎯 Profil de l'étudiant")
        temps_trajet = st.slider("Temps de trajet (minutes)", 5, 180, 30, 5)
        boursier = st.selectbox("Statut de boursier", ["Non", "Oui"])
        filiere = st.selectbox("Filière", list(le_filiere.classes_))
    
    submitted = st.form_submit_button("🔍 Évaluer le risque", use_container_width=True)

# ============================================================
# PRÉDICTION
# ============================================================
if submitted:
    profil = pd.DataFrame([{
        'moyenne_semestre_precedent': moyenne_prec,
        'note_controle_1': note_cc1,
        'absences': absences,
        'temps_trajet_min': temps_trajet,
        'boursier': 1 if boursier == "Oui" else 0,
        'filiere': le_filiere.transform([filiere])[0]
    }])
    
    proba_echec = model.predict_proba(profil)[0][1]
    proba_reussite = 1 - proba_echec
    
    # Affichage du résultat
    st.markdown("---")
    st.subheader("📊 Résultat de l'évaluation")
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        if proba_echec >= 0.6:
            st.error(f"⚠️ Risque ÉLEVÉ d'échec\n\n**{proba_echec*100:.1f}%**")
        elif proba_echec >= 0.35:
            st.warning(f"⚡ Risque MODÉRÉ d'échec\n\n**{proba_echec*100:.1f}%**")
        else:
            st.success(f"✅ Risque FAIBLE d'échec\n\n**{proba_echec*100:.1f}%**")
    
    with col_b:
        st.markdown("**Probabilités**")
        st.progress(float(proba_echec), text=f"Échec : {proba_echec*100:.1f}%")
        st.progress(float(proba_reussite), text=f"Réussite : {proba_reussite*100:.1f}%")
    
    # Recommandations
    st.markdown("---")
    st.subheader("💡 Recommandations")
    
    if proba_echec >= 0.6:
        st.error("""
        **Actions prioritaires :**
        - 📞 Contacter l'étudiant par téléphone dans les 48h
        - 📅 Planifier un entretien avec le bureau d'aide
        - 📚 Proposer un tutorat dans les matières concernées
        - 👥 Intégrer l'étudiant à un groupe de travail
        """)
    elif proba_echec >= 0.35:
        st.warning("""
        **Actions recommandées :**
        - 📧 Envoyer un email de sensibilisation
        - 📊 Suivre les résultats du prochain contrôle
        - 📖 Proposer des ressources pédagogiques en ligne
        """)
    else:
        st.success("""
        **Aucune action urgente nécessaire.**
        - Continuer le suivi habituel
        - Encourager l'étudiant dans sa progression
        """)
    
    # Facteurs de risque
    with st.expander("🧠 Quels sont les facteurs qui influencent cette prédiction ?"):
        importances = pd.DataFrame({
            'Facteur': ['Moyenne semestre préc.', 'Note contrôle 1', 'Absences',
                        'Temps trajet', 'Boursier', 'Filière'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importances.set_index('Facteur'))
        
        st.markdown("""
        **Interprétation** : Plus la barre est haute, plus le facteur est déterminant 
        dans la prédiction. Cela permet de comprendre *pourquoi* l'étudiant est à risque.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Projet de prédiction du risque d'échec - Faculté des Sciences de Kénitra<br>
    Université Ibn Tofail - 2025</small>
</div>
""", unsafe_allow_html=True)