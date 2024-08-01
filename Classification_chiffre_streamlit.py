import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import io
import matplotlib.pyplot as plt
import os

# Vérifier si le fichier du modèle existe
model_path = 'cnn_modele.h5'
if not os.path.exists(model_path):
    st.error(f"Le fichier du modèle '{model_path}' est introuvable. Veuillez vous assurer qu'il est présent dans le répertoire approprié.")
else:
    try:
        # Charger le modèle pré-entraîné
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")

# Fonction pour afficher une image aléatoire et prédire le chiffre
def predict_random_image(X_test):
    idx = np.random.randint(0, X_test.shape[0])
    image = X_test[idx]
    image_reshaped = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image_reshaped)
    predicted_digit = np.argmax(prediction)
    return idx, image, predicted_digit, prediction

# Fonction pour prédire le chiffre dessiné
def predict_drawn_image(drawn_image):
    drawn_image = Image.fromarray(drawn_image).convert('L')
    drawn_image = ImageOps.fit(drawn_image, (28, 28))
    drawn_image = np.array(drawn_image).reshape(1, 28, 28, 1)
    drawn_image = drawn_image / 255.0
    prediction = model.predict(drawn_image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit, prediction

# Initialiser les variables de session
if 'validation_count_test' not in st.session_state:
    st.session_state.validation_count_test = 0
if 'correct_count_test' not in st.session_state:
    st.session_state.correct_count_test = 0
if 'validation_count_drawn' not in st.session_state:
    st.session_state.validation_count_drawn = 0
if 'correct_count_drawn' not in st.session_state:
    st.session_state.correct_count_drawn = 0
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0  # To track canvas state for resetting
if 'last_drawn_prediction' not in st.session_state:
    st.session_state.last_drawn_prediction = None  # Store the last prediction results for drawn digits

# Interface Streamlit
st.title("Reconnaissance de chiffres")

# Menu déroulant pour sélectionner une option
option = st.selectbox(
    "Choisissez une option",
    ("Donnée de Test", "Dessine moi un Chiffre", "Statistiques", "Diagramme")
)

if option == "Donnée de Test":
    st.header("Affichage d'une image aléatoire du jeu de test")

    # Lire le fichier de test prédéfini
    try:
        X_test = pd.read_csv("test.csv").values
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_test = X_test / 255.0

        idx, image, predicted_digit, prediction = predict_random_image(X_test)
        st.image(image.squeeze(), width=150, caption=f"Indice: {idx}")
        st.write(f"Chiffre prédit: {predicted_digit}")

        # Affichage de la probabilité du chiffre prédit
        prob_predicted_digit = prediction[0][predicted_digit]
        st.write(f"Probabilité du chiffre prédit ({predicted_digit}): {prob_predicted_digit:.4f}")

        # Validation de la prédiction
        st.header("Valider la prédiction")
        correct = st.radio("La prédiction est-elle correcte?", ("Oui", "Non"), key="correct_test")
        if st.button("Valider", key="validate_test"):
            st.session_state.validation_count_test += 1
            if correct == "Oui":
                st.session_state.correct_count_test += 1
                st.write("La prédiction a été validée comme correcte.")
            else:
                st.write("La prédiction a été validée comme incorrecte.")

            # Ajouter la prédiction à l'historique
            st.session_state.predictions.append({
                'source': 'Test',
                'image_idx': idx,
                'predicted_digit': predicted_digit,
                'probability': prob_predicted_digit,
                'is_correct': correct == "Oui"
            })

        # Affichage des comptes pour les données de test
        st.write(f"Validation Count Test: {st.session_state.validation_count_test}")
        st.write(f"Correct Count Test: {st.session_state.correct_count_test}")

    except FileNotFoundError:
        st.error("Le fichier de test 'test.csv' est introuvable. Veuillez vous assurer qu'il est présent dans le répertoire approprié.")

elif option == "Dessine moi un Chiffre":
    st.header("Dessiner un chiffre et prédire")

    # Réinitialiser le dessin si demandé
    if st.button("Réinitialiser le dessin"):
        # Change la clé du canvas pour le réinitialiser
        st.session_state.canvas_key += 1
        st.session_state.last_drawn_prediction = None  # Clear last prediction on reset

    # Configuration du canvas avec des valeurs fixes
    canvas_result = st_canvas(
        stroke_width=9,  # Épaisseur du trait fixe
        stroke_color="#FFFFFF",  # Couleur du trait fixe (blanc)
        background_color="black",  # Fond noir
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",  # Dynamic key to reset canvas
    )

    # Bouton pour prédire le chiffre dessiné
    if st.button("Predict Drawn Digit"):
        if canvas_result.image_data is not None:
            drawn_image = canvas_result.image_data

            # Prétraitement et prédiction de l'image dessinée
            predicted_digit, prediction = predict_drawn_image(drawn_image)

            # Stocker la dernière prédiction dans la session
            st.session_state.last_drawn_prediction = {
                'predicted_digit': predicted_digit,
                'probability': prediction[0][predicted_digit]
            }

    # Afficher le dernier résultat de prédiction si disponible
    if st.session_state.last_drawn_prediction is not None:
        predicted_digit = st.session_state.last_drawn_prediction['predicted_digit']
        prob_predicted_digit = st.session_state.last_drawn_prediction['probability']
        st.image(canvas_result.image_data, width=150, caption="Image dessinée après prétraitement")
        st.write(f"Chiffre prédit: {predicted_digit}")
        st.write(f"Probabilité du chiffre prédit ({predicted_digit}): {prob_predicted_digit:.4f}")

        # Validation de la prédiction
        st.header("Valider la prédiction")
        correct = st.radio("La prédiction est-elle correcte?", ("Oui", "Non"), key="correct_drawn")
        if st.button("Valider", key="validate_drawn"):
            st.session_state.validation_count_drawn += 1
            if correct == "Oui":
                st.session_state.correct_count_drawn += 1
                st.write("La prédiction du chiffre dessiné a été validée comme correcte.")
            else:
                st.write("La prédiction du chiffre dessiné a été validée comme incorrecte.")

            # Ajouter la prédiction à l'historique
            st.session_state.predictions.append({
                'source': 'Dessiné',
                'predicted_digit': predicted_digit,
                'probability': prob_predicted_digit,
                'is_correct': correct == "Oui"
            })

        # Affichage des comptes pour les dessins
        st.write(f"Validation Count Drawn: {st.session_state.validation_count_drawn}")
        st.write(f"Correct Count Drawn: {st.session_state.correct_count_drawn}")

elif option == "Statistiques":
    st.header("Monitoring des Prédictions")

    # Calcul des pourcentages
    if st.session_state.validation_count_test > 0:
        test_success_rate = (st.session_state.correct_count_test / st.session_state.validation_count_test) * 100
    else:
        test_success_rate = 0.0

    if st.session_state.validation_count_drawn > 0:
        drawn_success_rate = (st.session_state.correct_count_drawn / st.session_state.validation_count_drawn) * 100
    else:
        drawn_success_rate = 0.0

    total_validations = st.session_state.validation_count_test + st.session_state.validation_count_drawn
    total_correct = st.session_state.correct_count_test + st.session_state.correct_count_drawn

    if total_validations > 0:
        overall_success_rate = (total_correct / total_validations) * 100
    else:
        overall_success_rate = 0.0

    # Afficher les statistiques sous forme de tableau
    stats_data = {
        'Type': ['Images de Test', 'Chiffres Dessinés', 'Total'],
        'Validations': [st.session_state.validation_count_test, st.session_state.validation_count_drawn, total_validations],
        'Correctes': [st.session_state.correct_count_test, st.session_state.correct_count_drawn, total_correct],
        'Taux de Réussite (%)': [f"{test_success_rate:.2f}", f"{drawn_success_rate:.2f}", f"{overall_success_rate:.2f}"]
    }
    stats_df = pd.DataFrame(stats_data)
    st.table(stats_df)

    # Télécharger les statistiques en CSV
    st.subheader("Télécharger les Statistiques en CSV")
    predictions_df = pd.DataFrame(st.session_state.predictions)
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Télécharger les statistiques en CSV",
        data=csv,
        file_name="statistiques.csv",
        mime="text/csv"
    )

elif option == "Diagramme":
    st.header("Diagramme des Prédictions")

    # Calcul des pourcentages
    if st.session_state.validation_count_test > 0:
        test_success_rate = (st.session_state.correct_count_test / st.session_state.validation_count_test) * 100
    else:
        test_success_rate = 0.0

    if st.session_state.validation_count_drawn > 0:
        drawn_success_rate = (st.session_state.correct_count_drawn / st.session_state.validation_count_drawn) * 100
    else:
        drawn_success_rate = 0.0

    total_validations = st.session_state.validation_count_test + st.session_state.validation_count_drawn
    total_correct = st.session_state.correct_count_test + st.session_state.correct_count_drawn

    if total_validations > 0:
        overall_success_rate = (total_correct / total_validations) * 100
    else:
        overall_success_rate = 0.0

    # Données pour le graphique
    types = ['Images de Test', 'Chiffres Dessinés', 'Total']
    success_rates = [test_success_rate, drawn_success_rate, overall_success_rate]

    # Créer le graphique
    fig, ax = plt.subplots()
    ax.plot(success_rates, types, marker='o', label='Images de Test')
    ax.plot(success_rates, types, marker='o', label='Chiffres Dessinés')
    ax.set_ylabel('Type')
    ax.set_xlabel('Taux de Réussite (%)')
    ax.set_title('Taux de Réussite des Prédictions')
    ax.set_xlim(0, 100)
    ax.legend()

    # Ajouter du design et des couleurs
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f0f0f0')
    fig.patch.set_facecolor('#e0e0e0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(axis='x', colors='#333333')
    ax.tick_params(axis='y', colors='#333333')

    # Afficher le graphique
    st.pyplot(fig)
