# sauvegarde_modele.py
import joblib
import os

# Create the 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Suppose que rbm1, rbm2, clf sont entraînés
joblib.dump(rbm1, 'models/rbm1.pkl')
joblib.dump(rbm2, 'models/rbm2.pkl')
joblib.dump(clf, 'models/classifier.pkl')
print("Modèles sauvegardés !")