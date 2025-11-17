"""
Script pour tester que tous les artifacts sont présents et valides
À exécuter dans le dossier de votre API
"""

import os
import pickle
import json
import joblib

print("="*60)
print("VÉRIFICATION DES ARTIFACTS")
print("="*60)

required_files = {
    'best_random_forest_model.pkl': 'pickle',
    'location_price_map.pkl': 'pickle',
    'overall_mean_price.pkl': 'pickle',
    'scaler.joblib': 'joblib',
    'feature_columns.json': 'json',
    'bath_median.pkl': 'pickle',
    'balcony_median.pkl': 'pickle'
}

all_ok = True

for filename, file_type in required_files.items():
    print(f"\n{filename}:")
    
    # Vérifier existence
    if not os.path.exists(filename):
        print(f"  ✗ FICHIER MANQUANT")
        all_ok = False
        continue
    
    # Vérifier taille
    size = os.path.getsize(filename)
    print(f"  ✓ Existe ({size:,} bytes)")
    
    # Vérifier chargement
    try:
        if file_type == 'pickle':
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Afficher des infos selon le type de données
            if filename == 'location_price_map.pkl':
                print(f"  ✓ {len(data)} locations dans le mapping")
                print(f"    Exemple: {list(data.items())[:3]}")
            elif filename == 'overall_mean_price.pkl':
                print(f"  ✓ Prix moyen: {data:.2f}")
            elif filename in ['bath_median.pkl', 'balcony_median.pkl']:
                print(f"  ✓ Médiane: {data}")
            elif filename == 'best_random_forest_model.pkl':
                print(f"  ✓ Modèle chargé: {type(data).__name__}")
                
        elif file_type == 'joblib':
            data = joblib.load(filename)
            print(f"  ✓ Scaler chargé: {type(data).__name__}")
            
        elif file_type == 'json':
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  ✓ {len(data)} colonnes")
            print(f"    Premières: {data[:5]}")
            print(f"    Dernières: {data[-5:]}")
            
    except Exception as e:
        print(f"  ✗ ERREUR DE CHARGEMENT: {e}")
        all_ok = False

print("\n" + "="*60)
if all_ok:
    print("✓ TOUS LES ARTIFACTS SONT OK")
    print("\nVous pouvez maintenant lancer l'API:")
    print("  uvicorn price_API:app --reload")
    print("\nEt tester:")
    print("  http://127.0.0.1:8000/health")
    print("  http://127.0.0.1:8000/artifacts")
else:
    print("✗ CERTAINS ARTIFACTS MANQUENT OU SONT INVALIDES")
    print("\nVeuillez exécuter le script de sauvegarde dans votre notebook")

print("="*60)