# Archive - Travaux de Recherche Initiaux

Ce dossier contient le code de recherche académique initial du projet, avant le pivot vers un SaaS production.

## Contenu

### Modèles Implémentés
- **SRCNN.py** : Super-Resolution CNN (3 couches)
- **FSRCNN.py** : Fast SRCNN (architecture avec bottleneck)

### Scripts d'Entraînement
- **training/** : Scripts pour entraîner SRCNN et FSRCNN
- **logs/** : Historiques d'entraînement (JSON)

### Modèles Entraînés
- **srcnn_baseline.pth** : SRCNN entraîné sur DIV2K (PSNR: 27.31 dB)
- **fsrcnn_baseline.pth** : FSRCNN entraîné (nécessite debug)

### Notebooks d'Expérimentation
- **baseline.ipynb** : Expériences initiales
- **exploration.ipynb** : Exploration du dataset
- **srcnn_architecture.ipynb** : Analyse de l'architecture SRCNN
- **test_dataset.ipynb** : Tests du dataset loader
- **plot_training.ipynb** : Visualisation des courbes d'entraînement

## Pourquoi Archivé ?

Ce code représente la phase de **recherche et apprentissage** du projet. Il a été archivé lors du pivot vers un projet **Fullstack AI SaaS** utilisant Real-ESRGAN (modèle SOTA pré-entraîné) pour la production.

## Enseignements Clés

1. **SRCNN** : Simple mais efficace pour comprendre les bases
2. **FSRCNN** : Plus complexe, nécessite fine-tuning des hyperparamètres
3. **Entraînement** : Importance de la loss function (MSE vs L1 vs Perceptual)
4. **Métriques** : PSNR/SSIM ne capturent pas la qualité perceptuelle

## Réutilisation

Ce code peut être réutilisé pour :
- Comprendre les fondamentaux de la super-résolution
- Comparer des architectures custom vs SOTA
- Enseignement / tutoriels
- Benchmarking

---

**Date d'archivage** : Janvier 2025
**Raison** : Pivot vers production SaaS avec Real-ESRGAN
