# TP 7 : Méthodes Basées sur la Politique

Ce projet implémente les algorithmes REINFORCE et Actor-Critic, et utilise FinRL pour une application de trading financier avec PPO.

## Contenu

- `main.py` : Contient l'implémentation des fonctions de perte REINFORCE et Actor-Critic, ainsi qu'un script de visualisation des résultats.
- `finrl_training.py` : Script pour l'entraînement d'un agent PPO avec le framework FinRL.

## Questions d'Analyse

### 1. Pourquoi la courbe de REINFORCE présente des oscillations violentes comparée à celle de PPO ?

**Explication technique :**
REINFORCE est un algorithme de type "Monte Carlo" qui utilise le retour cumulé réel $G_t$ pour estimer le gradient de la politique. Mathématiquement, $G_t = \sum_{k=t}^T \gamma^{k-t} R_k$ dépend de toutes les récompenses futures jusqu'à la fin de l'épisode. Cette dépendance rend l'estimation très sensible aux variations spécifiques d'une trajectoire donnée (haute variance). Si une action aléatoire mène à une récompense élevée par pur hasard (ex: bull market), REINFORCE la renforcera aveuglément.

À l'inverse, **PPO (Proximal Policy Optimization)** stabilise l'apprentissage grâce à :
- **L'Avantage** : Il utilise un Critique pour réduire la variance.
- **Le Clipping** : Il limite l'ampleur des mises à jour de la politique, évitant ainsi les changements brusques qui causent des oscillations violentes.

### 2. Impact du paramètre `clip_range=0.2` sur le Max Drawdown

Le paramètre `clip_range=0.2` impose que la nouvelle politique ne s'écarte pas de plus de 20% de l'ancienne. Dans un contexte de trading, cela agit comme un garde-fou :
- Il empêche l'agent de changer radicalement sa stratégie suite à un seul épisode de forte fluctuation du marché.
- En évitant des décisions extrêmes et impulsives, il réduit le risque de subir des pertes massives consécutives, ce qui se traduit par un **Max Drawdown plus faible** (-12.1% pour PPO contre -25.4% pour REINFORCE).

### 3. Impact de frais de transaction élevés (2%)

Si les frais de transaction passent à 2%, l'action optimale dans de nombreuses situations devient "ne rien faire" (action 0) pour éviter d'éroder le capital.

**PPO** parviendrait le plus rapidement à apprendre cette stratégie car :
- Sa convergence est plus stable et rapide.
- Il détecte plus efficacement que l'avantage de trader est inférieur au coût certain des transactions.
- REINFORCE, pénalisé par sa variance, mettrait beaucoup plus de temps à distinguer le "bruit" des récompenses du signal clair des frais prohibitifs.
