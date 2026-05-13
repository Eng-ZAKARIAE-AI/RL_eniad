import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Part 1: REINFORCE Algorithm ---

def compute_reinforce_loss(rewards, log_probs, gamma=0.99):
    """
    Calculates the REINFORCE loss.
    
    Args:
        rewards: List of rewards for each step.
        log_probs: List of log probabilities of the actions taken.
        gamma: Discount factor.
    """
    G = 0
    returns = []
    
    # 1. Calcul des retours cumules (de la fin vers le debut)
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
        
    returns = torch.tensor(returns)
    
    # Normalisation des retours pour stabiliser l'apprentissage
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    policy_loss = []
    # 2. Calcul de la fonction de perte
    for log_prob, G_t in zip(log_probs, returns):
        # PyTorch minimise la loss, donc on utilise -log_prob * G_t
        loss_t = -log_prob * G_t
        policy_loss.append(loss_t)
        
    # On retourne la somme des pertes
    return torch.stack(policy_loss).sum()


# --- Part 2: Actor-Critic Algorithm ---

def compute_actor_critic_loss(rewards, states, next_states, log_probs, critic_net, gamma=0.99):
    """
    Calculates the Actor-Critic loss.
    
    Note: critic_net is passed as an argument here for clarity.
    """
    actor_loss = 0
    critic_loss = 0
    
    for i in range(len(rewards)):
        # 1. Valeur de l'etat actuel et du prochain etat
        V_s = critic_net(states[i])
        V_s_next = critic_net(next_states[i])
        
        # Calcul de la cible TD (Temporal Difference)
        td_target = rewards[i] + gamma * V_s_next
        
        # Calcul de l'avantage
        advantage = td_target - V_s
        
        # 2. Loss du Critique (Mean Squared Error)
        critic_loss += (td_target - V_s).pow(2)
        
        # 3. Loss de l'Acteur
        # On detache l'avantage pour ne pas retropropager dans le critique
        actor_loss += -log_probs[i] * advantage.detach()
        
    return actor_loss, critic_loss


# --- Part 4: Visualization (Simulation) ---

def visualize_results():
    # Simulation des recompenses par episode pour chaque algorithme
    episodes = np.arange(1, 501)
    
    # REINFORCE (Haute Variance)
    reinforce_rewards = np.random.normal(loc=np.log(episodes)*10, scale=30)
    
    # Actor-Critic (A2C) (Moins bruite)
    a2c_rewards = np.random.normal(loc=np.log(episodes)*15, scale=15)
    
    # PPO (Stable et optimal)
    ppo_rewards = np.log(episodes)*22 + np.random.normal(loc=0, scale=5)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(episodes, reinforce_rewards, label='REINFORCE (Haute Variance)', alpha=0.5)
    plt.plot(episodes, a2c_rewards, label='Actor-Critic (A2C)', alpha=0.7)
    plt.plot(episodes, ppo_rewards, label='PPO (Stable)', linewidth=2)
    
    plt.xlabel('Episodes d\'entrainement')
    plt.ylabel('Valeur Finale du Portefeuille ($)')
    plt.title('Comparaison de la Convergence : REINFORCE vs A2C vs PPO')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    print("Graphique de visualisation enregistre sous 'learning_curves.png'")
    # plt.show() # Can't show in CLI easily, so we save it.

def main():
    print("TP 7 : Methodes Basees sur la Politique")
    visualize_results()

if __name__ == "__main__":
    main()
