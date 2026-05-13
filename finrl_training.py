try:
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    import pandas as pd
    FINRL_AVAILABLE = True
except ImportError:
    FINRL_AVAILABLE = False

def train_ppo():
    if not FINRL_AVAILABLE:
        print("FinRL is not installed. Skipping Part 3.")
        return

    # 1. Chargement des donnees pre-traitees (Dow Jones 30, 2009 a 2020)
    try:
        df = pd.read_csv('train_data.csv')
    except FileNotFoundError:
        print("train_data.csv not found. Skipping Part 3.")
        return

    # Configuration de l'environnement FinRL
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": 30,
        "stock_dim": 30,
        "action_space": 30,
        "reward_scaling": 1e-4
    }

    env_train = StockTradingEnv(df=df, **env_kwargs)
    agent = DRLAgent(env=env_train)

    # --- A COMPLETER : Instancier le modele PPO ---
    # Hyperparametres : learning_rate = 0.00025, batch_size = 128
    PPO_PARAMS = {
        "learning_rate": 0.00025,
        "batch_size": 128,
        "ent_coef": 0.01,
        "clip_range": 0.2 # Limite stricte pour eviter les grands sauts de politique
    }

    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    # --- A COMPLETER : Entrainer sur 50000 timesteps ---
    print("Demarrage de l'entrainement PPO...")
    trained_ppo = agent.train_model(model=model_ppo,
                                     tb_log_name='ppo',
                                     total_timesteps=50000)
    print("Entrainement termine.")

if __name__ == "__main__":
    train_ppo()
