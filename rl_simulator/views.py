from django.shortcuts import render
from django.http import request, HttpResponse
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from io import BytesIO
import base64
from django.shortcuts import render
from stable_baselines3 import PPO
from rl_models.env_portfolio import StockPortfolioEnv
from rl_models.models import DRLAgent
import pickle
import base64

# Create your views here.

def asset_allocation(request):
    test_df_all = pickle.load(open('rl_models/rl_models_test_data.pkl', 'rb'))
    if request.method == 'POST':
        total_investment = int(request.POST.get('total-investment'))
        start_date = request.POST.get('start-date')
        end_date = request.POST.get('end-date')

        test_df = test_df_all[(test_df_all.date >= start_date) & (test_df_all.date < end_date)]
        test_df = test_df.sort_values(["date", "tic"], ignore_index=True)
        test_df.index = test_df.date.factorize()[0]

        trained_a2c = PPO.load("rl_models/trained_a2c_model.zip")
        trained_ppo = PPO.load("rl_models/trained_ppo_model.zip")

        stock_dimension = 20
        tech_indicator_list = ['f01','f02','f03','f04']

        env_kwargs_test = {
        "hmax": 500, 
        "initial_amount": total_investment, 
        "transaction_cost_pct": 0.001, 
        "state_space": stock_dimension, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": tech_indicator_list, 
        "action_space": stock_dimension, 
        "reward_scaling": 0,
        'initial_weights': [1/stock_dimension]*stock_dimension
        }

        e_trade_gym = StockPortfolioEnv(df = test_df, **env_kwargs_test)
        env_trade, obs_trade = e_trade_gym.get_sb_env()

        # A2C Test Model
        a2c_test_daily_return, a2c_test_weights = DRLAgent.DRL_prediction(model=trained_a2c,
                                test_data = test_df,
                                test_env = env_trade,
                                test_obs = obs_trade)
        
        a2c_test_weights.to_csv('results/a2c_test_weights.csv')
        
        # PPO Test Model
        ppo_test_daily_return, ppo_test_weights = DRLAgent.DRL_prediction(model=trained_ppo,
                                test_data = test_df,
                                test_env = env_trade,
                                test_obs = obs_trade)
        
        ppo_test_weights.to_csv('results/ppo_test_weights.csv')
        
        a2c_test_returns = a2c_test_daily_return.copy()
        ppo_test_returns = ppo_test_daily_return.copy()


        a2c_test_cum_returns = (1 + a2c_test_returns['daily_return']).cumprod()
        a2c_test_cum_returns.name = 'Portfolio 1: a2c Model'


        ppo_test_cum_returns = (1 + ppo_test_returns['daily_return']).cumprod()
        ppo_test_cum_returns.name = 'Portfolio 2: ppo Model'

        fig, ax = plt.subplots(figsize=(10,4))

        a2c_test_cum_returns.plot(ax=ax, color='darkorange', alpha=.4)
        ppo_test_cum_returns.plot(ax=ax, color='green', alpha=.4)
        plt.legend(loc="best")
        plt.grid(True)
        ax.set_ylabel("cummulative return")
        ax.set_title("Backtest based on the data from {0} to {1} with initial amount ${2}".format(start_date, end_date, total_investment), fontsize=14)
        fig.savefig('results/back_test_on_test_data.png')

        with open('results/back_test_on_test_data.png', 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        context = {
            'image': encoded_image
        }

        return render(request, 'image.html', context)

    dates = test_df_all['date'].unique()
    return render(request, 'simulator_3.html', {'dates': dates})



def display_a2c_test_weights(request):

    df = pd.read_csv('results/a2c_test_weights.csv')

    return render(request, 'plot_weigths_a2c.html', {'dataframe': df.to_html()})

def display_ppo_test_weights(request):

    df = pd.read_csv('results/ppo_test_weights.csv')

    return render(request, 'plot_weigths_a2c.html', {'dataframe': df.to_html()})