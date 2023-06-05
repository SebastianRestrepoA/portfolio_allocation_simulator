"""
URL configuration for portfolio_allocation_simulator project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from rl_simulator.views import asset_allocation, display_a2c_test_weights, display_ppo_test_weights

urlpatterns = [
    path('admin/', admin.site.urls),
    path('allocation/', asset_allocation, name='asset_allocation'),
    path('', asset_allocation, name="index"),
    path('plot_weigths_a2c/', display_a2c_test_weights),
    path('plot_weigths_ppo/', display_ppo_test_weights),
]

