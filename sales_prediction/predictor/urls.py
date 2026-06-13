
# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.predict_sales, name='predict'),  # Root URL goes directly to prediction page
# ]
from django.urls import path
from . import views

urlpatterns = [

    path('', views.home, name='home'),

    path('predict/', views.predict_sales, name='predict_sales'),
    path('peak-month/', views.peak_month, name='peak_month'),

]