from django.urls import path
from .views import homePageView, aboutPageView, paulPageView, homePost, results


urlpatterns = [
    path('', homePageView, name='home'),
    path('about/', aboutPageView, name='about'),
    path('paul/', paulPageView, name='paul'),
    path('homePost/', homePost, name='homePost'),
    path('results/<int:dist_covered>/<int:cost>/<int:prev_club_cost>/<str:club>', results, name='results'),
]
