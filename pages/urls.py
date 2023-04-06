from django.urls import path
from .views import homePageView, aboutPageView, paulPageView, homePost, results


urlpatterns = [
    path('', homePageView, name='home'),
    path('about/', aboutPageView, name='about'),
    path('paul/', paulPageView, name='paul'),
    path('homePost/', homePost, name='homePost'),
    path('results/<int:city>/<int:room_type>/<int:cleanliness>/<int:restaurant_index>/<int:attraction_index>/'
         '<int:num_bedroom>/<int:business_offer>/<int:super_host>/<int:private_room>'
         '/<int:multiple_rooms>', results, name='results'),
]