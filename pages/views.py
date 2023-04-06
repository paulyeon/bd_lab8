# pages/views.py
from django.http import HttpResponse
from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView


def homePageView(request):
    # return request object and specify page.
    return render(request, 'home.html', {
    })


def aboutPageView(request):
    # return request object and specify page.
    return render(request, 'about.html')


def paulPageView(request):
    # return request object and specify page.
    return render(request, 'paul.html')


def homePost(request):
    # Use request object to extract choice.
    city = int(request.POST['city'])
    room_type = int(request.POST['room_type'])
    cleanliness = int(request.POST['cleanliness'])
    restaurant_index = int(request.POST['restaurant_index'])
    attraction_index = int(request.POST['attraction_index'])
    num_bedroom = int(request.POST['num_bedroom'])
    business_offer = int(request.POST['business_offer'])
    super_host = int(request.POST['super_host'])
    private_room = int(request.POST['private_room'])
    multiple_rooms = int(request.POST['multiple_rooms'])

    if cleanliness < 0 or cleanliness > 10:
        return render(request, 'home.html', {
            'errorMessage': '*** Cleanliness must be between 0 - 10.',
        })
    elif restaurant_index < 0 or restaurant_index > 100:
        return render(request, 'home.html', {
            'errorMessage': '*** Restaurant index must be between 0 - 100.',
        })
    elif attraction_index < 0 or attraction_index > 100:
        return render(request, 'home.html', {
            'errorMessage': '*** Attraction index must be between 0 - 100.',
        })
    elif num_bedroom < 0 or num_bedroom > 10:
        return render(request, 'home.html', {
            'errorMessage': '*** Number of bedrooms must be between 0 - 10.',
        })
    else:
        return HttpResponseRedirect(reverse('results', kwargs={
            'city': city,
            'room_type': room_type,
            'cleanliness': cleanliness,
            'restaurant_index': restaurant_index,
            'attraction_index': attraction_index,
            'num_bedroom': num_bedroom,
            'business_offer': business_offer,
            'super_host': super_host,
            'private_room': private_room,
            'multiple_rooms': multiple_rooms,
        },
            ))


import pickle
import sklearn  # You must perform a pip install.
import pandas as pd


def results(request, city, room_type, cleanliness, restaurant_index,
            attraction_index, num_bedroom, business_offer, super_host, private_room, multiple_rooms):
    # load saved model
    with open('./model_pkl', 'rb') as f:
        loadedModel = pickle.load(f)

    # Create a single prediction.
    singleSampleDf = pd.DataFrame(columns=['Cleanliness Rating', 'Normalised Restaurant Index',
                                           'Normalised Attraction Index', 'City',
                                           'Bedrooms', 'Business', 'Superhost', 'Room Type',
                                           'Private Room', 'Multiple Rooms'])

    simpleData = {
            'City': city,
            'Room Type': room_type,
            'Cleanliness Rating': cleanliness,
            'Normalised Restaurant Index': restaurant_index,
            'Normalised Attraction Index': attraction_index,
            'Bedrooms': num_bedroom,
            'Business': business_offer,
            'Superhost': super_host,
            'Private Room': private_room,
            'Multiple Rooms': multiple_rooms,
        }

    singleSampleDf = pd.concat([singleSampleDf,
                                pd.DataFrame.from_records([simpleData])])

    singlePrediction = loadedModel.predict(singleSampleDf)


    city_dict = {'Amsterdam': 0, 'Athens': 1, 'Barcelona': 2, 'Berlin': 3, 'Budapest': 4, 'Lisbon': 5, 'Paris': 6, 'Rome': 7, 'Vienna': 8}
    room_type_dict = {'Private room': 0, 'Entire home/apt': 1, 'Shared room': 2}
    tf_dict = {False: 0, True: 1}

    for key, val in city_dict.items():
        if val == city:
            city = key
    for key, val in room_type_dict.items():
        if val == room_type:
            room_type = key
    for key, val in tf_dict.items():
        if val == super_host:
            super_host = key
    for key, val in tf_dict.items():
        if val == private_room:
            private_room = key
    for key, val in tf_dict.items():
        if val == multiple_rooms:
            multiple_rooms = key

    return render(request, 'results.html', {
        'city': city,
        'room_type': room_type,
        'cleanliness': cleanliness,
        'restaurant_index': restaurant_index,
        'attraction_index': attraction_index,
        'num_bedroom': num_bedroom,
        'business_offer': business_offer,
        'super_host': super_host,
        'private_room': private_room,
        'multiple_rooms': multiple_rooms,
        'prediction': str(singlePrediction[0])})
