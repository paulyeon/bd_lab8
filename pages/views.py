# pages/views.py
from django.http import HttpResponse
from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView


def homePageView(request):
    # return request object and specify page.
    return render(request, 'home.html', {
        'distance_covered': [0, 7],
        'cost': [0, 201],
        'prev_club_cost': [0, 107],
        'clubs': ['CHE', 'MUN', 'LIV'],
    })


def aboutPageView(request):
    # return request object and specify page.
    return render(request, 'about.html')


def paulPageView(request):
    # return request object and specify page.
    return render(request, 'paul.html')


def homePost(request):
    # Use request object to extract choice.
    # Extract value from request object by control name.
    dist_covered = int(request.POST['dist_covered'])
    cost = int(request.POST['cost'])
    prev_club_cost = int(request.POST['prev_club_cost'])
    club = request.POST['choice']

    if dist_covered < 0 or dist_covered > 7:
        return render(request, 'home.html', {
            'errorMessage': '*** Distance Covered must be between 0 and 7.',
            'distance_covered': [0, 7],
            'cost': [0, 201],
            'prev_club_cost': [0, 107],
            'clubs': ['CHE', 'MUN', 'LIV'], })
    elif cost < 0 or cost > 201:
        return render(request, 'home.html', {
            'errorMessage': '*** Cost must be between 0 and 201.',
            'distance_covered': [0, 7],
            'cost': [0, 201],
            'prev_club_cost': [0, 107],
            'clubs': ['CHE', 'MUN', 'LIV'], })
    elif prev_club_cost < 0 or prev_club_cost > 107:
        return render(request, 'home.html', {
            'errorMessage': '*** Previous Club Cost must be between 0 and 107.',
            'distance_covered': [0, 7],
            'cost': [0, 201],
            'prev_club_cost': [0, 107],
            'clubs': ['CHE', 'MUN', 'LIV'], })
    else:
        return HttpResponseRedirect(reverse('results', kwargs={
            'dist_covered': dist_covered,
            'cost': cost,
            'prev_club_cost': prev_club_cost,
            'club': club}, ))


import pickle
import sklearn  # You must perform a pip install.
import pandas as pd


def results(request, dist_covered, cost, prev_club_cost, club):
    # load saved model
    with open('./model_pkl', 'rb') as f:
        loadedModel = pickle.load(f)

    # Create a single prediction.
    singleSampleDf = pd.DataFrame(columns=['DistanceCovered(InKms)', 'Cost',
                                           'PreviousClubCost', 'CHE', 'MUN', 'LIV'])
    che = 0
    mun = 0
    liv = 0
    if club == "CHE":
        che = 1
    elif club == "MUN":
        mun = 1
    else:
        liv = 1

    simpleData = {'DistanceCovered(InKms)': dist_covered,
                  'const': 1.0,
                  'Cost': cost,
                  'PreviousClubCost': prev_club_cost,
                  'CHE': che,
                  'MUN': mun,
                  'LIV': liv}

    singleSampleDf = pd.concat([singleSampleDf,
                                pd.DataFrame.from_records([simpleData])])

    singlePrediction = loadedModel.predict(singleSampleDf)

    print("Single prediction: " + str(singlePrediction))

    return render(request, 'results.html', {'dist_covered': dist_covered,
                                            'cost': cost,
                                            'prev_club_cost': prev_club_cost,
                                            'club': club, 'prediction': str(singlePrediction)})
