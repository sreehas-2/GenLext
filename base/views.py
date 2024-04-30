from django.shortcuts import render
from django.http import JsonResponse
import random
import time
from agora_token_builder import RtcTokenBuilder
from .models import RoomMember
import json
from django.views.decorators.csrf import csrf_exempt                                     
# import requests
from io import BytesIO
from PIL import Image
from .emotions import predict_emotion
import base64
import io
import numpy as np
# Create your views here.

def lobby(request):
    return render(request, 'base/lobby.html')

def room(request):
    return render(request, 'base/room.html')


def getToken(request):
    appId = "1f41b377b05d47898f70f49ad50cde6b"
    appCertificate = "f4d5a92e98254e7490ce263e76920dd6"
    channelName = request.GET.get('channel')
    uid = random.randint(1, 230)
    expirationTimeInSeconds = 3600
    currentTimeStamp = int(time.time())
    privilegeExpiredTs = currentTimeStamp + expirationTimeInSeconds
    role = 1

    token = RtcTokenBuilder.buildTokenWithUid(appId, appCertificate, channelName, uid, role, privilegeExpiredTs)
    return JsonResponse({'token': token, 'uid': uid}, safe=False)

@csrf_exempt
def createMember(request): 
    data = json.loads(request.body)
    member, created = RoomMember.objects.get_or_create(
        name=data['name'],
        uid=data['UID'],
        room_name=data['room_name']
    )

    return JsonResponse({'name':data['name']}, safe=False)


def getMember(request):
    uid = request.GET.get('UID')
    room_name = request.GET.get('room_name')

    member = RoomMember.objects.get(
        uid=uid,
        room_name=room_name,
    )
    name = member.name
    return JsonResponse({'name':member.name}, safe=False)

@csrf_exempt
def deleteMember(request):
    data = json.loads(request.body)
    member = RoomMember.objects.get(
        name=data['name'],
        uid=data['UID'],
        room_name=data['room_name']
    )
    member.delete()
    return JsonResponse('Member deleted', safe=False)

@csrf_exempt
def capture_and_process(request): 
    # print ("inside")
    if request.method == 'POST':
        base64_screenshot = request.POST.get('screenshot')
        binary_data = base64.b64decode(base64_screenshot.split(',')[1]) 
        image = np.array(Image.open(io.BytesIO(binary_data)))
        list1= predict_emotion(image)
        print(list1)
        # print((list1[1]))
        result_emotion_id=list1[0]
        sleep_bool=list1[1]
        # if len(list1)==2:
        #     sleep_bool=list1[1]
        return JsonResponse({'emotions': result_emotion_id,'sleep': sleep_bool})
    return JsonResponse({'error': 'Invalid request method'}, status=400)
