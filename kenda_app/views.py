from django.shortcuts import render
from django.http import StreamingHttpResponse
import yolov5
import torch
import numpy as np
import cv2
import os
from PIL import Image

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render, redirect

from django.core.files.storage import FileSystemStorage

"""
def index(request):
 
    if request.method == 'POST':
        form = HotelForm(request.POST, request.FILES)
 
        if form.is_valid():
            form.save()
            return redirect('result')
    else:
        form = HotelForm()
    return render(request, 'index.html', {'form': form})
 
"""
def index(request):
    if request.method == 'POST' and request.FILES['my_image']:
        my_image = request.FILES['my_image']
        fs = FileSystemStorage()
        filename = fs.save("kenda.jpg", my_image)
        uploaded_file_url = fs.url(filename)

        ################################ Model #####################################
        model = yolov5.load('best.pt')
        #device = select_device('') # 0 for gpu, '' for cpu
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        model.conf = 0.25
        model.iou = 0.30
 
    
        ################################ File upload #####################################
        file_source= os.path.join(settings.MEDIA_ROOT, uploaded_file_url.split("/")[-1])
        #file_source1='C:\Django_Projects\Kenda_yolo\kenda_app\static\images\Baa5.jpg'
        img = cv2.imread(file_source)
        results = model(img, augment=True)


        #pdd=results.pandas().xyxy[0]
        #label = f'{pred_class_name} {pred_score:.2f}'
        
        for row in results.pandas().xyxy[0].to_dict("records"):
            x0 = int(row["xmin"])
            y0 = int(row["ymin"])
            x1 = int(row["xmax"])
            y1 = int(row["ymax"])
            cv2.rectangle(img, (x0, y0), (x1, y1), (200, 50, 0), 2)
            cv2.putText(img, str(row["name"]) +"/Score:"+  str(round(float(row["confidence"]),2)), (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,50,0), 1)
        

            #cropped_img = img[int(row["ymin"]) : int(row["ymax"]), int(row["xmin"]) : int(row["xmax"])]

        """"
        for box,class_id ,score in zip(results.xyxy[0],results.xyxyn[0][:, -1].numpy(),results.xyxyn[0][:, -2].numpy()): 
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            cv2.rectangle(img, (x0, y0), (x1, y1), (200, 50, 0), 2)
            cv2.putText(img, str(class_id) +"/Score:"+  f"{score:.2f}%", (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,50,0), 1)

        """
        img1 = Image.fromarray(img)
        filename = settings.MEDIA_ROOT+'\img_inf.jpg'
        img1.save(filename)
        return render(request,'index.html',{'img_inf': '/media/img_inf.jpg'} ) #, 
    return render(request, 'index.html')
 
  


    



