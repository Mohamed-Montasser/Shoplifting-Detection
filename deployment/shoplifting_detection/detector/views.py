import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import VideoUploadForm
from .model_utils import predict_video


def index(request):
    prediction_result = None
    video_url = None

    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the uploaded video
            video_file = request.FILES['video']

            # Save video temporarily
            fs = FileSystemStorage()
            filename = fs.save(video_file.name, video_file)
            video_path = fs.path(filename)
            video_url = fs.url(filename)

            try:
                # Make prediction
                prediction_result = predict_video(video_path)
            except Exception as e:
                prediction_result = {'error': str(e)}
            finally:
                # Optional: Delete the video after prediction to save space
                # os.remove(video_path)
                pass
    else:
        form = VideoUploadForm()

    context = {
        'form': form,
        'prediction_result': prediction_result,
        'video_url': video_url,
    }
    return render(request, 'detector/index.html', context)
