from django import forms

class VideoUploadForm(forms.Form):
    video = forms.FileField(
        label='Upload Video',
        help_text='Upload a video file (MP4 format recommended)',
        widget=forms.FileInput(attrs={'accept': 'video/*'})
    )
