from django import forms
from .models import ContactSubmission

class ContactForm(forms.ModelForm):
    class Meta:
        model = ContactSubmission
        fields = ['name', 'problem', 'email']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Enter Name'}),
            'problem': forms.TextInput(attrs={'placeholder': 'Share your problem with us'}),
            'email': forms.EmailInput(attrs={'placeholder': 'Enter Email Address'}),
        }
