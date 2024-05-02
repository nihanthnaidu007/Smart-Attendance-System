from django import forms
from project.models import facultyrecord

class empforms(forms.ModelForm):
	class Meta:
		model=facultyrecord
		fields="__all__"