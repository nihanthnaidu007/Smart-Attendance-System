from django.db import models
from django.db.models import Model

# Create your models here.
class adminlogin(models.Model):
	username=models.CharField(max_length=50,null=False)
	password=models.CharField(max_length=50,null=False)
class faculty(models.Model):
	username=models.CharField(max_length=50,null=False)
	password=models.CharField(max_length=50,null=False)
	
class editupdaterecord(models.Model):
	username=models.CharField(max_length=100)
	email=models.CharField(max_length=100)
	salary=models.IntegerField()
	class Meta:
		db_table="empdetails"

class facultyrecord(models.Model):
	username=models.CharField(max_length=100)
	email=models.CharField(max_length=100)
	password=models.CharField(max_length=100)
	dept=models.CharField(max_length=50)
	dob=models.CharField(max_length=50)
	gen=models.CharField(max_length=10)
	designation=models.CharField(max_length=50)
	class Meta:
		db_table="facdetails"

class report(models.Model):
	Name=models.CharField(max_length=100)
	Time=models.CharField(max_length=100)
	Date=models.CharField(max_length=100)
	class Meta:
		db_table="report"