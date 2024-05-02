"""final_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from project import views

urlpatterns = [
    path('admin/', admin.site.urls), 
    path('home/',views.home,name='home'),
    path('',views.home,name='home'),
    path('admin1/',views.admin1,name='admin1'),
    path('fac/',views.fac,name='fac'),
    path('adminstart/',views.adminstart,name='adminstart'),
    path('facreg/',views.facreg,name='facreg'),
    path('stdreg/',views.stdreg,name='stdreg'),
    path('fac1/',views.fac1,name='fac1'),
    path('detect', views.detect,name="detect"),
    path('TakeImages', views.TakeImages,name="TakeImages"),
    path('train', views.train,name="train"),
    path('fac2',views.fac2,name="fac2"),
    path('std',views.std,name="std"),
    path('facdata',views.facdata,name='facdata'),
    path('displaydata',views.displaydata,name='displaydata'),
    path('editemp/<int:id>',views.editemp,name='editemp'),
    path('updateemp/<int:id>',views.updateemp,name='updateemp'),
    path('ad1/',views.ad1,name='ad1'),
    path('atten',views.atten,name='atten'),

]