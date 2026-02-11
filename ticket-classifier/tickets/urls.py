"""
URL configuration for tickets app.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'tickets', views.TicketViewSet)
router.register(r'categories', views.CategoryViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('submit/', views.submit_ticket_view, name='submit_ticket'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
]
