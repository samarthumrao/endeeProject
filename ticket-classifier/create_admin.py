import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ticket_system.settings')
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

# Create superuser
username = 'admin'
email = 'admin@ticketsystem.com'
password = 'admin123'

if not User.objects.filter(username=username).exists():
    User.objects.create_superuser(username=username, email=email, password=password)
    print(f"Superuser created successfully!")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print(f"Email: {email}")
else:
    print("Superuser already exists!")
    print(f"Username: {username}")
    print(f"Password: {password}")
