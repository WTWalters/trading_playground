#!/bin/bash

# TITAN Trading System Django Setup Script
# This script helps setup the Django environment and database

# Setup virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    echo "Virtual environment created."
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please create it manually."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e ..

# Create databases
echo "Creating databases..."
createdb -U $USER titan_django
createdb -U $USER titan_timescale

# Setup TimescaleDB extension on timescale database
echo "Setting up TimescaleDB extension..."
psql -U $USER -d titan_timescale -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"

# Run migrations
echo "Running migrations..."
python manage.py makemigrations
python manage.py migrate --database=default
python manage.py migrate --database=timescale

# Create superuser
echo "Creating superuser..."
python manage.py createsuperuser

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --no-input

echo "Setup complete!"
echo "You can now run the development server with: python manage.py runserver"
