#!/bin/bash

# Create necessary directories
mkdir -p app/{api,core,db,services,utils}
mkdir -p static templates data extracted_best
mkdir -p tests

# Touch empty __init__.py files to make directories importable
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
touch app/db/__init__.py
touch app/services/__init__.py
touch app/utils/__init__.py

# Create .gitkeep files in empty directories
touch data/.gitkeep
touch extracted_best/.gitkeep
touch static/.gitkeep
touch templates/.gitkeep

# Make sure the HTML files are in templates directory
if [ -f index.html ] && [ ! -f templates/index.html ]; then
    cp index.html templates/
fi

# Make sure CSS files are in static directory
if [ -f styles.css ] && [ ! -f static/styles.css ]; then
    cp styles.css static/
fi

# Move any images to static directory
find . -maxdepth 1 -name "*.png" -o -name "*.jpg" -o -name "*.gif" | xargs -I{} cp {} static/

echo "Directory setup complete!"