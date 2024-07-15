from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import sqlite3
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE'] = 'database.db'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create a database if it doesn't exist
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, filename TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('SELECT filename FROM images')
    images = cursor.fetchall()
    conn.close()
    images = [image[0] for image in images]
    return render_template('catrat_layout.html', images=images)

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    if files:
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        for file in files:
            if file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                cursor.execute('INSERT INTO images (filename) VALUES (?)', (filename,))
        conn.commit()
        conn.close()
    return redirect(url_for('home'))

@app.route('/process')
def process_files():
    # Simulate processing by changing progress bar
    # Here you would add your actual processing code
    return redirect(url_for('home'))

@app.route('/download')
def download_file():
    # Here you should add logic to zip processed files or similar
    # For simplicity, let's assume we are downloading a single processed file
    processed_file = 'path_to_processed_file.zip'  # Update this with actual processed file path
    return send_from_directory(directory='.', path=processed_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
