from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from sfm_module import run_sfm

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'files[]' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('files[]')
        filenames = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                filenames.append(file_path)
        points_3d = run_sfm(filenames)
        return render_template('result.html', points_3d=points_3d)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
