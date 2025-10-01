from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for
import subprocess
import os
import uuid

app = Flask(__name__)

# Folder konfigurasi
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'runs/detect'

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/deteksi-luka')
def deteksi_luka():
    return render_template('deteksi_luka.html')




@app.route('/detect', methods=['POST'])
def ajax_detect():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # running YOLOv5
    result = subprocess.run([
        'python', 'detect.py',
        '--weights', 'best_old.pt',
        '--source', filepath,
        '--save-txt',
        '--save-conf'
    ], capture_output=True, text=True)

    print("YOLOv5 Output:\n", result.stdout)
    if result.stderr:
        print("YOLOv5 Errors:\n", result.stderr)

    exp_folders = [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith('exp')]
    if not exp_folders:
        return jsonify({'success': False, 'message': 'No result folder found'}), 500

    latest_exp = max(exp_folders, key=lambda x: os.path.getctime(os.path.join(OUTPUT_FOLDER, x)))
    detected_image_path = os.path.join(latest_exp, filename)
    result_url = url_for('detected_image', path=detected_image_path)

    label_file = os.path.join(OUTPUT_FOLDER, latest_exp, 'labels', filename.rsplit('.', 1)[0] + '.txt')
    if not os.path.exists(label_file):
        return jsonify({'success': False, 'message': 'Detection label not found'}), 500

    with open(label_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line:
            return jsonify({'success': False, 'message': 'No detection found'}), 500
        class_index = int(first_line.split()[0]) 

    import json
    with open('penanganan.json', 'r') as f:
        instructions_data = json.load(f)

    matched_data = next((item for item in instructions_data if item['index'] == class_index), None)
    if not matched_data:
        return jsonify({'success': False, 'message': 'Instruction data not found for this class'}), 500

    return jsonify({
        'success': True,
        'result_url': result_url,
        'class_name': matched_data['label'],
        'instructions': matched_data['instructions']
    })


@app.route('/detected/<path:path>')
def detected_image(path):
    """
    Serve file hasil deteksi
    """
    directory = os.path.join(app.config['OUTPUT_FOLDER'], os.path.dirname(path))
    filename = os.path.basename(path)
    return send_from_directory(directory, filename)


# if __name__ == '__main__':
#     app.run(debug=True)
