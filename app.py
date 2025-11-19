import os
from flask import Flask, render_template, request, jsonify, send_from_directory,url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
# Configuration
UPLOAD_FOLDER = 'uploads'
SPECTROGRAM_FOLDER = 'static/spectrograms'
MODEL_PATH = 'model/bird_model.h5' # put your .h5 here
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

SR = 22050
DURATION = 5 # seconds
SAMPLES = SR * DURATION
N_MELS = 128
HOP_LENGTH = 512
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Load model once at startup
print('Loading model...')
model = tf.keras.models.load_model(MODEL_PATH)
print('Model loaded.')
# Replace with your real label list (in same order used during training)
BIRD_LABELS = ['bewickii','cardinalis','melodia','migratorius','polyglottos']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_audio_fixed(filepath, sr=SR, duration=DURATION, n_mels=N_MELS,
hop_length=HOP_LENGTH):
    y, _ = librosa.load(filepath, sr=sr, mono=True)
# Trim/pad to fixed duration
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
    hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # normalize to 0-1
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    # Ensure fixed time frames (e.g., 216 frames) by trimming/padding time axis
    # Many models accept variable; if yours requires exact shape, adapt here.
    return mel_db, mel_norm

def save_spectrogram_image(mel_db, out_path):
    plt.figure(figsize=(6, 3))
    librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH,x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/api/predict', methods=['POST'])

def api_predict():
# Expects multipart/form-data with field 'file'
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(save_path)
    # Preprocess and create spectrogram image
        mel_db, mel_norm = preprocess_audio_fixed(save_path)
    # For model input shape: expand dims to (1, H, W, 1) if model expects channels
        x = mel_norm
    # If model was trained on a fixed time dimension, ensure x shape matches
        x = np.expand_dims(x, axis=-1)
        x = np.expand_dims(x, axis=0)
    # Predict
        preds = model.predict(x)[0]
    # Top-3
        top_idx = preds.argsort()[-3:][::-1]
        top = [{'species': BIRD_LABELS[int(i)], 'confidence':float(preds[int(i)])} for i in top_idx]
    # Save spectrogram image and return URL
        base, _ = os.path.splitext(filename)
        out_png = f'{base}.png'
        out_path = os.path.join(SPECTROGRAM_FOLDER, out_png)
        save_spectrogram_image(mel_db, out_path)
        spectrogram_url = url_for('static', filename=f'spectrograms/{out_png}')
        return jsonify({'predictions': top, 'spectrogram_url': spectrogram_url})
    return jsonify({'error': 'file type not allowed'}), 400
if __name__ == '__main__':
    app.run(debug=True)