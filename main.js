// ------------------------------
// DOM ELEMENTS
// ------------------------------
const fileInput = document.getElementById("fileinput");
const uploadBtn = document.getElementById("uploadbtn");

const progress = document.getElementById("progress");
const bar = document.getElementById("bar");
const status = document.getElementById("status");

const predlist = document.getElementById("predlist");
const specimg = document.getElementById("specimg");
const speccanvas = document.getElementById("speccanvas");
const results = document.getElementById("results");


// ------------------------------
// HANDLE FILE SELECTION — DRAW WAVEFORM
// ------------------------------
fileInput.onchange = (e) => {
    if (e.target.files.length > 0) {
        drawWaveform(e.target.files[0]);
        status.innerText = "Ready to upload.";
    }
};


// ------------------------------
// UPLOAD BUTTON CLICK
// ------------------------------
uploadBtn.onclick = () => {
    const file = fileInput.files[0];
    if (!file) {
        status.innerText = "Please select an audio file first.";
        return;
    }

    const form = new FormData();
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/predict", true);

    // Show progress bar
    progress.style.display = "block";

    // Upload progress
    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const p = (e.loaded / e.total) * 100;
            bar.style.width = p + "%";
            status.innerText = `Uploading... ${Math.round(p)}%`;
        }
    };

    // When upload completes
    xhr.onload = () => {
        progress.style.display = "none";
        bar.style.width = "0%";

        if (xhr.status >= 200 && xhr.status < 300) {
            const data = JSON.parse(xhr.responseText);
            showResults(data);
        } else {
            status.innerText = "Upload failed.";
        }
    };

    xhr.onerror = () => {
        progress.style.display = "none";
        status.innerText = "Network error.";
    };

    xhr.send(form);
};


// ------------------------------
// SHOW RESULTS FROM BACKEND
// ------------------------------
function showResults(data) {
    if (data.error) {
        status.innerText = data.error;
        return;
    }

    status.innerText = "Prediction complete.";

    // Fill top predictions
    predlist.innerHTML = "";
    data.predictions.forEach(p => {
        const li = document.createElement("li");
        li.innerHTML = `
            <span>${p.species}</span>
            <span>${(p.confidence * 100).toFixed(1)}%</span>
        `;
        predlist.appendChild(li);
    });

    // Show spectrogram image
    if (data.spectrogram_url) {
        specimg.src = data.spectrogram_url;
        specimg.style.display = "block";
    }

    results.hidden = false;
}


// ------------------------------
// QUICK WAVEFORM DRAWER (PREVIEW)
// ------------------------------
function drawWaveform(file) {
    const reader = new FileReader();

    reader.onload = function (ev) {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

        audioCtx.decodeAudioData(ev.target.result).then(buffer => {
            const raw = buffer.getChannelData(0);

            const canvas = speccanvas;
            const ctx = canvas.getContext("2d");

            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#0b1220";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.lineWidth = 1;
            ctx.strokeStyle = "#60a5fa";
            ctx.beginPath();

            const step = Math.ceil(raw.length / canvas.width);
            const amp = canvas.height / 2;

            for (let i = 0; i < canvas.width; i++) {
                let min = 1.0, max = -1.0;

                for (let j = 0; j < step; j++) {
                    const datum = raw[(i * step) + j];
                    if (datum < min) min = datum;
                    if (datum > max) max = datum;
                }

                ctx.moveTo(i, (1 + min) * amp);
                ctx.lineTo(i, (1 + max) * amp);
            }

            ctx.stroke();
        });
    };

    reader.readAsArrayBuffer(file);
}
