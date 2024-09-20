// static/client.js
const video = document.getElementById('video');
const processedVideo = document.getElementById('processed_video');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');

let intervalId;

startButton.addEventListener('click', startStream);
stopButton.addEventListener('click', stopStream);

function startStream() {
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(localStream => {
                video.srcObject = localStream;
                video.play();
                intervalId = setInterval(processFrame, 100);
            })
            .catch(err => {
                console.log("Something went wrong: " + err);
            });
    }
}

function stopStream() {
    video.pause();
    clearInterval(intervalId);

    let stream = video.srcObject;
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
}

function processFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/video_feed', true);
        xhr.responseType = 'blob';
        xhr.onload = function () {
            if (this.status === 200) {
                const blob = this.response;
                const url = URL.createObjectURL(blob);
                processedVideo.src = url;
            }
        };
        xhr.send(blob);
    }, 'image/jpeg');
}