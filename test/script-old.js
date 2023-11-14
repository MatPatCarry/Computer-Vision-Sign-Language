let mediaRecorder;
let recordedBlobs = [];

async function fetchData(file) {

    const dataFromForm = new FormData();
    dataFromForm.append('file', file);

    const options = {
        method: 'POST',
        body: dataFromForm,
    };

    try {

        const response = await fetch(
            url='http://localhost:5000/get_prediction', 
            options
        )

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const responseData = await response.json();
        console.log("Success:", responseData);

        localStorage.setItem('responseData', JSON.stringify(responseData));
        location.reload();

         
    } catch (error) {

        localStorage.setItem('error', error)
        console.log('Error when fetching resource:', error);
    }
}

document.querySelector('#start').addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
        document.querySelector('video').srcObject = stream;

        if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
        } else {
            mediaRecorder = new MediaRecorder(stream);
        }

        mediaRecorder.onstop = async () => {
            const blob = new Blob(recordedBlobs, { type: 'video/webm' });
            const file = new File([blob], 'myRecording.webm', { type: 'video/webm' });

            try {
                await fetchData(file);
            } catch (error) {
                console.error('Error in fetchData: ', error);
            }
        };

        mediaRecorder.ondataavailable = handleDataAvailable;
        mediaRecorder.start();
        document.querySelector('#start').disabled = true;
        document.querySelector('#stop').disabled = false;
        

    } catch (err) {
        console.error('Error when starting: ', err);
    }
});

document.querySelector('#stop').addEventListener('click', (event) => {
    event.preventDefault();
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    document.querySelector('#start').disabled = false;
    document.querySelector('#stop').disabled = true;
});

function handleDataAvailable(event) {
    if (event.data && event.data.size > 0) {
        recordedBlobs.push(event.data);
    }
}

// Load data from localStorage when content is loaded
let responseDataString = localStorage.getItem('responseData');
let responseData = responseDataString ? JSON.parse(responseDataString) : null;

console.log(`error :${localStorage.getItem('error')}`)
console.log(`resp :${localStorage.getItem('resp')}`)


if (responseData) {
    console.log(`responseData: ${responseData}`)
    document.getElementById("prediction").innerHTML = responseData.prediction;
    document.getElementById("probs").innerHTML = responseData.probs;
}

