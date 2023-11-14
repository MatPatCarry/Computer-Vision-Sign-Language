let isRecording = false;
let mediaRecorder;
let recordedBlobs = [];
let fetchDataPromise;

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

        if (responseData) {

            console.log(`responseData: ${responseData}`);
            const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
            const response = await chrome.tabs.sendMessage(tab.id, {prediction: responseData});
            window.close();
            console.log(`resp: ${response}`)
        }
         
    } catch (error) {
        console.log('Error when fetching resource:', error);
        throw error;
    }
}

function handleDataAvailable(event) {
    if (event.data && event.data.size > 0) {
        recordedBlobs.push(event.data);
    }
}

function startRecording(videoElemId) {

    navigator.mediaDevices.getUserMedia({ audio: false, video: true })
    .then(stream => {

        videoElement = document.getElementById(videoElemId);
        videoElement.srcObject = stream;

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
    })
    .catch(err => console.error('Error when starting: ', err));
}

function stopRecording() {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
}

// chrome.action.onClicked.addListener(async () => {
async function main() {

    if (isRecording) {

        console.log(`Recording is active: ${isRecording}`)
        console.log(`Recording set to false`);

        stopRecording();

        mediaRecorder = null;
        recordedBlobs = [];
        console.log(`Recording stoppped`)

        isRecording = false;

    } else {

        const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
        const condResponse = await chrome.tabs.sendMessage(tab.id, {checkCond: "check"});

        console.log(`checkCond: ${condResponse.checkCond}`)

        if (condResponse.checkCond === "ok") {

            console.log(`Recording is NOT active: ${isRecording}`)
            console.log(`Recording set to true`);

            console.log(`Start of recording`)
            startRecording("preview")
            isRecording = true;

        } else {

            const condWarningElement = document.createElement("h2");

            condWarningElement.id = "condWarning";
            condWarningElement.textContent = "Text area not detected!";

            const videoElement = document.getElementById("preview");

            if (videoElement) {
                videoElement.style.display = "none"; // Or you can use videoElement.style.visibility = "hidden"; to hide without affecting layout
            }

            document.body.appendChild(condWarningElement);
        }
    };
};

main();

chrome.runtime.onMessage.addListener(
    async function(request, sender, sendResponse) {

        if (request.recording === "stop") {

            if (isRecording) {

                main();
                sendResponse({popup_msg: "end of popup"});

            } else {
                sendResponse({popup_msg: "not recording"});
            };
        };
    }
);