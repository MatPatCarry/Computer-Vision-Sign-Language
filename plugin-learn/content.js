let isRecording = false;
let videoDivId = null;
let mediaRecorder;
let recordedBlobs = [];

function createVideoDiv() {
    // Create a <div> element to contain the <video> element
    const divElement = document.createElement('div');
    divElement.style.border = '1px solid #000'; // Add a border to the <div>
    divElement.style.padding = '10px'; // Add padding to the <div>

    const divId = 'videoDivMatPat';
    divElement.id = divId;
    // Create a <video> element and set attributes
    const videoElement = document.createElement('video');
    videoElement.setAttribute('autoplay', '');
    videoElement.setAttribute('playsinline', '');
  
    divElement.appendChild(videoElement); // Append the <video> element to the <div>
  
    // Insert the <div> containing the <video> element next to the active element
    const activeElement = document.activeElement;
    if (activeElement) {
      const parentElement = activeElement.parentElement;
      if (parentElement) {
        parentElement.insertBefore(divElement, activeElement.nextSibling);
      } else {
        activeElement.insertAdjacentElement('afterend', divElement);
      }
  
      return divId;
    }
  }

function updateFormData(responseData) {
    document.activeElement.value += " " + responseData.prediction;
}

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
            updateFormData(responseData)
        }
         
    } catch (error) {
        localStorage.setItem('error', error)
        console.log('Error when fetching resource:', error);
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

chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {

        if (request.recording) {
            console.log(`received ${request.recording}`);
         
            if (isRecording) {

                console.log(`Recording is active: ${isRecording}`)
                console.log(`Recording set to false`);

                stopRecording()
                mediaRecorder = null;
                recordedBlobs = [];
                console.log(`Recording stoppped`)

                const videoDivElem = document.getElementById(videoDivId)
                videoDivElem.remove();

                videoDivId = null;
                isRecording = false;

            } else {

                console.log(`Recording is NOT active: ${isRecording}`)
                console.log(`Recording set to true`);

                videoDivId = createVideoDiv();

                console.log(`Start of recording`)
                startRecording(videoDivId)
                isRecording = true;
            }
            
            sendResponse({farewell: "goodbye"});
        }
    }   
);