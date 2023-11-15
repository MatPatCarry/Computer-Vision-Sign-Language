function updateFormData(responseData) {

    const responseMappings = {
        "hello": "Hi, I am writing to you, because I got your message.",
        "love": "Much love Piotrek",
        "please": "Please contact me ASAP.",
        "what": "What did you mean? I did not understand.",
        "thank you": "Thank you in advance for your fast reply.",
    }; 

    const activeElement = document.activeElement;

    const predictionText = responseMappings.hasOwnProperty(responseData.prediction) 
        ? responseMappings[responseData.prediction] : null;

    if (predictionText) {

        let currentIndex = 0;

        if (activeElement.value || activeElement.textContent) {
            
            if (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA') {
                activeElement.value += ' ';
            } else {
                activeElement.textContent += ' ';
            }
        }

        const intervalId = setInterval(function () {

            if (currentIndex < predictionText.length) {

                if (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA') {
                    activeElement.value += predictionText[currentIndex];
                } else {
                    activeElement.textContent += predictionText[currentIndex];
                }

                currentIndex++;

            } else {
                clearInterval(intervalId); // Stop the interval when all characters are filled
            }

            const range = document.createRange();
            const sel = window.getSelection();

            range.setStart(activeElement.childNodes[0], activeElement.textContent.length);
            range.collapse(true);
            
            sel.removeAllRanges();
            sel.addRange(range);

        }, 10);
    }
}

function checkIfActive() {
    return document.activeElement ? document.activeElement : null
}

function checkIfForm(activeElement) {
    return activeElement.tagName === 'INPUT' 
        || activeElement.tagName === 'TEXTAREA'
        || activeElement.classList.contains('LW-avf') ? true : false
}

function checkInputConditions() {

    activeElement = checkIfActive();
    
    if (!activeElement) {
        return false;
    };

    return checkIfForm(activeElement) ? true : false;

};

chrome.runtime.onMessage.addListener(

    function(request, sender, sendResponse) {

        if (request.prediction) {

            console.log(`received ${request.prediction}`);
            console.log(`received ${JSON.stringify(request.prediction)}`)

            updateFormData(request.prediction)
            sendResponse({farewell: "goodbye"});

        }

        else if (request.checkCond) {

            if (checkInputConditions()) {
                sendResponse({checkCond: "ok"})
            } else {
                sendResponse({checkCond: "not"})
            }
        }
    }
);