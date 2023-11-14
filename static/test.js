let previouslyFocusedInput = null;

document.addEventListener('keydown', function(event) {
  if (event.ctrlKey && event.altKey && event.key === 'q') {
    // Store a reference to the currently focused input element
    previouslyFocusedInput = document.activeElement;
    console.log(previouslyFocusedInput)
    console.log(previouslyFocusedInput.tagName)

    previouslyFocusedInput.value = 13 
    // Send a message to the background script to start recording
    // chrome.runtime.sendMessage({ action: 'startRecording' });
  }
});

window.addEventListener('focus', function() {
    lastFocusedElement = document.activeElement;
    console.log(lastFocusedElement)
}, true);