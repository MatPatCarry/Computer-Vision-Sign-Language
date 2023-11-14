chrome.action.onClicked.addListener(async () => {

    const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
    const response = await chrome.tabs.sendMessage(tab.id, {recording: "hello"});
    console.log(response);
    
});