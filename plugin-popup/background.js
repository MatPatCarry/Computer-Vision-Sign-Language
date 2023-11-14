chrome.commands.onCommand.addListener(async (command) => {

    if (command === 'stop') {
    
        const response = await chrome.runtime.sendMessage({recording: "stop"});
        console.log(response.popup_msg);
    }
});
