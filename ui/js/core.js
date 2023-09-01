const image = document.querySelector("#frame").src
// buttons
const saveSettings = document.querySelector("#saveSettings")
const updateFrame = document.querySelector("#updateFrame")
const saveframe = document.querySelector("#saveframe")
const recordToTape = document.querySelector("#recordToTape")
const defSett = [1.92, 2, -10, 16, 2, 3, 1.7];
const getBase64StringFromDataURL = (dataURL) =>
    dataURL.replace('data:', '').replace(/^.+,/, '');

function saveSet() {
    const name = prompt("Write a preset name");
    if (!name) return; // Exit if the user cancels the prompt

    const settings = {
        "name": name,
        "lumaCompressionRate": parseFloat(document.querySelector("#lumaCompressionRate").value),
        "lumaNoiseMean": parseFloat(document.querySelector("#lumaNoiseMean").value),
        "chromaCompressionRate": parseFloat(document.querySelector("#chromaCompressionRate").value),
        "borderSize": parseFloat(document.querySelector("#borderSize").value),
        "lumaNoiseSigma": parseFloat(document.querySelector("#lumaNoiseSigma").value),
        "generation": parseFloat(document.querySelector("#generations").value)
    };
    
    const jsonString = JSON.stringify(settings);
    saveTextAs(jsonString, name+".json");
}

function update() {
    const base64 = getBase64StringFromDataURL(image);
    let lumaCompressionRate = document.querySelector("#lumaCompressionRate").value
    let lumaNoiseMean = document.querySelector("#lumaNoiseMean").value
    let chromaCompressionRate = document.querySelector("#chromaCompressionRate").value
    let chromaNoiseIntensity = document.querySelector("#chromaNoiseIntensity").value
    let chromaSaturation = document.querySelector("#chromaSaturation").value
    let borderSize = document.querySelector("#borderSize").value
    let lumaNoiseSigma = document.querySelector("#lumaNoiseSigma").value
    let generations = document.querySelector("#generations").value
    let verticalBlur = document.querySelector('#verticalBlur').value
    let horizontalBlur = document.querySelector("#horizontalBlur").value
    eel.updateFrame(base64, parseFloat(lumaCompressionRate), parseFloat(lumaNoiseSigma), parseFloat(lumaNoiseMean), parseFloat(chromaCompressionRate), parseFloat(verticalBlur), parseFloat(horizontalBlur), parseFloat(chromaNoiseIntensity), parseFloat(borderSize), parseFloat(generations))
};
function lockBlur() {
    var cb = document.getElementById('lockValues').checked;
    if(cb) {
        document.querySelector('#verticalBlur').value = document.querySelector('#horizontalBlur').value
    }
}
function lockBlur2() {
    var cb = document.getElementById('lockValues').checked;
    if(cb) {
        document.querySelector('#horizontalBlur').value = document.querySelector('#verticalBlur').value
    }
}
eel.expose(setImage);
function setImage(data) {
    const image = document.getElementById('frame');
    image.src = "data:image/png;base64," + data
}
function saveFr() {
    const base64Image = document.querySelectorAll('img')[0].src;
    const binaryString = atob(base64Image.split(',')[1]);
    // Create a Blob object from binary data
    const blob = new Blob([new Uint8Array([...binaryString].map(char => char.charCodeAt(0)))], { type: 'image/png' });

    // Create a URL for the Blob
    const blobUrl = URL.createObjectURL(blob);

    // Create a link element for download
    const downloadLink = document.createElement('a');
    downloadLink.href = blobUrl;
    downloadLink.download = 'image.png'; // Change the filename as needed

    // Simulate a click on the link to trigger the download
    downloadLink.click();

    // Clean up: remove the temporary Blob URL
    URL.revokeObjectURL(blobUrl);
    alert("frame saved")
}
function recordTape() {
    eel.fileChooser();
    //alert("recording to tape")
}
function reset() {
    let lumaCompressionRate = document.querySelector("#lumaCompressionRate")
    let lumaNoiseMean = document.querySelector("#lumaNoiseMean")
    let chromaCompressionRate = document.querySelector("#chromaCompressionRate")
    let chromaNoiseIntensity = document.querySelector("#chromaNoiseIntensity")
    let chromaSaturation = document.querySelector("#chromaSaturation")
    let borderSize = document.querySelector("#borderSize")
    let lumaNoiseSigma = document.querySelector("#lumaNoiseSigma")
    let generations = document.querySelector("#generations")
    lumaCompressionRate.value = defSett[0]
    lumaNoiseSigma.value = defSett[1]
    lumaNoiseMean.value = defSett[2]
    chromaCompressionRate.value = defSett[3]
    chromaNoiseIntensity.value = defSett[4]
    borderSize.value = defSett[6]
    generations.value = 3
    update()
}
saveSettings.addEventListener("click", saveSet)
saveframe.addEventListener("click", saveFr)
recordToTape.addEventListener("click", recordTape)
document.querySelector("#reset").addEventListener("click", reset)
window.onload = () => {
    reset()
    document.querySelectorAll('input[type="range"]').forEach((input)=> {
        input.addEventListener("change", update)
    })
}
// Function to be called when the window is closed
function onClose() {
    eel.on_window_close();
}
document.querySelector("#horizontalBlur").addEventListener("change", lockBlur)
document.querySelector("#verticalBlur").addEventListener("change", lockBlur2)

// Attach the onClose function to the window's "beforeunload" event
window.addEventListener("beforeunload", onClose);