function myalert() {
    alert("Welcome")
}

function checkUpload() {
    var name = document.getElementById('audiofile');
    if (name.files.length == 0) {
        alert("Please select a file to upload !!!");
        return false;
    }
}


function checkFileExtension() {
    fileName = document.querySelector('#audiofile').value;
    extension = fileName.split('.').pop();
    if (extension == "wav"){
        return true;
    }else{
        alert("Select .wav files only!");
        return false;
    }

};
