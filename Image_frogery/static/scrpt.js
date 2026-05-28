function uploadImage() {
    const input = document.getElementById('image-input');
    const file = input.files[0];

    if (!file) {
        alert("Please select an image");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    // Image preview
    let preview = document.querySelector(".o img");
    if (!preview) {
        preview = document.createElement("img");
        document.querySelector(".o div").appendChild(preview);
    }
    preview.src = URL.createObjectURL(file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            let resultText = document.getElementById("result");
            let progressBar = document.querySelector(".pbar1");
            let result = document.querySelector(".prog h1");
            let point = document.querySelector(".pbar2");
            if (data.error === "Invalid file type. Only PNG, JPG, and JPEG are allowed.") {
                resultText.innerHTML = "⚠️ " + data.error;
                resultText.style.color = "red";
            }

            if (data.result === "REAL") {
                resultText.innerHTML = "✅ This image is REAL (" + (data.confidence * 100).toFixed(2) + "%)";
                resultText.style.color = "green";
                if (data.confidence !== null) {
                    result.innerHTML = "Processing Completed! The image is REAL.";
                    result.style.color = "green";
                    progressBar.style.width = (data.confidence * 100) + "%";
                    point.innerHTML = (data.confidence * 100).toFixed(2) + "%";
                    progressBar.style.backgroundColor = "green";
                }
            } else {
                resultText.innerHTML = "⚠️ This image is FORGED (" + (data.confidence * 100).toFixed(2) + "%)";
                resultText.style.color = "red";
                if (data.confidence !== null) {
                    result.innerHTML = "Processing Completed! The image is FORGED.";
                    result.style.color = "red";
                    progressBar.style.width = (data.confidence * 100) + "%";
                    point.innerHTML = (data.confidence * 100).toFixed(2) + "%";
                    progressBar.style.backgroundColor = "red";
                }
            }


        })
        .catch(error => {
            console.error("Error:", error);
        });


}
function upload_old_image() {
    const input = document.getElementById('image-input');
    const file = input.files[0];

    if (!file) {
        alert("Please select an image");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    // Image preview
    let preview = document.querySelector(".o img");
    if (!preview) {
        preview = document.createElement("img");
        document.querySelector(".o div").appendChild(preview);
    }
    preview.src = URL.createObjectURL(file);

    fetch("/predict_old", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            let resultText = document.getElementById("result");
            let progressBar = document.querySelector(".pbar1");
            let result = document.querySelector(".prog h1");
            let point = document.querySelector(".pbar2");
            if (data.error === "Invalid file type. Only PNG, JPG, and JPEG are allowed.") {
                resultText.innerHTML = "⚠️ " + data.error;
                resultText.style.color = "red";
            }

            if (data.result === "REAL") {
                resultText.innerHTML = "✅ This image is REAL (" + (data.confidence * 100).toFixed(2) + "%)";
                resultText.style.color = "green";
                if (data.confidence !== null) {
                    result.innerHTML = "Processing Completed! The image is REAL.";
                    result.style.color = "green";
                    progressBar.style.width = (data.confidence * 100) + "%";
                    point.innerHTML = (data.confidence * 100).toFixed(2) + "%";
                    progressBar.style.backgroundColor = "green";
                }
            } else {
                resultText.innerHTML = "⚠️ This image is FORGED (" + (data.confidence * 100).toFixed(2) + "%)";
                resultText.style.color = "red";
                if (data.confidence !== null) {
                    result.innerHTML = "Processing Completed! The image is FORGED.";
                    result.style.color = "red";
                    progressBar.style.width = (data.confidence * 100) + "%";
                    point.innerHTML = (data.confidence * 100).toFixed(2) + "%";
                    progressBar.style.backgroundColor = "red";
                }
            }


        })
        .catch(error => {
            console.error("Error:", error);
        });
}