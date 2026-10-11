const video = document.getElementById("video");
const status = document.getElementById("status");

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
    status.innerText = "Scanning face...";
});

let processing = false;

setInterval(() => {

    if (processing) return;

    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 300;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg");

    processing = true;

    fetch("/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
    })
    .then(res => res.json())
    .then(data => {

        if (data.status === "success") {
            status.innerText = "Access Granted";
            status.style.color = "green";

            setTimeout(() => {
                window.location.href = "/admin";
            }, 800);
        }

        // 🔥 ADD THIS (MISSING PART)
        else if (data.status === "denied") {
            status.innerText = "Access Denied";
            status.style.color = "red";
        }

        else {
            status.innerText = "Scanning...";
            status.style.color = "black";
        }

        processing = false;
    })
    .catch(() => {
        processing = false;
        status.innerText = "Error connecting to server";
        status.style.color = "orange";
    });

}, 1000);