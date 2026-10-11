const chatBox = document.getElementById("chat-box");

function addMessage(message, type) {

    const div = document.createElement("div");

    div.className =
        type === "user"
        ? "user-message"
        : "bot-message";

    div.innerHTML = message;

    chatBox.appendChild(div);

    chatBox.scrollTop = chatBox.scrollHeight;
}

async function uploadPDF() {

    const file =
        document.getElementById("pdfFile").files[0];

    if (!file) {
        alert("Select a PDF first");
        return;
    }

    const formData = new FormData();

    formData.append("pdf", file);

    document.getElementById("uploadStatus")
        .innerHTML = "Uploading PDF...";

    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    document.getElementById("uploadStatus")
        .innerHTML = data.message;

    addMessage(
        "📄 PDF uploaded successfully.",
        "bot"
    );
}

async function sendMessage() {

    const input =
        document.getElementById("message");

    const message = input.value.trim();

    if (!message) return;

    addMessage(message, "user");

    input.value = "";

    document.getElementById("typing")
        .style.display = "block";

    const response = await fetch("/chat", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            message: message
        })

    });

    const data = await response.json();

    document.getElementById("typing")
        .style.display = "none";

    let reply = data.reply;

    if (data.sources) {

        reply +=
            `<br><br><small>📑 Pages: ${data.sources.join(", ")}</small>`;
    }

    addMessage(reply, "bot");
}

function handleKey(event) {

    if (event.key === "Enter") {
        sendMessage();
    }
}
