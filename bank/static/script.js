function addMessage(text, type) {
    let box = document.getElementById("chatbox");

    let div = document.createElement("div");
    div.className = "msg " + type;
    div.innerText = text;

    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

function sendMessage(msg = null) {
    let input = document.getElementById("input");

    let message = msg || input.value;
    if (!message) return;

    addMessage("🧑 " + message, "user");

    showTyping(); // 🔥 START ANIMATION

    fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message})
    })
    .then(res => res.json())
    .then(data => {

        hideTyping(); // 🔥 STOP ANIMATION

        addMessage("🏦 " + data.text, "bot");
        showCards(data.options);
    });

    input.value = "";
}

function showCards(options) {
    let container = document.getElementById("cards");
    container.innerHTML = "";

    if (!options) return;

    options.forEach(opt => {
        let card = document.createElement("div");
        card.className = "course-card";

        card.innerHTML = `
            <div class="course-title">${opt}</div>
            <div class="course-sub">Click to explore</div>
        `;

        card.onclick = () => sendMessage(opt);

        container.appendChild(card);
    });
}
function showTyping() {
    document.getElementById("typing").classList.remove("hidden");
}

function hideTyping() {
    document.getElementById("typing").classList.add("hidden");
}
function showOptions(options) {
    let box = document.getElementById("options");

    // 🔥 IMPORTANT: clear old buttons
    box.innerHTML = "";

    if (!options || options.length === 0) return;

    options.forEach(opt => {
        let btn = document.createElement("button");
        btn.innerText = opt;

        btn.onclick = () => sendMessage(opt);

        box.appendChild(btn);
    });
}

// auto start
window.onload = function () {
    sendMessage("hi");
};