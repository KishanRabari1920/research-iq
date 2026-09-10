const BASE_URL = "http://127.0.0.1:8000";

// =============================
// Elements
// =============================

const chatBox = document.getElementById("chat");
const queryInput = document.getElementById("user-query-input");
const askBtn = document.getElementById("ask-btn");

const newChatBtn = document.getElementById("new-chat");
const viewPapersBtn = document.getElementById("db-table");

// =============================
// Welcome Screen
// =============================

function createWelcome() {


    const welcome = document.createElement("div");

    welcome.className = "welcome";
    welcome.id = "welcome";

    welcome.innerHTML = `
    <div class="welcome-icon">
        ✦
    </div>

    <h3>
        Explore your research papers
    </h3>

    <p>
        Ask questions about your indexed research papers.
        Research IQ will retrieve relevant information and
        generate an answer from your knowledge base.
    </p>
`;

    chatBox.appendChild(welcome);


}

// =============================
// Append Message
// =============================

function appendMessage(sender, text) {


    const msgDiv = document.createElement("div");

    msgDiv.className =
        sender === "user"
            ? "user-msg"
            : "ai-msg";

    msgDiv.textContent = text;

    chatBox.appendChild(msgDiv);

    chatBox.scrollTop = chatBox.scrollHeight;


}

// =============================
// Loading Message
// =============================

function showLoading() {


    const loadingDiv = document.createElement("div");

    loadingDiv.className = "ai-msg";
    loadingDiv.id = "loading-message";

    loadingDiv.innerHTML = `
    <div class="loading">
        <span></span>
        <span></span>
        <span></span>
    </div>
`;

    chatBox.appendChild(loadingDiv);

    chatBox.scrollTop = chatBox.scrollHeight;


}

// =============================
// Remove Loading
// =============================

function removeLoading() {


    const loading =
        document.getElementById("loading-message");

    if (loading) {
        loading.remove();
    }


}

// =============================
// Ask Question
// =============================

async function askQuestion() {


    const question = queryInput.value.trim();

    // Don't send empty questions
    if (!question) {
        return;
    }


    // Remove welcome screen
    const welcome =
        document.getElementById("welcome");

    if (welcome) {
        welcome.remove();
    }


    // Add user question
    appendMessage("user", question);


    // Clear input
    queryInput.value = "";


    // Disable Ask button
    askBtn.classList.add("loading-btn");

    askBtn.querySelector(".ask-text").textContent = "...";


    // Show loading animation
    showLoading();


    try {

        const res = await fetch(
            `${BASE_URL}/ask?question=${encodeURIComponent(question)}`
        );


        // Check HTTP response
        if (!res.ok) {

            throw new Error(
                `HTTP error: ${res.status}`
            );

        }


        // Convert response to JSON
        const data = await res.json();


        // Remove loading
        removeLoading();


        // Get answer
        const answer =
            data.answer ||
            "I couldn't find a relevant answer in the research papers.";


        // Display answer
        appendMessage("ai", answer);


        // Browser console
        console.log(
            "Research IQ Answer:",
            answer
        );


    } catch (error) {

        console.error(
            "Backend Error:",
            error
        );


        // Remove loading
        removeLoading();


        // Display error
        appendMessage(
            "ai",
            "⚠️ Unable to connect to the Research IQ backend. Please make sure the FastAPI server is running."
        );


    } finally {

        // Enable Ask button
        askBtn.classList.remove("loading-btn");

        askBtn.querySelector(".ask-text").textContent = "Ask";

        // Focus input again
        queryInput.focus();
    }


}

// =============================
// Ask Button
// =============================

askBtn.addEventListener(
    "click",
    askQuestion
);

// =============================
// Enter Key
// =============================

queryInput.addEventListener(
    "keydown",
    (event) => {


        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            askQuestion();
        }

    }


);

// =============================
// New Chat
// =============================

newChatBtn.addEventListener(
    "click",
    () => {


        // Clear all messages
        chatBox.innerHTML = "";


        // Add welcome screen
        createWelcome();


        // Focus input
        queryInput.focus();

    }


);

// =============================
// View Papers
// =============================

viewPapersBtn.addEventListener(
    "click",
    () => {


        window.location.href =
            "db_table.html";

    }


);

// =============================
// Initial Focus
// =============================

queryInput.focus();
