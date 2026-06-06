/**
 * RAG PDF Chatbot — Frontend Logic
 * ================================
 * Handles:
 *  - PDF upload via drag-and-drop or file picker
 *  - Sending questions to the backend /ask endpoint
 *  - Rendering user and bot messages
 *  - Displaying source chunks used for answers
 *  - Maintaining chat history (sent to backend each turn)
 */

"use strict";

// ── Config ───────────────────────────────────────────────────────────────────
const API_BASE = "https://rag-lwiy.onrender.com";   // Change if running on a different port

// ── State ────────────────────────────────────────────────────────────────────
let chatHistory = [];      // [{role: "user"|"assistant", content: "..."}]
let isLoading   = false;   // Prevents double-sends while LLM is thinking
let pdfLoaded   = false;   // Whether a PDF has been successfully indexed
let lastSources = [];      // Source chunks from the last bot response

// ── DOM Elements ─────────────────────────────────────────────────────────────
const pdfInput       = document.getElementById("pdfInput");
const uploadZone     = document.getElementById("uploadZone");
const uploadStatus   = document.getElementById("uploadStatus");
const statusFill     = document.getElementById("statusFill");
const statusText     = document.getElementById("statusText");
const fileCard       = document.getElementById("fileCard");
const fileName       = document.getElementById("fileName");
const fileChunks     = document.getElementById("fileChunks");
const resetBtn       = document.getElementById("resetBtn");
const chatContainer  = document.getElementById("chatContainer");
const emptyState     = document.getElementById("emptyState");
const questionInput  = document.getElementById("questionInput");
const sendBtn        = document.getElementById("sendBtn");
const clearChatBtn   = document.getElementById("clearChatBtn");
const pdfBadge       = document.getElementById("pdfBadge");
const pdfBadgeName   = document.getElementById("pdfBadgeName");
const sourcesPanel   = document.getElementById("sourcesPanel");
const sourcesContent = document.getElementById("sourcesContent");
const sourcesToggle  = document.getElementById("sourcesToggle");
const sourcesToggleText = document.getElementById("sourcesToggleText");

// ══════════════════════════════════════════════════════════════════════════════
// PDF UPLOAD
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Click on upload zone → trigger hidden file input
 */
uploadZone.addEventListener("click", () => pdfInput.click());

/**
 * Drag-and-drop support
 */
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.style.borderColor = "var(--accent)";
});
uploadZone.addEventListener("dragleave", () => {
  uploadZone.style.borderColor = "";
});
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.style.borderColor = "";
  const file = e.dataTransfer.files[0];
  if (file && file.type === "application/pdf") {
    handlePDFUpload(file);
  } else {
    showError("Please drop a valid PDF file.");
  }
});

/**
 * File selected via input element
 */
pdfInput.addEventListener("change", () => {
  if (pdfInput.files.length > 0) {
    handlePDFUpload(pdfInput.files[0]);
  }
});

/**
 * Main upload handler: POST the PDF to /upload, show progress, update UI.
 */
async function handlePDFUpload(file) {
  if (!file.name.endsWith(".pdf")) {
    showError("Only PDF files are supported.");
    return;
  }

  // Show progress UI
  uploadZone.classList.add("hidden");
  uploadStatus.classList.remove("hidden");
  fileCard.classList.add("hidden");
  animateStatusBar();
  statusText.textContent = "Uploading PDF…";

  const formData = new FormData();
  formData.append("file", file);

  try {
    statusText.textContent = "Extracting text & building embeddings…";

    const response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Upload failed.");
    }

    // ── Success ──────────────────────────────────────────────────────────────
    statusFill.style.width = "100%";
    statusText.textContent = "✓ PDF indexed successfully!";

    // Update file card
    fileName.textContent   = file.name;
    fileChunks.textContent = `${data.chunks_indexed} chunks indexed`;

    // Update top bar badge
    pdfBadgeName.textContent = file.name;
    pdfBadge.classList.remove("hidden");

    // Enable chat input
    questionInput.disabled = false;
    sendBtn.disabled = false;
    pdfLoaded = true;

    // Transition UI
    setTimeout(() => {
      uploadStatus.classList.add("hidden");
      fileCard.classList.remove("hidden");
      uploadZone.classList.add("hidden");
    }, 800);

  } catch (err) {
    uploadStatus.classList.add("hidden");
    uploadZone.classList.remove("hidden");
    showError(`Upload error: ${err.message}`);
  }
}

/** Animate the status bar from 0% → 85% (final 15% fills on success) */
function animateStatusBar() {
  statusFill.style.width = "0%";
  let progress = 0;
  const interval = setInterval(() => {
    progress += Math.random() * 8;
    if (progress >= 85) { clearInterval(interval); progress = 85; }
    statusFill.style.width = `${progress}%`;
  }, 300);
}

/** Reset button: clear vector store and reset UI */
resetBtn.addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/reset`, { method: "DELETE" });
  } catch (_) { /* ignore if backend is down */ }

  // Reset state
  pdfLoaded = false;
  chatHistory = [];
  lastSources = [];

  // Reset UI
  fileCard.classList.add("hidden");
  pdfBadge.classList.add("hidden");
  uploadZone.classList.remove("hidden");
  uploadStatus.classList.add("hidden");
  statusFill.style.width = "0%";
  questionInput.disabled = true;
  sendBtn.disabled = true;
  pdfInput.value = "";

  // Clear chat
  clearMessages();
  hideSources();
});

// ══════════════════════════════════════════════════════════════════════════════
// CHAT — SENDING QUESTIONS
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Send on Enter key (Shift+Enter = newline)
 */
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

/** Auto-resize textarea as user types */
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + "px";
});

sendBtn.addEventListener("click", sendQuestion);

/**
 * Main question send handler:
 *  1. Read user question
 *  2. Display user message in chat
 *  3. POST to /ask with question + history
 *  4. Display bot response
 *  5. Update chat history
 */
async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question || isLoading || !pdfLoaded) return;

  // ── UI: user message ─────────────────────────────────────────────────────
  isLoading = true;
  questionInput.value = "";
  questionInput.style.height = "auto";
  sendBtn.disabled = true;
  sendBtn.classList.add("loading");
  sendBtn.querySelector("#sendIcon").textContent = "⟳";

  emptyState.classList.add("hidden");   // Hide empty state on first message
  hideSources();

  appendUserMessage(question);

  // Add to history
  chatHistory.push({ role: "user", content: question });

  // ── Thinking indicator ───────────────────────────────────────────────────
  const thinkingEl = appendThinkingIndicator();

  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        chat_history: chatHistory.slice(-10)   // Send last 10 turns max
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Failed to get answer.");
    }

    // ── Bot answer ─────────────────────────────────────────────────────────
    thinkingEl.remove();
    appendBotMessage(data.answer);

    // Update history with bot reply
    chatHistory.push({ role: "assistant", content: data.answer });

    // Show source chunks
    lastSources = data.source_chunks || [];
    if (lastSources.length > 0) {
      showSources(lastSources);
    }

  } catch (err) {
    thinkingEl.remove();
    appendErrorMessage(`Error: ${err.message}`);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    sendBtn.classList.remove("loading");
    sendBtn.querySelector("#sendIcon").textContent = "↑";
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// MESSAGE RENDERING
// ══════════════════════════════════════════════════════════════════════════════

function appendUserMessage(text) {
  const el = createMessageEl("user", "You", text);
  chatContainer.appendChild(el);
  scrollToBottom();
}

function appendBotMessage(text) {
  const el = createMessageEl("bot", "AI", text);
  chatContainer.appendChild(el);
  scrollToBottom();
}

function appendErrorMessage(text) {
  const el = createMessageEl("bot", "AI", `<span class="error-text">${escapeHTML(text)}</span>`, true);
  chatContainer.appendChild(el);
  scrollToBottom();
}

function appendThinkingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot";
  wrapper.innerHTML = `
    <div class="avatar">⬡</div>
    <div class="bubble thinking-bubble">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </div>`;
  chatContainer.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

/**
 * Create a message bubble element.
 * @param {"user"|"bot"} role
 * @param {string}       label   Avatar label
 * @param {string}       text    Message text (may contain HTML for bot messages)
 * @param {boolean}      rawHTML Whether to set innerHTML directly (for errors)
 */
function createMessageEl(role, label, text, rawHTML = false) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatarChar = role === "user" ? "U" : "⬡";
  const content    = rawHTML ? text : formatMessage(escapeHTML(text));

  wrapper.innerHTML = `
    <div class="avatar">${avatarChar}</div>
    <div class="bubble">${content}</div>`;
  return wrapper;
}

/**
 * Format a plain-text bot message: turn newlines into <br>, bold **text**.
 */
function formatMessage(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")    // **bold**
    .replace(/\n/g, "<br/>");                             // newlines
}

/** Escape HTML to prevent XSS */
function escapeHTML(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function clearMessages() {
  // Remove all message elements (keep emptyState)
  [...chatContainer.children].forEach(child => {
    if (child.id !== "emptyState") child.remove();
  });
  emptyState.classList.remove("hidden");
  chatHistory = [];
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ══════════════════════════════════════════════════════════════════════════════
// SOURCE CHUNKS PANEL
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Populate and open the sources panel below the chat.
 */
function showSources(chunks) {
  sourcesContent.innerHTML = "";
  chunks.forEach((chunk, i) => {
    const div = document.createElement("div");
    div.className = "source-chunk";
    div.textContent = `[Chunk ${i + 1}]\n${chunk}`;
    sourcesContent.appendChild(div);
  });

  sourcesToggleText.textContent = `▸ View source chunks (${chunks.length})`;
  // Don't auto-open — let user click
}

function hideSources() {
  sourcesPanel.classList.remove("open");
  sourcesContent.innerHTML = "";
  sourcesToggleText.textContent = "▸ View source chunks (0)";
}

/** Toggle the sources panel open/closed */
function toggleSources() {
  const isOpen = sourcesPanel.classList.toggle("open");
  const arrow  = isOpen ? "▾" : "▸";
  const count  = lastSources.length;
  sourcesToggleText.textContent = `${arrow} View source chunks (${count})`;
}

// ══════════════════════════════════════════════════════════════════════════════
// UTILITY
// ══════════════════════════════════════════════════════════════════════════════

clearChatBtn.addEventListener("click", () => {
  clearMessages();
  hideSources();
  chatHistory = [];
});

/** Fill input with an example query from the empty-state pills */
function fillExample(el) {
  questionInput.value = el.textContent;
  questionInput.dispatchEvent(new Event("input"));   // trigger auto-resize
  questionInput.focus();
}

function showError(msg) {
  // Simple alert — upgrade to toast in production
  alert(msg);
}

// ── Expose globals needed by inline onclick handlers ─────────────────────────
window.toggleSources = toggleSources;
window.fillExample   = fillExample;
