const API_BASE = BACKEND_URL;
const POLL_INTERVAL_MS = 3000;

const form = document.getElementById("search-form");
const queryInput = document.getElementById("query-input");
const submitBtn = document.getElementById("submit-btn");
const statusArea = document.getElementById("status-area");
const statusText = document.getElementById("status-text");
const elapsedTimeEl = document.getElementById("elapsed-time");
const errorArea = document.getElementById("error-area");
const errorText = document.getElementById("error-text");
const resultsArea = document.getElementById("results-area");
const synthesizedList = document.getElementById("synthesized-list");
const stage2List = document.getElementById("stage2-list");
const stage1List = document.getElementById("stage1-list");

let pollTimer = null;
let searchStartTime = null;

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    resetUI();
    setSubmitting(true);
    showStatus("Submitting search...");

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            throw new Error(`Search request failed (status ${response.status}).`);
        }

        const data = await response.json();
        searchStartTime = Date.now();
        showStatus("Search running -- this can take 1-2 minutes...");
        startPolling(data.job_id);

    } catch (err) {
        showError(err.message || "Failed to submit search.");
        setSubmitting(false);
    }
});

function startPolling(jobId) {
    pollTimer = setInterval(() => pollJobStatus(jobId), POLL_INTERVAL_MS);
    pollJobStatus(jobId); // check immediately too, don't wait for the first interval
}

async function pollJobStatus(jobId) {
    updateElapsedTime();

    try {
        const response = await fetch(`${API_BASE}/search/${jobId}`);
        if (!response.ok) {
            throw new Error(`Status check failed (status ${response.status}).`);
        }

        const job = await response.json();

        if (job.status === "complete") {
            stopPolling();
            setSubmitting(false);
            hideStatus();
            renderResults(job.result);
        } else if (job.status === "failed") {
            stopPolling();
            setSubmitting(false);
            hideStatus();
            showError(job.error || "Search failed unexpectedly.");
        }
        // if "pending" or "running", just keep polling -- no action needed

    } catch (err) {
        stopPolling();
        setSubmitting(false);
        hideStatus();
        showError("Lost connection while checking search status.");
    }
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

function updateElapsedTime() {
    if (!searchStartTime) return;
    const seconds = Math.floor((Date.now() - searchStartTime) / 1000);
    elapsedTimeEl.textContent = `${seconds}s elapsed`;
}

function renderResults(result) {
    resultsArea.classList.remove("hidden");

    // Synthesized results -- the main event, full verdict detail
    synthesizedList.innerHTML = "";
    if (result.results.length === 0) {
        synthesizedList.innerHTML = "<p>No candidates reached full synthesis.</p>";
    }
    result.results.forEach((r, i) => {
        synthesizedList.appendChild(renderSynthesizedCard(r, i));
    });

    // Stage 2 candidates -- score-only, excluding ones already shown above
    const synthesizedIds = new Set(result.results.map((r) => r.patent_id));
    const remainingStage2 = result.stage2_candidates.filter(
        (c) => !synthesizedIds.has(c.patent_id)
    );
    stage2List.innerHTML = "";
    if (remainingStage2.length === 0) {
        document.getElementById("stage2-results").classList.add("hidden");
    } else {
        document.getElementById("stage2-results").classList.remove("hidden");
        remainingStage2.forEach((c, i) => stage2List.appendChild(renderCandidateCard(c, i)));
    }

    // Stage 1 candidates -- broadest, abstract-only matches
    stage1List.innerHTML = "";
    if (result.stage1_candidates.length === 0) {
        document.getElementById("stage1-results").classList.add("hidden");
    } else {
        document.getElementById("stage1-results").classList.remove("hidden");
        result.stage1_candidates.forEach((c, i) => stage1List.appendChild(renderCandidateCard(c, i)));
    }
}

function renderSynthesizedCard(r, i) {
    const card = document.createElement("div");
    card.className = "candidate-card";
    card.style.setProperty("--i", i);

    if (r.error) {
        card.innerHTML = `
            <h3>${escapeHtml(r.title)}</h3>
            <div class="candidate-meta">PATENT ${escapeHtml(r.patent_id)} &middot; RANK ${String(r.rank).padStart(2, "0")}</div>
            <p class="candidate-error">Synthesis failed for this candidate: ${escapeHtml(r.error)}</p>
        `;
        return card;
    }

    const s = r.synthesis;
    const verdictClass = {
        HIGH_RELEVANCE: "verdict-high",
        POSSIBLE_RELEVANCE: "verdict-possible",
        LOW_RELEVANCE: "verdict-low",
        INSUFFICIENT_EVIDENCE: "verdict-insufficient",
    }[s.verdict] || "verdict-low";

    card.innerHTML = `
        <h3>${escapeHtml(r.title)}</h3>
        <div class="candidate-meta">
            PATENT ${escapeHtml(r.patent_id)} &middot; RANK ${String(r.rank).padStart(2, "0")} &middot;
            ABSTRACT ${r.abstract_score.toFixed(2)} &middot; CLAIMS ${r.claim_score.toFixed(2)} &middot; FINAL ${r.final_score.toFixed(2)}
        </div>
        <span class="verdict-badge ${verdictClass}">${s.verdict.replace(/_/g, " ")}</span>
        <div class="candidate-detail">
            <strong>Overlap</strong>
            ${escapeHtml(s.overlap_summary)}
        </div>
        <div class="candidate-detail">
            <strong>Key difference</strong>
            ${escapeHtml(s.key_difference)}
        </div>
        <div class="evidence-quote">
            &ldquo;${escapeHtml(s.supporting_evidence.quote)}&rdquo; &mdash; ${escapeHtml(s.supporting_evidence.source)}
        </div>
    `;
    return card;
}

function renderCandidateCard(c, i) {
    const card = document.createElement("div");
    card.className = "candidate-card";
    card.style.setProperty("--i", i);
    card.innerHTML = `
        <h3>${escapeHtml(c.title)}</h3>
        <div class="candidate-meta">PATENT ${escapeHtml(c.patent_id)} &middot; SCORE ${c.score.toFixed(2)}</div>
    `;
    return card;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function resetUI() {
    hideStatus();
    hideError();
    resultsArea.classList.add("hidden");
}

function setSubmitting(isSubmitting) {
    submitBtn.disabled = isSubmitting;
    submitBtn.textContent = isSubmitting ? "Searching..." : "Search";
}

function showStatus(text) {
    statusArea.classList.remove("hidden");
    statusText.textContent = text;
}

function hideStatus() {
    statusArea.classList.add("hidden");
}

function showError(message) {
    errorArea.classList.remove("hidden");
    errorText.textContent = message;
}

function hideError() {
    errorArea.classList.add("hidden");
}
