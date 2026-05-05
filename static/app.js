/* SSE client for /summarize-stream + in-place result rendering. */
(function () {
  const I18N = window.I18N || {};
  const UI_LANG = window.UI_LANG || "en";
  const isRTL = UI_LANG === "ar";

  const overlay = document.getElementById("loadingOverlay");
  const loadingText = document.getElementById("loadingText");
  const cancelBtn = document.getElementById("cancelBtn");
  const formSection = document.getElementById("formSection");
  const resultSection = document.getElementById("resultSection");
  const form = document.getElementById("summarizeForm");
  const errorHost = document.getElementById("formErrorHost");

  let abortController = null;
  let quillInstance = null;

  function tr(key) { return I18N[key] || key; }
  function showOverlay() { if (overlay) overlay.setAttribute("style", "display: flex !important;"); }
  function hideOverlay() { if (overlay) overlay.setAttribute("style", "display: none !important;"); }
  function setStatus(text) { if (loadingText) loadingText.textContent = text; }
  function clearError() { if (errorHost) errorHost.innerHTML = ""; }
  function showError(message) {
    if (!errorHost) return;
    errorHost.innerHTML = `<div class="alert mb-3"><div class="fw-semibold">${escapeHtml(message)}</div></div>`;
  }
  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function wordCount(s) {
    return (s || "").trim().split(/\s+/).filter(Boolean).length;
  }

  function looksLikeHtml(s) {
    return /<\/?(p|h[1-6]|ul|ol|li|strong|em|b|i|br|span|div|mark)\b/i.test(s || "");
  }

  function markdownToHtml(text) {
    if (!text) return "";
    if (looksLikeHtml(text)) return text;
    if (typeof window.marked === "undefined") {
      // Fallback: preserve newlines.
      return escapeHtml(text).replace(/\n/g, "<br>");
    }
    try {
      window.marked.setOptions({ gfm: true, breaks: true });
      const html = window.marked.parse(text);
      if (typeof window.DOMPurify !== "undefined") {
        return window.DOMPurify.sanitize(html);
      }
      return html;
    } catch (e) {
      return escapeHtml(text).replace(/\n/g, "<br>");
    }
  }

  function statusFromEvent(evt) {
    switch (evt.step) {
      case "received":    return tr("status_received");
      case "extracting":  return tr("status_extracting");
      case "extracted":   return tr("status_extracted");
      case "summarizing":
        return tr("status_summarizing")
          .replace("{i}", evt.chunk).replace("{n}", evt.total);
      case "merging":     return tr("status_merging");
      case "done":        return tr("status_done");
      case "cancelled":   return tr("cancelled");
      default:            return tr("loading");
    }
  }

  async function streamSummarize(formData) {
    abortController = new AbortController();
    clearError();
    setStatus(tr("loading"));
    showOverlay();

    let resp;
    try {
      resp = await fetch("/summarize-stream", {
        method: "POST",
        body: formData,
        signal: abortController.signal,
      });
    } catch (e) {
      if (e.name === "AbortError") { hideOverlay(); return; }
      hideOverlay();
      showError(String(e));
      return;
    }

    if (!resp.ok || !resp.body) {
      hideOverlay();
      showError(tr("error_ollama"));
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    const original = (formData.get("input_text") || "").trim();
    let originalForResult = original;
    let finished = false;

    try {
      while (!finished) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const raw = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const line = raw.split("\n").find(l => l.startsWith("data:"));
          if (!line) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          let evt;
          try { evt = JSON.parse(payload); } catch { continue; }
          setStatus(statusFromEvent(evt));

          if (evt.step === "extracted" && evt.chars) {
            // We only know extracted size; original text will be revealed via "done".
          }
          if (evt.step === "done") {
            finished = true;
            hideOverlay();
            const finalOriginal = (evt.original != null && evt.original !== "")
              ? evt.original
              : originalForResult;
            renderResult({ original: finalOriginal, summary: evt.summary || "" });
            break;
          }
          if (evt.step === "error") {
            finished = true;
            hideOverlay();
            showError(evt.message || tr("error_ollama"));
            break;
          }
          if (evt.step === "cancelled") {
            finished = true;
            hideOverlay();
            break;
          }
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        showError(String(e));
      }
      hideOverlay();
    }
  }

  function renderResult({ original, summary }) {
    formSection.style.display = "none";
    resultSection.style.display = "block";

    const wcOriginal = wordCount(original);
    const summaryHtml = markdownToHtml(summary);
    const summaryPlain = summaryHtml.replace(/<[^>]*>/g, " ");
    const wcSummary = wordCount(summaryPlain);

    resultSection.innerHTML = `
      <div class="d-flex align-items-center justify-content-between mb-3">
        <div class="section-title">${escapeHtml(tr("result_title"))}</div>
        <div class="d-flex gap-2">
          <button id="toggleOriginal" type="button" class="btn btn-ghost btn-sm">
            <span>${escapeHtml(tr("original_text"))}</span>
          </button>
          <form id="exportPdfForm" action="/export-pdf" method="post" style="display:inline;">
            <input type="hidden" name="summary" id="exportSummary" value=""/>
            <input type="hidden" name="original" value="${escapeHtml(original)}"/>
            <input type="hidden" name="ui_lang" value="${escapeHtml(UI_LANG)}"/>
            <button type="submit" class="btn btn-ghost btn-sm text-danger" title="${escapeHtml(tr("export_pdf"))}">
              ${escapeHtml(tr("export_pdf"))}
            </button>
          </form>
          <button id="newSummaryBtn" type="button" class="btn btn-ghost btn-sm">
            ${escapeHtml(tr("new_summary"))}
          </button>
        </div>
      </div>

      <div class="row g-4" id="resultRow">
        <div class="col-lg-6" id="originalCol">
          <div class="card glass-card shadow-sm">
            <div class="card-body p-4">
              <div class="d-flex align-items-center justify-content-between mb-2">
                <div class="section-title">${escapeHtml(tr("original_text"))}</div>
                <div class="text-muted small">${escapeHtml(tr("words"))}: ${wcOriginal}</div>
              </div>
              <pre id="originalPre"></pre>
            </div>
          </div>
        </div>
        <div class="col-lg-6" id="summaryCol">
          <div class="card glass-card shadow-sm">
            <div class="card-body p-4">
              <div class="d-flex align-items-center justify-content-between mb-2">
                <div class="section-title">${escapeHtml(tr("summary"))}</div>
                <div class="text-muted small" id="summaryWordCount">${escapeHtml(tr("words"))}: ${wcSummary}</div>
              </div>
              <div id="summaryEditor" class="ql-editor-content"></div>
            </div>
          </div>
        </div>
      </div>
    `;

    document.getElementById("originalPre").textContent = original;
    const editorEl = document.getElementById("summaryEditor");
    editorEl.innerHTML = summaryHtml;

    quillInstance = new Quill("#summaryEditor", {
      theme: "snow",
      modules: {
        toolbar: [
          ["bold", "italic", "underline", "strike"],
          [{ header: [1, 2, 3, false] }],
          [{ list: "ordered" }, { list: "bullet" }],
          [{ color: [] }, { background: [] }],
          [{ align: [] }],
          ["clean"],
        ],
      },
    });

    const exportSummary = document.getElementById("exportSummary");
    const wcEl = document.getElementById("summaryWordCount");
    if (exportSummary) exportSummary.value = quillInstance.root.innerHTML;

    quillInstance.on("text-change", () => {
      const html = quillInstance.root.innerHTML;
      if (exportSummary) exportSummary.value = html;
      const text = quillInstance.getText().trim();
      const count = text ? text.split(/\s+/).filter(Boolean).length : 0;
      if (wcEl) wcEl.textContent = `${tr("words")}: ${count}`;
    });

    // Toggle original visibility
    const toggleBtn = document.getElementById("toggleOriginal");
    const originalCol = document.getElementById("originalCol");
    const summaryCol = document.getElementById("summaryCol");
    toggleBtn.addEventListener("click", () => {
      const hidden = originalCol.style.display === "none";
      if (hidden) {
        originalCol.style.display = "block";
        summaryCol.classList.replace("col-lg-12", "col-lg-6");
      } else {
        originalCol.style.display = "none";
        summaryCol.classList.replace("col-lg-6", "col-lg-12");
      }
      if (quillInstance && quillInstance.update) quillInstance.update();
    });

    // New summary
    document.getElementById("newSummaryBtn").addEventListener("click", () => {
      resultSection.innerHTML = "";
      resultSection.style.display = "none";
      formSection.style.display = "";
      quillInstance = null;
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  function supportsStreaming() {
    return typeof window.fetch === "function"
      && typeof window.ReadableStream === "function"
      && typeof window.AbortController === "function";
  }

  if (form && supportsStreaming()) {
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const fd = new FormData(form);
      streamSummarize(fd);
    });
  } else if (form) {
    // Fallback: native form post to /summarize. Just show overlay.
    form.addEventListener("submit", () => {
      setStatus(tr("loading"));
      showOverlay();
    });
  }

  if (cancelBtn) {
    cancelBtn.addEventListener("click", () => {
      if (abortController) abortController.abort();
      hideOverlay();
    });
  }
})();
