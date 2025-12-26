(function () {
  const STORAGE_KEY = "theme";
  const root = document.documentElement;

  function systemTheme() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    const btn = document.getElementById("themeToggle");
    if (btn) {
      btn.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
      btn.textContent = theme === "dark" ? "🌙" : "☀️";
      btn.title = theme === "dark" ? "Dark" : "Light";
    }
  }

  function initTheme() {
    const saved = localStorage.getItem(STORAGE_KEY);
    const theme = saved || systemTheme();
    applyTheme(theme);

    // follow OS if user didn't pick
    if (!saved && window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
        applyTheme(e.matches ? "dark" : "light");
      });
    }
  }

  function toggleTheme() {
    const current = root.getAttribute("data-theme") || systemTheme();
    const next = current === "dark" ? "light" : "dark";
    localStorage.setItem(STORAGE_KEY, next);
    applyTheme(next);
  }

  window.__toggleTheme = toggleTheme;
  window.addEventListener("DOMContentLoaded", initTheme);
})();