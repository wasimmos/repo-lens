/**
 * RepoLens Web & Mobile Single Page Application
 * Zero external JS dependencies.
 */

(function () {
  "use strict";

  // App State
  let state = {
    activeTab: "chat",
    status: {},
    repos: [],
    providers: [],
    hybrid: false,
    selectedProvider: "ollama",
    isAsking: false,
  };

  // DOM Elements
  const elements = {
    // Navigation
    desktopTabs: document.querySelectorAll(".nav-tab-btn"),
    bottomTabs: document.querySelectorAll(".bottom-nav-item"),
    views: document.querySelectorAll(".view-container"),

    // Header Badges
    badgeRepo: document.getElementById("badge-repo"),
    labelActiveRepo: document.getElementById("label-active-repo"),
    badgeProvider: document.getElementById("badge-provider"),
    labelActiveProvider: document.getElementById("label-active-provider"),
    btnPhone: document.getElementById("btn-phone"),

    // Chat
    chatHistory: document.getElementById("chat-history"),
    chatWelcome: document.getElementById("chat-welcome"),
    chatInput: document.getElementById("chat-input"),
    btnSend: document.getElementById("btn-send"),
    btnToggleHybrid: document.getElementById("btn-toggle-hybrid"),
    selectTopK: document.getElementById("select-top-k"),
    starterChips: document.querySelectorAll(".chip"),

    // Search
    searchInput: document.getElementById("search-input"),
    btnSearchExec: document.getElementById("btn-search-exec"),
    searchResults: document.getElementById("search-results"),

    // Repos
    ingestSource: document.getElementById("ingest-source"),
    ingestPythonOnly: document.getElementById("ingest-python-only"),
    ingestForce: document.getElementById("ingest-force"),
    btnIngest: document.getElementById("btn-ingest"),
    btnIngestText: document.getElementById("btn-ingest-text"),
    reposList: document.getElementById("repos-list"),

    // Settings
    providerGrid: document.getElementById("provider-grid"),
    configProviderName: document.getElementById("config-provider-name"),
    cfgModel: document.getElementById("cfg-model"),
    cfgKey: document.getElementById("cfg-key"),
    cfgUrl: document.getElementById("cfg-url"),
    cfgKeyGroup: document.getElementById("cfg-key-group"),
    cfgUrlGroup: document.getElementById("cfg-url-group"),
    btnSaveProvider: document.getElementById("btn-save-provider"),
    settingsLanUrl: document.getElementById("settings-lan-url"),
    btnCopyUrl: document.getElementById("btn-copy-url"),

    // Modal
    modalLan: document.getElementById("modal-lan"),
    modalLanUrl: document.getElementById("modal-lan-url"),
    btnModalCopy: document.getElementById("btn-modal-copy"),
    btnModalClose: document.getElementById("btn-modal-close"),
  };

  // -------------------------------------------------------------------------
  // API Helpers
  // -------------------------------------------------------------------------

  async function apiGet(endpoint) {
    const res = await fetch(endpoint);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `GET ${endpoint} failed (${res.status})`);
    }
    return res.json();
  }

  async function apiPost(endpoint, data) {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(json.error || `POST ${endpoint} failed (${res.status})`);
    }
    return json;
  }

  async function apiDelete(endpoint) {
    const res = await fetch(endpoint, { method: "DELETE" });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(json.error || `DELETE ${endpoint} failed (${res.status})`);
    }
    return json;
  }

  // -------------------------------------------------------------------------
  // Navigation & Tabs
  // -------------------------------------------------------------------------

  function switchTab(tabName) {
    state.activeTab = tabName;

    elements.desktopTabs.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });

    elements.bottomTabs.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });

    elements.views.forEach((v) => {
      v.classList.toggle("active", v.id === `view-${tabName}`);
    });

    if (tabName === "repos") loadRepos();
    if (tabName === "settings") loadProviders();
    if (tabName === "chat") {
      setTimeout(() => elements.chatInput.focus(), 100);
    }
  }

  // -------------------------------------------------------------------------
  // Status & Init
  // -------------------------------------------------------------------------

  async function refreshStatus() {
    try {
      const data = await apiGet("/api/status");
      state.status = data;

      elements.labelActiveRepo.textContent = data.active_repo || "No repo selected";
      elements.labelActiveProvider.textContent = `${data.active_provider} (${data.active_model})`;

      const mobileUrl = `http://${data.lan_ip}:${data.server_port}`;
      elements.settingsLanUrl.textContent = mobileUrl;
      elements.modalLanUrl.textContent = mobileUrl;
    } catch (err) {
      console.error("Status load error:", err);
    }
  }

  // -------------------------------------------------------------------------
  // Markdown & Citation Formatter
  // -------------------------------------------------------------------------

  function renderMarkdown(raw) {
    if (!raw) return "";
    let text = raw;

    // Code blocks
    text = text.replace(/```([a-zA-Z0-9_\-]+)?\n([\s\S]*?)```/g, function (_, lang, code) {
      return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
    });

    // Inline code
    text = text.replace(/`([^`\n]+)`/g, function (_, code) {
      return `<code>${escapeHtml(code)}</code>`;
    });

    // Citations markdown links: [file:line-range](url)
    text = text.replace(/\[([^:\[\]\n]+:\d+-\d+)\]\(([^\)]+)\)/g, function (_, label, url) {
      return `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="citation-pill">📄 ${escapeHtml(label)}</a>`;
    });

    // Bracket citations: [file:line-range]
    text = text.replace(/\[([^:\[\]\n]+:\d+-\d+)\]/g, function (_, label) {
      return `<span class="citation-pill">📄 ${escapeHtml(label)}</span>`;
    });

    // Bold
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic
    text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Bullet points
    text = text.replace(/^\s*[-*]\s+(.+)$/gm, "<li>$1</li>");
    text = text.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");

    // Paragraph line breaks
    text = text.replace(/\n\n+/g, "</p><p>");
    text = text.replace(/\n/g, "<br>");

    return `<p>${text}</p>`;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // -------------------------------------------------------------------------
  // Chat Logic
  // -------------------------------------------------------------------------

  function appendMessage(role, content, evidence = []) {
    if (elements.chatWelcome) {
      elements.chatWelcome.style.display = "none";
    }

    const msgEl = document.createElement("div");
    msgEl.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = role === "user" ? "👤" : "🤖";

    const bubble = document.createElement("div");
    bubble.className = "msg-content";
    bubble.innerHTML = renderMarkdown(content);

    // Evidence Accordion
    if (evidence && evidence.length > 0) {
      const evBox = document.createElement("div");
      evBox.className = "evidence-box";

      const evHeader = document.createElement("div");
      evHeader.className = "evidence-header";
      evHeader.innerHTML = `<span>📚 Retrieved Passages (${evidence.length})</span> <span>▼</span>`;

      const evList = document.createElement("div");
      evList.className = "evidence-list";

      evidence.forEach((item) => {
        const itemEl = document.createElement("div");
        itemEl.className = "evidence-item";

        const locationHtml = item.link
          ? `<a class="evidence-link" href="${escapeHtml(item.link)}" target="_blank" rel="noopener">${escapeHtml(item.location)}</a>`
          : `<span class="evidence-link">${escapeHtml(item.location)}</span>`;

        itemEl.innerHTML = `
          <div class="evidence-item-top">
            <strong>${escapeHtml(item.name)}</strong>
            <span style="color: var(--accent-cyan); font-size: 0.72rem;">Score: ${item.score}</span>
          </div>
          <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">
            ${locationHtml} • <em>${escapeHtml(item.kind)}</em>
          </div>
          ${item.docstring ? `<div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 4px;">${escapeHtml(item.docstring)}</div>` : ""}
          <pre class="evidence-code">${escapeHtml(item.code || "")}</pre>
        `;
        evList.appendChild(itemEl);
      });

      evHeader.addEventListener("click", () => {
        const isOpen = evList.classList.toggle("open");
        evHeader.querySelector("span:last-child").textContent = isOpen ? "▲" : "▼";
      });

      evBox.appendChild(evHeader);
      evBox.appendChild(evList);
      bubble.appendChild(evBox);
    }

    msgEl.appendChild(avatar);
    msgEl.appendChild(bubble);
    elements.chatHistory.appendChild(msgEl);

    // Auto-scroll
    elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
  }

  async function handleSendQuestion() {
    const question = elements.chatInput.value.trim();
    if (!question || state.isAsking) return;

    elements.chatInput.value = "";
    elements.chatInput.style.height = "auto";
    appendMessage("user", question);

    // Thinking placeholder
    state.isAsking = true;
    elements.btnSend.disabled = true;

    const thinkingMsg = document.createElement("div");
    thinkingMsg.className = "message assistant";
    thinkingMsg.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-content">
        <div class="pulse-loader">
          <span></span><span></span><span></span>
        </div>
        <span style="margin-left: 8px; font-size: 0.85rem; color: var(--text-muted);">Analyzing repository evidence...</span>
      </div>
    `;
    elements.chatHistory.appendChild(thinkingMsg);
    elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;

    const topK = parseInt(elements.selectTopK.value, 10) || 4;

    try {
      const res = await apiPost("/api/ask", {
        question: question,
        repo: state.status.active_repo,
        top_k: topK,
        hybrid: state.hybrid,
      });

      thinkingMsg.remove();
      appendMessage("assistant", res.answer, res.evidence);
    } catch (err) {
      thinkingMsg.remove();
      appendMessage("assistant", `❌ **Error:** ${err.message}`);
    } finally {
      state.isAsking = false;
      elements.btnSend.disabled = false;
    }
  }

  // -------------------------------------------------------------------------
  // Search Logic
  // -------------------------------------------------------------------------

  async function handleSearch() {
    const query = elements.searchInput.value.trim();
    if (!query) return;

    elements.searchResults.innerHTML = `
      <p style="text-align: center; color: var(--text-muted); padding: 20px 0;">
        <span class="pulse-loader"><span></span><span></span><span></span></span>
        Searching passages...
      </p>
    `;

    try {
      const res = await apiPost("/api/search", {
        query: query,
        repo: state.status.active_repo,
        top_k: 10,
        hybrid: state.hybrid,
      });

      if (!res.results || res.results.length === 0) {
        elements.searchResults.innerHTML = `<p style="text-align: center; color: var(--text-muted); padding: 40px 0;">No matching code or documentation found.</p>`;
        return;
      }

      elements.searchResults.innerHTML = "";
      res.results.forEach((item) => {
        const card = document.createElement("div");
        card.className = "result-card";

        const locHtml = item.link
          ? `<a class="result-location" href="${escapeHtml(item.link)}" target="_blank" rel="noopener">${escapeHtml(item.location)}</a>`
          : `<span class="result-location">${escapeHtml(item.location)}</span>`;

        card.innerHTML = `
          <div class="result-top">
            <span class="result-symbol">${escapeHtml(item.name)}</span>
            <span class="result-kind">${escapeHtml(item.kind)} (Score: ${item.score})</span>
          </div>
          <div>${locHtml}</div>
          ${item.docstring ? `<div class="result-docstring">${escapeHtml(item.docstring)}</div>` : ""}
          <pre class="result-code">${escapeHtml(item.code || "")}</pre>
        `;
        elements.searchResults.appendChild(card);
      });
    } catch (err) {
      elements.searchResults.innerHTML = `<p style="color: var(--accent-red); padding: 20px 0;">Error: ${escapeHtml(err.message)}</p>`;
    }
  }

  // -------------------------------------------------------------------------
  // Repositories Management
  // -------------------------------------------------------------------------

  async function loadRepos() {
    elements.reposList.innerHTML = `<p style="color: var(--text-muted);">Loading repositories...</p>`;
    try {
      const data = await apiGet("/api/repos");
      state.repos = data.repos;

      if (!data.repos || data.repos.length === 0) {
        elements.reposList.innerHTML = `<p style="color: var(--text-muted); padding: 12px 0;">No repositories indexed yet. Clone or load one above!</p>`;
        return;
      }

      elements.reposList.innerHTML = "";
      data.repos.forEach((repo) => {
        const card = document.createElement("div");
        card.className = `repo-card ${repo.is_active ? "active-repo" : ""}`;

        const activeBadge = repo.is_active
          ? `<span style="color: var(--accent-green); font-size: 0.78rem; font-weight: 700;">✔ Active</span>`
          : "";

        card.innerHTML = `
          <div class="repo-card-top">
            <span class="repo-name">${escapeHtml(repo.name)}</span>
            ${activeBadge}
          </div>
          <div class="repo-meta">
            Passages: <strong>${repo.passages}</strong> | Files: ${repo.scanned_files} | Branch: ${escapeHtml(repo.branch)}
          </div>
          ${repo.web_url ? `<div style="font-size: 0.74rem; color: var(--accent-cyan);">${escapeHtml(repo.web_url)}</div>` : ""}
          <div class="repo-actions">
            ${!repo.is_active ? `<button class="btn-secondary btn-use-repo" data-id="${escapeHtml(repo.id)}">Set Active</button>` : ""}
            <button class="btn-danger btn-del-repo" data-id="${escapeHtml(repo.id)}">Delete</button>
          </div>
        `;
        elements.reposList.appendChild(card);
      });

      // Bind use buttons
      document.querySelectorAll(".btn-use-repo").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const id = btn.dataset.id;
          await apiPost("/api/repos/use", { repo: id });
          await refreshStatus();
          loadRepos();
        });
      });

      // Bind delete buttons
      document.querySelectorAll(".btn-del-repo").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const id = btn.dataset.id;
          if (confirm(`Are you sure you want to delete repository '${id}' and its indexes?`)) {
            await apiDelete(`/api/repos/${id}`);
            await refreshStatus();
            loadRepos();
          }
        });
      });
    } catch (err) {
      elements.reposList.innerHTML = `<p style="color: var(--accent-red);">Failed to load repositories: ${err.message}</p>`;
    }
  }

  async function handleIngest() {
    const source = elements.ingestSource.value.trim();
    if (!source) {
      alert("Please enter a GitHub repository URL or shorthand.");
      return;
    }

    elements.btnIngest.disabled = true;
    elements.btnIngestText.textContent = "Cloning & Indexing...";

    try {
      const res = await apiPost("/api/repos/ingest", {
        source: source,
        force: elements.ingestForce.checked,
        python_only: elements.ingestPythonOnly.checked,
      });

      alert(`Successfully ingested and indexed ${res.passages} passages for [${res.repo_id}]!`);
      elements.ingestSource.value = "";
      await refreshStatus();
      loadRepos();
      switchTab("chat");
    } catch (err) {
      alert(`Ingestion failed: ${err.message}`);
    } finally {
      elements.btnIngest.disabled = false;
      elements.btnIngestText.textContent = "Clone & Index Repository";
    }
  }

  // -------------------------------------------------------------------------
  // Providers & Settings
  // -------------------------------------------------------------------------

  async function loadProviders() {
    try {
      const data = await apiGet("/api/providers");
      state.providers = data.providers;

      elements.providerGrid.innerHTML = "";
      data.providers.forEach((prov) => {
        const card = document.createElement("div");
        const isSelected = prov.name === state.selectedProvider;
        card.className = `provider-card ${isSelected ? "selected" : ""}`;
        card.dataset.name = prov.name;

        const typeBadge = prov.is_cloud ? "Cloud" : "Local";
        const activeMark = prov.is_active ? "✔ Active" : "";

        card.innerHTML = `
          <div class="provider-card-top">
            <span class="provider-title">${escapeHtml(prov.name)}</span>
            <span style="font-size: 0.72rem; color: ${prov.is_active ? "var(--accent-green)" : "var(--text-muted)"}; font-weight: 700;">
              ${activeMark || typeBadge}
            </span>
          </div>
          <div class="provider-desc">${escapeHtml(prov.description)}</div>
          <div class="provider-status">
            Model: <strong style="color: #fff;">${escapeHtml(prov.model)}</strong>
          </div>
        `;

        card.addEventListener("click", () => {
          selectProvider(prov.name);
        });

        elements.providerGrid.appendChild(card);
      });

      selectProvider(state.selectedProvider || data.active);
    } catch (err) {
      console.error("Providers load error:", err);
    }
  }

  function selectProvider(name) {
    state.selectedProvider = name;
    document.querySelectorAll(".provider-card").forEach((c) => {
      c.classList.toggle("selected", c.dataset.name === name);
    });

    const prov = state.providers.find((p) => p.name === name) || {};
    elements.configProviderName.textContent = `Configure ${prov.display_name || name}`;
    elements.cfgModel.value = prov.model || "";
    elements.cfgModel.placeholder = `Default: ${prov.model || ""}`;

    // Show/hide API key or URL
    elements.cfgKeyGroup.style.display = prov.is_cloud ? "block" : "none";
    elements.cfgUrlGroup.style.display = name === "custom" ? "block" : "none";
    elements.cfgKey.value = "";
  }

  async function handleSaveProvider() {
    const provName = state.selectedProvider;
    const model = elements.cfgModel.value.trim();
    const apiKey = elements.cfgKey.value.trim();
    const url = elements.cfgUrl.value.trim();

    try {
      await apiPost("/api/providers/switch", {
        provider: provName,
        model: model,
        api_key: apiKey,
        url: url,
      });

      alert(`Switched active provider to '${provName}'!`);
      await refreshStatus();
      loadProviders();
    } catch (err) {
      alert(`Failed to activate provider: ${err.message}`);
    }
  }

  // -------------------------------------------------------------------------
  // Event Bindings
  // -------------------------------------------------------------------------

  function bindEvents() {
    // Navigation Tabs
    elements.desktopTabs.forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });
    elements.bottomTabs.forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    // Header Badges
    elements.badgeRepo.addEventListener("click", () => switchTab("repos"));
    elements.badgeProvider.addEventListener("click", () => switchTab("settings"));

    // Phone / LAN Modal
    elements.btnPhone.addEventListener("click", () => {
      elements.modalLan.classList.add("open");
    });
    elements.btnModalClose.addEventListener("click", () => {
      elements.modalLan.classList.remove("open");
    });
    elements.modalLan.addEventListener("click", (e) => {
      if (e.target === elements.modalLan) {
        elements.modalLan.classList.remove("open");
      }
    });

    function copyLanLink(btnEl) {
      const url = elements.modalLanUrl.textContent;
      navigator.clipboard.writeText(url).then(() => {
        const orig = btnEl.textContent;
        btnEl.textContent = "✔ Copied!";
        setTimeout(() => (btnEl.textContent = orig), 2000);
      });
    }

    elements.btnCopyUrl.addEventListener("click", () => copyLanLink(elements.btnCopyUrl));
    elements.btnModalCopy.addEventListener("click", () => copyLanLink(elements.btnModalCopy));

    // Hybrid Search Toggle
    elements.btnToggleHybrid.addEventListener("click", () => {
      state.hybrid = !state.hybrid;
      elements.btnToggleHybrid.classList.toggle("active", state.hybrid);
    });

    // Chat Input
    elements.btnSend.addEventListener("click", handleSendQuestion);
    elements.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendQuestion();
      }
    });

    // Starter Chips
    elements.starterChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        elements.chatInput.value = chip.dataset.prompt;
        handleSendQuestion();
      });
    });

    // Search
    elements.btnSearchExec.addEventListener("click", handleSearch);
    elements.searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleSearch();
      }
    });

    // Ingest
    elements.btnIngest.addEventListener("click", handleIngest);

    // Settings
    elements.btnSaveProvider.addEventListener("click", handleSaveProvider);
  }

  // Init
  window.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    refreshStatus();
  });
})();
