/**
 * Atomic Tasks: render a category-grouped table with Task / Description / Horizon / Video.
 *
 * Bundled data (loaded via Sphinx html_js_files):
 * - window.ROBOCASA_ATOMIC_TASK_INDEX
 * - window.ROBOCASA_ATOMIC_TASK_ATTRIBUTES
 * - window.ROBOCASA_ATOMIC_EPISODE_LENGTHS (61 atomic horizons)
 * - window.ROBOCASA_EPISODE_LENGTHS (fallback for the few multi-stage "atomic" outliers)
 */
(() => {
  function onReady(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  function isAtomicTasksPage() {
    const path = window.location.pathname.replace(/\\/g, "/");
    return path.endsWith("/tasks/atomic_tasks.html") || path.endsWith("/tasks/atomic_tasks/");
  }

  function ensureBackToTopButton() {
    if (!isAtomicTasksPage()) return;
    if (document.querySelector(".rc-back-to-top")) return;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "rc-back-to-top";
    btn.setAttribute("aria-label", "Back to top");
    btn.title = "Back to top";
    btn.textContent = "↑";

    btn.addEventListener("click", () => {
      try {
        window.scrollTo({ top: 0, behavior: "smooth" });
      } catch {
        window.scrollTo(0, 0);
      }
    });

    const update = () => {
      const y = window.scrollY || document.documentElement.scrollTop || 0;
      btn.classList.toggle("rc-visible", y > 900);
    };

    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();

    document.body.appendChild(btn);
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatDescription(desc) {
    // Render "{foo/bar}" as italicized variable text without braces.
    // Also render "[foo/bar]" as "[<em>foo/bar</em>]" (keeps brackets visible).
    const raw = String(desc || "");
    const esc = escapeHtml(raw);
    let out = esc.replace(/\{([^}]+)\}/g, (_m, inner) => `<span class="rc-atomic-var"><em>${inner}</em></span>`);
    out = out.replace(/\[([^\]]+)\]/g, (_m, inner) => `[<span class="rc-atomic-var"><em>${inner}</em></span>]`);
    return out;
  }

  function getEpisodeLengthMap() {
    // Prefer atomic-specific lengths; fall back to composite lengths for any missing entries.
    const map = new Map();
    const a = window.ROBOCASA_ATOMIC_EPISODE_LENGTHS;
    if (a && typeof a === "object" && a.tasks && typeof a.tasks === "object") {
      for (const [taskName, data] of Object.entries(a.tasks)) {
        if (data && typeof data.mean_seconds === "number") map.set(taskName, data.mean_seconds);
      }
    }
    const c = window.ROBOCASA_EPISODE_LENGTHS;
    if (c && typeof c === "object" && c.tasks && typeof c.tasks === "object") {
      for (const [taskName, data] of Object.entries(c.tasks)) {
        if (!map.has(taskName) && data && typeof data.mean_seconds === "number") map.set(taskName, data.mean_seconds);
      }
    }
    return map;
  }

  function buildHorizonIntervals() {
    // Requested buckets: 5-10, 10-15, ..., 35-40, 40-50.
    const intervals = [];
    for (let min = 5; min < 40; min += 5) {
      const max = min + 5;
      intervals.push({ key: `${min}-${max}`, label: `${min}-${max}s`, min, max, inclusiveMax: false });
    }
    intervals.push({ key: "40-50", label: "40-50s", min: 40, max: 50, inclusiveMax: true });
    return intervals;
  }

  function horizonMatchesInterval(lengthSeconds, interval) {
    if (!Number.isFinite(lengthSeconds)) return false;
    if (interval.inclusiveMax) return lengthSeconds >= interval.min && lengthSeconds <= interval.max;
    return lengthSeconds >= interval.min && lengthSeconds < interval.max;
  }

  // Atomic Seen target tasks (from robocasa/utils/dataset_registry.py)
  const ATOM_SEEN_TARGET_TASKS = new Set([
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
  ]);

  const SORT_MODES = Object.freeze({
    SKILL: "skill",
    FIXTURE: "fixture",
  });

  // Atomic skill buckets (alphabetical by label)
  const SKILL_GROUPS = Object.freeze([
    // "Paired" complementary skills placed adjacent, then alphabetical.
    { id: "closing_doors", label: "Close Door" },
    { id: "opening_doors", label: "Open Door" },
    // Lids are combined into a single bucket
    { id: "lids", label: "Close & Open Lid" },
    { id: "insertion", label: "Insertion" },
    { id: "navigation", label: "Navigation" },
    { id: "pick_and_place", label: "Pick & Place" },
    { id: "pressing_buttons", label: "Press Button" },
    { id: "sliding_racks", label: "Slide Rack" },
    { id: "turning_levers", label: "Turn Lever" },
    { id: "twisting_knobs", label: "Twist Knob" },
  ]);

  const SKILL_LABEL_BY_ID = new Map(SKILL_GROUPS.map((s) => [s.id, s.label]));

  function getSkillIdForTaskName(taskName) {
    const name = String(taskName || "").trim();
    if (!name) return "pick_and_place";

    // Coffee tasks are "Insertion"
    if (name === "CoffeeSetupMug" || name === "CoffeeServeMug") return "insertion";

    // Electric kettle lid opens via a button
    if (name === "OpenElectricKettleLid") return "pressing_buttons";

    // Sink controls are "Turning Levers"
    if (
      name === "AdjustWaterTemperature" ||
      name === "TurnOnSinkFaucet" ||
      name === "TurnOffSinkFaucet" ||
      name === "TurnSinkSpout"
    ) {
      return "turning_levers";
    }

    // Explicit requests:
    // - OpenStandMixerHead => Opening Lids
    // - CloseStandMixerHead => Closing Lids
    if (name === "OpenStandMixerHead") return "lids";
    if (name === "CloseStandMixerHead") return "lids";

    // Lids (includes blender + electric kettle lids)
    if (/^(Open|Close).*(Lid|Head)$/.test(name)) return "lids";

    // Doors (fallback for drawers etc. until we add finer-grained skills)
    if (/^Open[A-Z]/.test(name)) return "opening_doors";
    if (/^Close[A-Z]/.test(name)) return "closing_doors";

    // Navigation
    if (name === "NavigateKitchen") return "navigation";

    // Sliding racks
    if (/^Slide/.test(name)) return "sliding_racks";

    // Buttons
    if (name === "StartCoffeeMachine") return "pressing_buttons";
    if (name === "TurnOnMicrowave" || name === "TurnOffMicrowave") return "pressing_buttons";
    if (name === "TurnOnBlender") return "pressing_buttons";
    if (name === "TurnOnElectricKettle") return "pressing_buttons";

    // Levers
    if (name === "TurnOnToaster") return "turning_levers";

    // Knobs / twisting-style controls
    if (name === "PreheatOven") return "twisting_knobs";
    if (name === "TurnOnStove" || name === "TurnOffStove") return "twisting_knobs";
    if (/^Adjust/.test(name)) return "twisting_knobs";
    if (name === "TurnSinkSpout" || name === "LowerHeat" || name === "TurnOnToasterOven") return "twisting_knobs";

    // Pick & place (default)
    return "pick_and_place";
  }

  // Fixture-view overrides:
  // - Remove fixture sections: Doors, Pick & Place, Navigation
  // - Create: "Fridge", "Cabinet", "Dishwasher" (virtual), and "Misc"
  // - Re-route former door + PickPlace + navigation tasks into their represented fixture buckets.
  const FIXTURE_VIEW_REMOVED_LABELS = new Set(["Doors", "Pick & Place", "Navigation"]);
  const FIXTURE_VIEW_MISC_TASKS = new Set([
    "CheesyBread",
    "MakeIcedCoffee",
    "PackDessert",
    "NavigateKitchen",
  ]);

  function spacedLowerFromIdentifier(name) {
    return (name || "")
      .replace(/_/g, " ")
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .toLowerCase()
      .trim();
  }

  function renderAtomicTargetTagIntoTaskCell(taskName, taskTd) {
    if (!taskName || !taskTd) return;

    // Idempotent: remove any existing indicator first (legacy stars + current tags)
    for (const old of Array.from(taskTd.querySelectorAll(".rc-atomic-star"))) old.remove();
    taskTd.querySelector(".rc-atomic-target-tagline")?.remove();

    if (!ATOM_SEEN_TARGET_TASKS.has(taskName)) return;

    const line = document.createElement("div");
    line.className = "rc-atomic-target-tagline";
    const pill = document.createElement("span");
    pill.className = "rc-task-tag rc-task-tag-atomic-seen";
    pill.textContent = "Atomic-Seen";
    pill.setAttribute("aria-label", "Atomic-Seen");
    line.appendChild(pill);

    // Place the tag under the task name (below the link / code)
    taskTd.appendChild(line);
  }

  function highlightRow(rowEl) {
    if (!rowEl) return;
    rowEl.classList.remove("rc-task-highlight");
    // force reflow so re-adding retriggers animation
    void rowEl.offsetWidth; // eslint-disable-line no-unused-expressions
    rowEl.classList.add("rc-task-highlight");
    window.setTimeout(() => rowEl.classList.remove("rc-task-highlight"), 1800);
  }

  function scrollToTaskRow(rowEl) {
    if (!rowEl) return;
    rowEl.scrollIntoView({ behavior: "smooth", block: "center" });

    // Match composite behavior: highlight when actually visible
    let highlighted = false;
    const doHighlight = () => {
      if (highlighted) return;
      highlighted = true;
      highlightRow(rowEl);
    };

    if (typeof IntersectionObserver !== "undefined") {
      const observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting && entry.intersectionRatio > 0.5) {
              observer.disconnect();
              window.setTimeout(doHighlight, 100);
              return;
            }
          }
        },
        { threshold: 0.5 }
      );
      observer.observe(rowEl);
      window.setTimeout(() => {
        observer.disconnect();
        doHighlight();
      }, 2000);
    } else {
      window.setTimeout(doHighlight, 350);
    }
  }

  function cssEscape(value) {
    // Use native CSS.escape when available; otherwise, fall back to a minimal escape.
    // https://developer.mozilla.org/en-US/docs/Web/API/CSS/escape
    if (typeof CSS !== "undefined" && typeof CSS.escape === "function") return CSS.escape(String(value));
    return String(value).replace(/["\\]/g, "\\$&");
  }

  function uniqueTokens(query) {
    const raw = (query || "")
      .toLowerCase()
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    return Array.from(new Set(raw));
  }

  function getVideoForTask(taskName) {
    if (!taskName) return null;
    const normalizedTaskName = String(taskName).replace(/\*/g, "").trim();
    // Public demo videos for atomic PickPlace tasks are stored with a "PnP" prefix on R2.
    // Example: PickPlaceCounterToDrawer -> PnPCounterToDrawer.mp4 ("PickPlace" is 9 chars)
    let videoTaskName = normalizedTaskName;
    if (videoTaskName.startsWith("PickPlace")) {
      videoTaskName = "PnP" + videoTaskName.slice(9);
    }
    const safe = encodeURIComponent(videoTaskName);
    const sources = [];

    const DEFAULT_PUBLIC_VIDEO_BASE_URL = "https://pub-4433dcd10060475196ea5832312785f9.r2.dev";
    const DEFAULT_PUBLIC_VIDEO_PREFIX = "robocasa365-videos";
    const base = window.ROBOCASA_VIDEO_BASE_URL;
    if (typeof base === "string" && base.trim()) {
      const b = base.replace(/\/+$/, "");
      sources.push(`${b}/${safe}.mp4`);
      if (!b.endsWith(`/${DEFAULT_PUBLIC_VIDEO_PREFIX}`)) {
        sources.push(`${b}/${DEFAULT_PUBLIC_VIDEO_PREFIX}/${safe}.mp4`);
      }
    } else {
      sources.push(`${DEFAULT_PUBLIC_VIDEO_BASE_URL}/${safe}.mp4`);
      sources.push(`${DEFAULT_PUBLIC_VIDEO_BASE_URL}/${DEFAULT_PUBLIC_VIDEO_PREFIX}/${safe}.mp4`);
    }
    return { label: "Demo", sources };
  }

  function ensureVideoModal() {
    let overlay = document.getElementById("rc-video-modal-overlay");
    if (overlay) return overlay;

    overlay = document.createElement("div");
    overlay.id = "rc-video-modal-overlay";
    overlay.className = "rc-video-modal-overlay";
    overlay.hidden = true;

    const modal = document.createElement("div");
    modal.className = "rc-video-modal";

    const close = document.createElement("button");
    close.type = "button";
    close.className = "rc-video-modal-close";
    close.setAttribute("aria-label", "Close video");
    close.textContent = "×";

    const header = document.createElement("div");
    header.className = "rc-video-modal-header";

    const title = document.createElement("div");
    title.className = "rc-video-modal-title";
    const titleCode = document.createElement("code");
    titleCode.className = "rc-video-modal-title-code";
    titleCode.textContent = "";
    title.appendChild(titleCode);

    const video = document.createElement("video");
    video.className = "rc-video-modal-player";
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";

    const instruction = document.createElement("div");
    instruction.className = "rc-video-modal-instruction";
    instruction.hidden = true;
    const instructionLabel = document.createElement("strong");
    const instructionText = document.createElement("span");
    instructionText.className = "rc-video-modal-instruction-text";
    instruction.appendChild(instructionLabel);
    instruction.appendChild(instructionText);

    const error = document.createElement("div");
    error.className = "rc-video-modal-error";
    error.hidden = true;

    let loadTimeout = null;
    let hasLoadedMetadata = false;

    function doClose() {
      overlay.hidden = true;
      document.body.classList.remove("rc-modal-open");
      try {
        video.pause();
      } catch {}
      if (loadTimeout) {
        clearTimeout(loadTimeout);
        loadTimeout = null;
      }
      hasLoadedMetadata = false;
      titleCode.textContent = "";
      instruction.hidden = true;
      instructionLabel.textContent = "";
      instructionText.textContent = "";
      error.hidden = true;
      error.textContent = "";
      video.removeAttribute("src");
      video.load();
    }

    close.addEventListener("click", doClose);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) doClose();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !overlay.hidden) doClose();
    });

    header.appendChild(title);
    header.appendChild(close);
    modal.appendChild(header);
    modal.appendChild(video);
    modal.appendChild(instruction);
    modal.appendChild(error);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    overlay._rcOpen = (sources, taskName, taskDescription) => {
      overlay.hidden = false;
      document.body.classList.add("rc-modal-open");
      error.hidden = true;
      error.textContent = "";
      titleCode.textContent = taskName ? `${taskName}` : "";

      const tName = (taskName || "").trim();
      const tDesc = (taskDescription || "").trim();
      if (tDesc) {
        instruction.hidden = false;
        // Title already shows task name; bottom should only show the description.
        instructionLabel.textContent = "";
        // description may contain inline HTML (italics for variables); render as-is
        instructionText.innerHTML = tDesc;
      } else {
        instruction.hidden = true;
        instructionLabel.textContent = "";
        instructionText.textContent = "";
      }

      const srcsRaw = Array.isArray(sources) ? sources : [sources];
      const srcs = srcsRaw.map((s) => String(s || "").trim()).filter(Boolean);
      let attempt = 0;
      let loadTimeout = null;
      let hasLoadedMetadata = false;

      function showUnavailable() {
        video.removeAttribute("src");
        video.load();
        error.hidden = false;
        error.textContent = "Demo video unavailable.";
      }

      function tryNextSource() {
        if (attempt >= srcs.length) {
          showUnavailable();
          return;
        }
        const src = srcs[attempt++];
        hasLoadedMetadata = false;
        
        // Clear any existing timeout
        if (loadTimeout) {
          clearTimeout(loadTimeout);
          loadTimeout = null;
        }

        // Safari: use canplay or loadedmetadata to detect successful load
        const onCanPlay = () => {
          hasLoadedMetadata = true;
          if (loadTimeout) {
            clearTimeout(loadTimeout);
            loadTimeout = null;
          }
          video.removeEventListener("canplay", onCanPlay);
          video.removeEventListener("loadedmetadata", onCanPlay);
        };
        video.addEventListener("canplay", onCanPlay);
        video.addEventListener("loadedmetadata", onCanPlay);

        video.src = src;
        video.load();
        
        // Safari: give it more time before assuming error (especially for R2/CDN)
        // Increased timeout for Safari which can be slower with cross-origin video loading
        loadTimeout = window.setTimeout(() => {
          if (!hasLoadedMetadata) {
            // Safari may not fire error immediately; try next source
            video.removeEventListener("canplay", onCanPlay);
            video.removeEventListener("loadedmetadata", onCanPlay);
            tryNextSource();
          }
        }, 5000);

        const p = video.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      }

      // Safari: only treat as error if we haven't loaded metadata yet
      video.onerror = () => {
        if (loadTimeout) {
          clearTimeout(loadTimeout);
          loadTimeout = null;
        }
        if (!hasLoadedMetadata) {
          tryNextSource();
        }
      };
      tryNextSource();
    };

    return overlay;
  }

  onReady(() => {
    if (!isAtomicTasksPage()) return;
    document.body.classList.add("rc-atomic-tasks");
    ensureBackToTopButton();

    const root = document.getElementById("rc-atomic-tasks-root");
    if (!root) return;

    const idx = window.ROBOCASA_ATOMIC_TASK_INDEX;
    const attrs = window.ROBOCASA_ATOMIC_TASK_ATTRIBUTES;
    const fixtures = idx && Array.isArray(idx.fixtures) ? idx.fixtures : [];
    const episodeLengthMap = getEpisodeLengthMap();
    const overlay = ensureVideoModal();
    const horizonIntervals = buildHorizonIntervals();

    root.textContent = "";
    if (!fixtures.length) return;

    // Build a canonical per-task metadata map so we can re-group by skill or fixture.
    const taskMetaByName = new Map();
    for (const fx of fixtures) {
      const tasks = Array.isArray(fx.tasks) ? fx.tasks : [];
      for (const t of tasks) {
        const name = t?.name;
        if (!name || taskMetaByName.has(name)) continue;
        taskMetaByName.set(name, {
          name,
          lineno: t?.lineno || null,
          github: String(t?.github || fx?.github || ""),
          fixtureId: String(fx?.id || ""),
          fixtureLabel: String(fx?.label || fx?.id || ""),
        });
      }
    }

    // Controls row
    const controls = document.createElement("div");
    controls.className = "rc-atomic-controls";

    // Sort mode (Skill vs Fixture)
    let sortMode = SORT_MODES.SKILL; // default
    const sortWrap = document.createElement("div");
    sortWrap.className = "rc-atomic-control rc-atomic-sort";
    const sortLabel = document.createElement("label");
    sortLabel.textContent = "Sort:";
    sortLabel.setAttribute("for", "rc-atomic-sort-select");
    const sortSelect = document.createElement("select");
    sortSelect.id = "rc-atomic-sort-select";
    const sortOptSkill = document.createElement("option");
    sortOptSkill.value = SORT_MODES.SKILL;
    sortOptSkill.textContent = "By Skill";
    const sortOptFixture = document.createElement("option");
    sortOptFixture.value = SORT_MODES.FIXTURE;
    sortOptFixture.textContent = "By Fixture";
    sortSelect.appendChild(sortOptSkill);
    sortSelect.appendChild(sortOptFixture);
    sortWrap.appendChild(sortLabel);
    sortWrap.appendChild(sortSelect);

    const catWrap = document.createElement("div");
    catWrap.className = "rc-atomic-control rc-atomic-fixture-select";
    const catLabel = document.createElement("label");
    catLabel.textContent = "Skill:";
    catLabel.setAttribute("for", "rc-atomic-fixture-select");
    const catSelect = document.createElement("select");
    catSelect.id = "rc-atomic-fixture-select";
    // Enable :invalid styling for placeholder text
    catSelect.required = true;
    const optAll = document.createElement("option");
    optAll.value = "";
    optAll.disabled = true;
    optAll.selected = true;
    optAll.textContent = "Select a skill…";
    catSelect.appendChild(optAll);
    catWrap.appendChild(catLabel);
    catWrap.appendChild(catSelect);

    const searchWrap = document.createElement("div");
    searchWrap.className = "rc-atomic-control rc-atomic-search";
    const searchLabel = document.createElement("label");
    searchLabel.textContent = "Task:";
    searchLabel.setAttribute("for", "rc-atomic-task-input");
    const inputWrap = document.createElement("div");
    inputWrap.className = "rc-atomic-task-input-wrap";
    const input = document.createElement("input");
    input.id = "rc-atomic-task-input";
    input.type = "search";
    input.placeholder = "Search a task…";
    input.autocomplete = "off";
    input.spellcheck = false;
    const suggest = document.createElement("div");
    suggest.className = "rc-atomic-task-suggest";
    suggest.hidden = true;
    suggest.setAttribute("role", "listbox");
    inputWrap.appendChild(input);
    inputWrap.appendChild(suggest);
    searchWrap.appendChild(searchLabel);
    searchWrap.appendChild(inputWrap);

    // Target Tasks filter (All Tasks vs Atomic Seen)
    // Mirrors Composite page "Target Tasks" dropdown UI.
    const targetWrap = document.createElement("div");
    targetWrap.className = "rc-atomic-control rc-filter rc-filter-target-tasks";

    const targetBtn = document.createElement("button");
    targetBtn.type = "button";
    targetBtn.className = "rc-task-attr-dropdown-btn";
    targetBtn.setAttribute("aria-label", "Target Tasks");
    targetBtn.textContent = "Target Tasks";

    const targetMenu = document.createElement("div");
    targetMenu.className = "rc-task-attr-dropdown";
    targetMenu.hidden = true;

    // All Tasks option
    const allTasksRow = document.createElement("label");
    allTasksRow.className = "rc-task-attr-item rc-target-all-tasks-row";
    const allTasksCb = document.createElement("input");
    allTasksCb.type = "checkbox";
    allTasksCb.checked = true; // default
    allTasksCb.setAttribute("aria-label", "All tasks");
    const allTasksText = document.createElement("span");
    allTasksText.textContent = "All Tasks";
    const allTasksCount = document.createElement("span");
    allTasksCount.className = "rc-task-attr-count";
    allTasksCount.textContent = "";
    allTasksRow.appendChild(allTasksCb);
    allTasksRow.appendChild(allTasksText);
    allTasksRow.appendChild(allTasksCount);
    targetMenu.appendChild(allTasksRow);

    // Divider (matches composite dropdowns visually)
    const targetDivider = document.createElement("div");
    targetDivider.style.height = "1px";
    targetDivider.style.background = "var(--pst-color-border, #e0e0e0)";
    targetDivider.style.margin = "0.25rem 0.15rem 0.35rem 0";
    targetMenu.appendChild(targetDivider);

    // Atomic Seen option
    const atomicSeenRow = document.createElement("label");
    atomicSeenRow.className = "rc-task-attr-item";
    const atomicSeenCb = document.createElement("input");
    atomicSeenCb.type = "checkbox";
    atomicSeenCb.checked = true;
    atomicSeenCb.value = "atomic_seen";
    atomicSeenCb.setAttribute("aria-label", "Atomic-Seen");
    const atomicSeenPill = document.createElement("span");
    atomicSeenPill.className = "rc-task-tag rc-task-tag-atomic-seen";
    atomicSeenPill.textContent = "Atomic-Seen";
    const atomicSeenCount = document.createElement("span");
    atomicSeenCount.className = "rc-task-attr-count";
    atomicSeenCount.textContent = "";
    atomicSeenRow.appendChild(atomicSeenCb);
    const atomicSeenRight = document.createElement("span");
    atomicSeenRight.className = "rc-task-attr-item-right";
    atomicSeenRight.appendChild(atomicSeenPill);
    atomicSeenRow.appendChild(atomicSeenRight);
    atomicSeenRow.appendChild(atomicSeenCount);
    targetMenu.appendChild(atomicSeenRow);

    function syncFromAllTasks() {
      if (!allTasksCb.checked) return;
      atomicSeenCb.checked = true;
    }

    allTasksCb.addEventListener("change", () => {
      syncFromAllTasks();
      applyFilters();
    });
    atomicSeenCb.addEventListener("change", () => {
      // Custom selection means not "All Tasks" mode.
      allTasksCb.checked = false;
      applyFilters();
    });

    targetBtn.addEventListener("click", () => {
      targetMenu.hidden = !targetMenu.hidden;
    });
    document.addEventListener("click", (e) => {
      if (targetWrap.contains(e.target)) return;
      targetMenu.hidden = true;
    });

    targetWrap.appendChild(targetBtn);
    targetWrap.appendChild(targetMenu);

    // Horizon dropdown (right side)
    const horizonWrap = document.createElement("div");
    // Use Composite "Episode Length" dropdown classes for identical sizing/look
    horizonWrap.className = "rc-atomic-control rc-filter rc-filter-length rc-atomic-filter-horizon";
    const horizonBtn = document.createElement("button");
    horizonBtn.type = "button";
    horizonBtn.className = "rc-length-dropdown-btn";
    horizonBtn.setAttribute("aria-label", "Horizon");
    const horizonMenu = document.createElement("div");
    horizonMenu.className = "rc-length-dropdown";
    horizonMenu.hidden = true;

    const horizonChecks = new Map();
    const horizonMeta = new Map();

    function updateHorizonButton() {
      const total = horizonIntervals.length;
      let checked = 0;
      for (const cb of horizonChecks.values()) if (cb.checked) checked += 1;
      if (checked === total) horizonBtn.textContent = "Horizon (All)";
      else if (checked === 0) horizonBtn.textContent = "Horizon (None)";
      else horizonBtn.textContent = `Horizon (${checked}/${total})`;
    }

    const hdr = document.createElement("div");
    hdr.className = "rc-length-header";
    const allRow = document.createElement("label");
    allRow.className = "rc-length-all";
    const allCb = document.createElement("input");
    allCb.type = "checkbox";
    allCb.checked = true;
    const allText = document.createElement("span");
    allText.textContent = "All";
    allRow.appendChild(allCb);
    allRow.appendChild(allText);
    hdr.appendChild(allRow);
    horizonMenu.appendChild(hdr);

    function syncAllCheckbox() {
      const total = horizonIntervals.length;
      let checked = 0;
      for (const cb of horizonChecks.values()) if (cb.checked) checked += 1;
      if (checked === 0) {
        allCb.checked = false;
        allCb.indeterminate = false;
      } else if (checked === total) {
        allCb.checked = true;
        allCb.indeterminate = false;
      } else {
        allCb.checked = false;
        allCb.indeterminate = true;
      }
    }

    allCb.addEventListener("change", () => {
      const total = horizonIntervals.length;
      let checked = 0;
      for (const cb of horizonChecks.values()) if (cb.checked) checked += 1;
      const shouldSelectAll = checked !== total;
      for (const cb of horizonChecks.values()) cb.checked = shouldSelectAll;
      syncAllCheckbox();
      updateHorizonButton();
      applyFilters();
    });

    for (const it of horizonIntervals) {
      const row = document.createElement("label");
      row.className = "rc-length-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = true;
      cb.value = it.key;
      const labelSpan = document.createElement("span");
      labelSpan.textContent = it.label;
      const countSpan = document.createElement("span");
      countSpan.className = "rc-length-item-count";
      countSpan.textContent = "(0)";
      row.appendChild(cb);
      row.appendChild(labelSpan);
      row.appendChild(countSpan);
      horizonMenu.appendChild(row);
      horizonChecks.set(it.key, cb);
      horizonMeta.set(it.key, { ...it, countEl: countSpan });
      cb.addEventListener("change", () => {
        syncAllCheckbox();
        updateHorizonButton();
        applyFilters();
      });
    }
    syncAllCheckbox();
    updateHorizonButton();

    horizonBtn.addEventListener("click", () => (horizonMenu.hidden = !horizonMenu.hidden));
    document.addEventListener("click", (e) => {
      if (horizonWrap.contains(e.target)) return;
      horizonMenu.hidden = true;
    });

    horizonWrap.appendChild(horizonBtn);
    horizonWrap.appendChild(horizonMenu);

    const countEl = document.createElement("div");
    countEl.className = "rc-atomic-count";
    countEl.textContent = "Showing 0 tasks";

    // Footer row: count (left) + Reset (right), matching Composite
    const footerRow = document.createElement("div");
    footerRow.className = "rc-atomic-controls-footer";

    // Match Composite: "Sort" immediately to the left of "Showing X tasks"
    sortWrap.appendChild(countEl);

    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "rc-reset-filters-btn";
    resetBtn.textContent = "Reset";
    resetBtn.setAttribute("aria-label", "Reset filters");
    resetBtn.addEventListener("click", () => {
      // Reset filter inputs to defaults
      input.value = "";
      suggest.hidden = true;
      // Horizon: select all
      for (const cb of horizonChecks.values()) cb.checked = true;
      syncAllCheckbox();
      updateHorizonButton();
      horizonMenu.hidden = true;
      // Target Tasks: back to "All Tasks" mode
      allTasksCb.checked = true;
      atomicSeenCb.checked = true;
      targetMenu.hidden = true;
      applyFilters();
    });

    footerRow.appendChild(sortWrap);
    footerRow.appendChild(resetBtn);

    controls.appendChild(catWrap);
    controls.appendChild(searchWrap);
    controls.appendChild(horizonWrap);
    controls.appendChild(targetWrap);
    controls.appendChild(footerRow);
    root.appendChild(controls);

    // Sections
    const sectionsWrap = document.createElement("div");
    sectionsWrap.className = "rc-atomic-sections";
    root.appendChild(sectionsWrap);

    const groupEls = new Map(); // groupId -> { sectionEl, badgeEl, titleEl }
    const groupTotals = new Map(); // groupId -> total tasks in group (before filtering)
    const taskIndex = [];

    function buildGroups(mode) {
      if (mode === SORT_MODES.FIXTURE) {
        // Build label -> fixture id for all "real" fixtures (excluding removed ones)
        const fixtureIdByLabel = new Map();
        const fixtureOrder = [];
        for (const fx of fixtures) {
          const label = String(fx?.label || fx?.id || "");
          const id = String(fx?.id || "");
          if (!label || !id) continue;
          if (FIXTURE_VIEW_REMOVED_LABELS.has(label)) continue;
          if (!fixtureIdByLabel.has(label)) fixtureIdByLabel.set(label, id);
          fixtureOrder.push(label);
        }

        const GROUP_MISC_ID = "rc_fixture_misc";
        const GROUP_FRIDGE_ID = "rc_fixture_fridge";
        const GROUP_CABINET_ID = "rc_fixture_cabinet";
        const GROUP_DISHWASHER_ID = "rc_fixture_dishwasher";

        function ensureGroup(groupsById, id, label) {
          if (groupsById.has(id)) return groupsById.get(id);
          const g = { id, label, tasks: [] };
          groupsById.set(id, g);
          return g;
        }

        function groupIdForDoorTask(name) {
          if (name.includes("Fridge")) return GROUP_FRIDGE_ID;
          if (name.includes("Cabinet")) return GROUP_CABINET_ID;
          if (name.includes("Dishwasher")) return GROUP_DISHWASHER_ID;
          if (name.includes("Microwave")) return fixtureIdByLabel.get("Microwave") || GROUP_MISC_ID;
          if (name.includes("Oven")) {
            // Prefer Toaster Oven before Oven
            if (name.includes("ToasterOven")) return fixtureIdByLabel.get("Toaster Oven") || GROUP_MISC_ID;
            return fixtureIdByLabel.get("Oven") || GROUP_MISC_ID;
          }
          if (name.includes("ToasterOven")) return fixtureIdByLabel.get("Toaster Oven") || GROUP_MISC_ID;
          return GROUP_MISC_ID;
        }

        function groupIdForPickPlaceTask(name) {
          // Map PickPlace tasks to the fixture they represent
          if (name.includes("Fridge")) return GROUP_FRIDGE_ID;
          if (name.includes("Cabinet")) return GROUP_CABINET_ID;
          if (name.includes("Blender")) return fixtureIdByLabel.get("Blender") || GROUP_MISC_ID;
          if (name.includes("Microwave")) return fixtureIdByLabel.get("Microwave") || GROUP_MISC_ID;
          if (name.includes("StandMixer")) return fixtureIdByLabel.get("Stand Mixer") || GROUP_MISC_ID;
          if (name.includes("ToasterOven")) return fixtureIdByLabel.get("Toaster Oven") || GROUP_MISC_ID;
          if (name.includes("Toaster")) return fixtureIdByLabel.get("Toaster") || GROUP_MISC_ID;
          if (name.includes("Oven")) return fixtureIdByLabel.get("Oven") || GROUP_MISC_ID;
          if (name.includes("Sink")) return fixtureIdByLabel.get("Sink") || GROUP_MISC_ID;
          if (name.includes("Stove")) return fixtureIdByLabel.get("Stove") || GROUP_MISC_ID;
          if (name.includes("Drawer")) return fixtureIdByLabel.get("Drawer") || GROUP_MISC_ID;
          return GROUP_MISC_ID;
        }

        // Assemble groups in a stable order:
        // - Existing fixtures (excluding removed ones)
        // - Fridge (new)
        // - Cabinet (new)
        // - Dishwasher (only if needed)
        // - Misc (new; last)
        const groupsById = new Map();
        for (const label of fixtureOrder) {
          const id = fixtureIdByLabel.get(label);
          ensureGroup(groupsById, id, label);
        }
        ensureGroup(groupsById, GROUP_FRIDGE_ID, "Fridge");
        ensureGroup(groupsById, GROUP_CABINET_ID, "Cabinet");
        // Dishwasher group is created lazily only if we route tasks to it
        ensureGroup(groupsById, GROUP_MISC_ID, "Miscellaneous");

        // Route every task into its fixture-view group
        for (const meta of taskMetaByName.values()) {
          const name = String(meta?.name || "");
          if (!name) continue;

          let groupId = null;
          let groupLabel = null;

          // Forced misc tasks
          if (FIXTURE_VIEW_MISC_TASKS.has(name)) {
            groupId = GROUP_MISC_ID;
            groupLabel = "Miscellaneous";
          }
          // Dishwasher rack slide belongs in Dishwasher (even though it's defined in Drawer env)
          else if (name === "SlideDishwasherRack") {
            groupId = GROUP_DISHWASHER_ID;
            groupLabel = "Dishwasher";
          }
          // Door tasks formerly under "Doors"
          else if (/^(Open|Close)(Cabinet|Dishwasher|Fridge|Microwave|Oven|ToasterOvenDoor|ToasterOven)/.test(name)) {
            groupId = groupIdForDoorTask(name);
          }
          // Any other Open/Close that were historically in Doors (keep behavior flexible)
          else if (/^(Open|Close)[A-Z]/.test(name) && (meta.fixtureLabel === "Doors" || meta.fixtureId.includes("kitchen_doors"))) {
            groupId = groupIdForDoorTask(name);
          }
          // Navigation task formerly under "Navigation"
          else if (name === "NavigateKitchen" || meta.fixtureLabel === "Navigation") {
            groupId = GROUP_MISC_ID;
            groupLabel = "Miscellaneous";
          }
          // PickPlace tasks formerly under "Pick & Place"
          else if (name.startsWith("PickPlace") || meta.fixtureLabel === "Pick & Place") {
            groupId = groupIdForPickPlaceTask(name);
          }

          // Fall back to original fixture bucket when not overridden
          if (!groupId) {
            // Keep original fixture unless it was removed; otherwise misc
            if (FIXTURE_VIEW_REMOVED_LABELS.has(String(meta.fixtureLabel || ""))) {
              groupId = GROUP_MISC_ID;
              groupLabel = "Miscellaneous";
            } else {
              groupId = String(meta.fixtureId || "") || GROUP_MISC_ID;
            }
          }

          // Resolve label for the chosen group id
          if (!groupLabel) {
            if (groupId === GROUP_FRIDGE_ID) groupLabel = "Fridge";
            else if (groupId === GROUP_CABINET_ID) groupLabel = "Cabinet";
            else if (groupId === GROUP_MISC_ID) groupLabel = "Miscellaneous";
            else if (groupId === GROUP_DISHWASHER_ID) groupLabel = "Dishwasher";
            else {
              // find label by id (reverse map), or keep original
              groupLabel =
                Array.from(fixtureIdByLabel.entries()).find(([, id]) => id === groupId)?.[0] ||
                String(meta.fixtureLabel || groupId);
            }
          }

          // Ensure group exists (dishwasher may be needed)
          const g = ensureGroup(groupsById, groupId, groupLabel);

          // Push a per-view copy with updated fixtureId/fixtureLabel so UI metadata matches.
          g.tasks.push({
            ...meta,
            fixtureId: groupId,
            fixtureLabel: groupLabel,
          });
        }

        // Materialize groups list:
        // - Alphabetical by label (Cabinet after Blender; Fridge between Electric Kettle and Microwave; etc.)
        // - Miscellaneous is always pinned to the bottom
        const miscGroup = groupsById.get(GROUP_MISC_ID);
        const mainGroups = Array.from(groupsById.values())
          .filter(Boolean)
          .filter((g) => g.id !== GROUP_MISC_ID)
          // Hide empty real-fixture groups, but always keep Cabinet/Fridge visible (even if empty)
          .filter((g) => g.tasks.length > 0 || g.id === GROUP_CABINET_ID || g.id === GROUP_FRIDGE_ID)
          .sort((a, b) => String(a.label || "").localeCompare(String(b.label || ""), undefined, { sensitivity: "base" }));

        // Pin Miscellaneous last (even if empty)
        if (miscGroup) mainGroups.push(miscGroup);
        return mainGroups;
      }

      // Skill mode (default)
      const skillGroups = new Map(SKILL_GROUPS.map((s) => [s.id, { id: s.id, label: s.label, tasks: [] }]));
      for (const meta of taskMetaByName.values()) {
        const skillId = getSkillIdForTaskName(meta.name);
        const g = skillGroups.get(skillId) || skillGroups.get("pick_and_place");
        g.tasks.push(meta);
      }
      // Preserve alphabetical ordering (SKILL_GROUPS already sorted)
      return SKILL_GROUPS.map((s) => skillGroups.get(s.id));
    }

    function repopulateCategorySelect(groups) {
      // Preserve the placeholder as the first option.
      while (catSelect.options.length > 1) catSelect.remove(1);
      for (const g of groups) {
        if (!g) continue;
        const opt = document.createElement("option");
        opt.value = g.id || g.label || "";
        opt.textContent = g.label || g.id || "Category";
        catSelect.appendChild(opt);
      }
    }

    function renderGroups() {
      sectionsWrap.textContent = "";
      groupEls.clear();
      groupTotals.clear();
      taskIndex.length = 0;

      const groups = buildGroups(sortMode).filter(Boolean);
      repopulateCategorySelect(groups);

      // Update UI labels / placeholder to match mode
      if (sortMode === SORT_MODES.SKILL) {
        catLabel.textContent = "Skill:";
        optAll.textContent = "Select a skill…";
      } else {
        catLabel.textContent = "Fixture:";
        optAll.textContent = "Select a fixture…";
      }

      for (const g of groups) {
        const tasks = Array.isArray(g.tasks) ? g.tasks : [];
        groupTotals.set(g.id, tasks.length);

        const section = document.createElement("section");
        section.className = "rc-atomic-fixture";
        section.dataset.fixtureId = g.id || "";

        const header2 = document.createElement("div");
        header2.className = "rc-atomic-fixture-header";
        const title2 = document.createElement("div");
        title2.className = "rc-atomic-fixture-title";
        title2.textContent = g.label || g.id || "Category";
        const badge = document.createElement("div");
        badge.className = "rc-atomic-fixture-badge";
        badge.textContent = `${tasks.length} task${tasks.length === 1 ? "" : "s"}`;
        header2.appendChild(title2);
        header2.appendChild(badge);
        section.appendChild(header2);

        const body = document.createElement("div");
        body.className = "rc-atomic-fixture-body";

        if (!tasks.length) {
          const empty = document.createElement("div");
          empty.className = "rc-atomic-empty";
          empty.textContent = "No tasks yet.";
          body.appendChild(empty);
          section.appendChild(body);
          sectionsWrap.appendChild(section);
          groupEls.set(g.id || "", { sectionEl: section, badgeEl: badge, titleEl: title2 });
          continue;
        }

        // Stable sort tasks by name inside a group
        tasks.sort((a, b) => String(a?.name || "").localeCompare(String(b?.name || "")));

        const table = document.createElement("table");
        table.className = "rc-atomic-table";

        const colgroup = document.createElement("colgroup");
        for (const cls of ["rc-atomic-col-task", "rc-atomic-col-desc", "rc-atomic-col-horizon", "rc-atomic-col-video"]) {
          const col = document.createElement("col");
          col.className = cls;
          colgroup.appendChild(col);
        }
        table.appendChild(colgroup);

        const thead = document.createElement("thead");
        const trh = document.createElement("tr");
        for (const col of ["Task", "Description", "Horizon", "Video"]) {
          const th = document.createElement("th");
          th.textContent = col;
          trh.appendChild(th);
        }
        thead.appendChild(trh);
        table.appendChild(thead);

        const tbody = document.createElement("tbody");
        for (const t of tasks) {
          const name = t?.name;
          if (!name) continue;
          const meta = attrs && typeof attrs === "object" ? attrs[name] : null;
          const desc = meta && typeof meta === "object" ? meta.description : "";
          const horizon = episodeLengthMap.get(name);

          const tr = document.createElement("tr");

          const tdTask = document.createElement("td");
          tdTask.className = "rc-atomic-task";
          const a = document.createElement("a");
          a.href = `${String(t?.github || "")}${t?.lineno ? `#L${t.lineno}` : ""}`;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          const code = document.createElement("code");
          code.textContent = name;
          a.appendChild(code);
          tdTask.appendChild(a);
          renderAtomicTargetTagIntoTaskCell(name, tdTask);
          tr.appendChild(tdTask);

          const tdDesc = document.createElement("td");
          tdDesc.className = "rc-atomic-desc";
          tdDesc.innerHTML = formatDescription(desc || "");
          tr.appendChild(tdDesc);

          const tdH = document.createElement("td");
          tdH.className = "rc-atomic-horizon";
          tdH.textContent = Number.isFinite(horizon) ? `${Math.round(horizon)}s` : "—";
          tr.appendChild(tdH);

          const tdV = document.createElement("td");
          tdV.className = "rc-atomic-video";
          const v = getVideoForTask(name);
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "rc-atomic-video-btn";
          btn.textContent = "Demo";
          btn.addEventListener("click", () => {
            const descHtml = (tdDesc.innerHTML || tdDesc.textContent || "").trim();
            overlay._rcOpen(v.sources, name, descHtml);
          });
          tdV.appendChild(btn);
          tr.appendChild(tdV);

          tbody.appendChild(tr);

          const groupLabel = String(g.label || g.id || "");
          const fixtureLabel = String(t?.fixtureLabel || "");
          const searchText = `${String(name).toLowerCase()} ${spacedLowerFromIdentifier(name)} ${String(desc || "")
            .toLowerCase()
            .replace(/\s+/g, " ")} ${groupLabel.toLowerCase()} ${fixtureLabel.toLowerCase()}`;

          taskIndex.push({
            task: name,
            taskLower: String(name).toLowerCase(),
            taskSpacedLower: spacedLowerFromIdentifier(name),
            groupId: String(g.id || ""),
            groupLabel,
            fixtureId: String(t?.fixtureId || ""),
            fixtureLabel,
            horizon: Number.isFinite(horizon) ? Math.round(horizon) : null,
            rowEl: tr,
            sectionEl: section,
            searchText,
          });
        }
        table.appendChild(tbody);
        body.appendChild(table);
        section.appendChild(body);
        sectionsWrap.appendChild(section);

        groupEls.set(g.id || "", { sectionEl: section, badgeEl: badge, titleEl: title2 });
      }

      // Populate target counts (static counts, like Composite page)
      allTasksCount.textContent = `(${taskIndex.length})`;
      atomicSeenCount.textContent = `(${taskIndex.filter((it) => ATOM_SEEN_TARGET_TASKS.has(it.task)).length})`;
    }

    sortSelect.value = sortMode;
    sortSelect.addEventListener("change", () => {
      sortMode = sortSelect.value === SORT_MODES.FIXTURE ? SORT_MODES.FIXTURE : SORT_MODES.SKILL;
      catSelect.value = "";
      renderGroups();
      applyFilters();
    });

    renderGroups();

    function getActiveHorizonIntervals() {
      const total = horizonIntervals.length;
      let checked = 0;
      const active = [];
      for (const it of horizonIntervals) {
        const cb = horizonChecks.get(it.key);
        if (cb && cb.checked) {
          checked += 1;
          active.push(it);
        }
      }
      if (checked === total) return null;
      if (checked === 0) return [];
      return active;
    }

    function getActiveTargetFilter() {
      if (allTasksCb.checked) return null; // no target filtering
      if (atomicSeenCb.checked) return "atomic_seen";
      return []; // none selected => show nothing
    }

    function passesFilters(it, tokens, activeIntervals, targetFilter) {
      if (tokens.length && !tokens.every((t) => it.searchText.includes(t))) return false;
      if (Array.isArray(activeIntervals)) {
        if (activeIntervals.length === 0) return false;
        if (it.horizon == null) return false;
        let ok = false;
        for (const inter of activeIntervals) {
          if (horizonMatchesInterval(it.horizon, inter)) {
            ok = true;
            break;
          }
        }
        if (!ok) return false;
      }
      if (targetFilter === "atomic_seen") {
        if (!ATOM_SEEN_TARGET_TASKS.has(it.task)) return false;
      } else if (Array.isArray(targetFilter)) {
        if (targetFilter.length === 0) return false;
      }
      return true;
    }

    function updateHorizonDropdownCounts(tokens) {
      const activeIntervals = null; // counts ignore current horizon selection
      const targetFilter = getActiveTargetFilter();
      for (const inter of horizonIntervals) {
        let count = 0;
        for (const it of taskIndex) {
          if (!passesFilters(it, tokens, activeIntervals, targetFilter)) continue;
          if (it.horizon == null) continue;
          if (!horizonMatchesInterval(it.horizon, inter)) continue;
          count += 1;
        }
        const meta = horizonMeta.get(inter.key);
        if (meta?.countEl) meta.countEl.textContent = `(${count})`;
      }
    }

    function applyFilters() {
      const tokens = uniqueTokens(input.value);
      const activeIntervals = getActiveHorizonIntervals();
      const targetFilter = getActiveTargetFilter();
      let visibleTotal = 0;
      const visibleByGroup = new Map();

      for (const it of taskIndex) {
        const ok = passesFilters(it, tokens, activeIntervals, targetFilter);
        it.rowEl.style.display = ok ? "" : "none";
        if (ok) {
          visibleTotal += 1;
          visibleByGroup.set(it.groupId, (visibleByGroup.get(it.groupId) || 0) + 1);
        }
      }

      for (const [groupId, meta] of groupEls.entries()) {
        const vis = visibleByGroup.get(groupId) || 0;
        meta.badgeEl.textContent = `${vis} task${vis === 1 ? "" : "s"}`;
        const total = groupTotals.get(groupId) || 0;
        // Keep empty skill buckets visible (useful as placeholders for future mapping)
        if (sortMode === SORT_MODES.SKILL && total === 0) meta.sectionEl.style.display = "";
        else meta.sectionEl.style.display = vis > 0 ? "" : "none";
      }

      countEl.textContent = `Showing ${visibleTotal} task${visibleTotal === 1 ? "" : "s"}`;
      updateHorizonDropdownCounts(tokens);
    }

    // Typeahead suggestions (simple)
    let lastMatches = [];
    let activeIndex = -1;
    function setActiveIndex(nextIndex) {
      const btns = Array.from(suggest.querySelectorAll(".rc-atomic-task-suggest-item"));
      if (!btns.length) {
        activeIndex = -1;
        return;
      }
      if (nextIndex < 0) activeIndex = -1;
      else activeIndex = Math.max(0, Math.min(nextIndex, btns.length - 1));
      for (const [i, b] of btns.entries()) b.classList.toggle("rc-active", i === activeIndex);
    }

    function renderSuggestions(q) {
      const tokens = uniqueTokens(q);
      suggest.innerHTML = "";
      if (!tokens.length) {
        suggest.hidden = true;
        return [];
      }
      const matches = taskIndex
        .filter((it) => tokens.every((t) => it.searchText.includes(t)))
        .sort((a, b) => a.task.length - b.task.length)
        .slice(0, 12);
      for (const [idx, it] of matches.entries()) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "rc-atomic-task-suggest-item";
        const metaText = sortMode === SORT_MODES.SKILL ? it.groupLabel : it.fixtureLabel;
        btn.innerHTML = `<span class="rc-atomic-task-suggest-name">${it.task}</span><span class="rc-atomic-task-suggest-meta">${escapeHtml(metaText)}</span>`;
        btn.addEventListener("click", () => {
          input.value = it.task;
          suggest.hidden = true;
          setActiveIndex(-1);
          applyFilters();
          scrollToTaskRow(it.rowEl);
        });
        btn.addEventListener("mouseenter", () => setActiveIndex(idx));
        suggest.appendChild(btn);
      }
      suggest.hidden = matches.length === 0;
      setActiveIndex(-1);
      return matches;
    }

    input.addEventListener("input", () => {
      lastMatches = renderSuggestions(input.value);
      applyFilters();
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        if (suggest.hidden) lastMatches = renderSuggestions(input.value);
        if (lastMatches.length) {
          suggest.hidden = false;
          setActiveIndex(activeIndex < 0 ? 0 : activeIndex + 1);
        }
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        if (suggest.hidden) lastMatches = renderSuggestions(input.value);
        if (lastMatches.length) {
          suggest.hidden = false;
          setActiveIndex(activeIndex < 0 ? lastMatches.length - 1 : activeIndex - 1);
        }
      } else if (e.key === "Escape" || e.key === "Tab") {
        suggest.hidden = true;
        setActiveIndex(-1);
      } else if (e.key === "Enter") {
        suggest.hidden = true;
        const q = String(input.value || "").trim().toLowerCase();
        let chosen = null;
        if (q) {
          chosen = taskIndex.find((it) => it.taskLower === q) || null;
        }
        if (!chosen && activeIndex >= 0 && lastMatches[activeIndex]) chosen = lastMatches[activeIndex];
        setActiveIndex(-1);
        if (chosen) {
          applyFilters();
          scrollToTaskRow(chosen.rowEl);
        }
      }
    });
    document.addEventListener("click", (e) => {
      if (!searchWrap.contains(e.target)) {
        suggest.hidden = true;
        setActiveIndex(-1);
      }
    });

    // Match Composite Tasks Activity behavior: selecting a category scrolls to it (no filtering).
    catSelect.addEventListener("change", () => {
      const id = catSelect.value;
      if (!id) return;
      const meta = groupEls.get(id);
      const sectionEl =
        meta?.sectionEl ||
        document.querySelector(`section.rc-atomic-fixture[data-fixture-id="${cssEscape(id)}"]`);
      if (!sectionEl) return;
      sectionEl.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    applyFilters();
  });
})();

