/**
 * Composite Tasks: Activity dropdown + collapsible sections.
 *
 * This runs only on tasks/composite_tasks.html and transforms each H3 "Activity"
 * section into a <details> dropdown containing its table.
 */
(() => {
  function onReady(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  /** {x} and [*x*] -> [<strong><em>x</em></strong>]; *x* or **x** -> <strong><em>x</em></strong> (bold+italic). */
  function formatDescriptionToHtml(desc) {
    if (!desc) return "";
    const escaped = String(desc)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    return escaped
      .replace(/\{([^}]+)\}/g, "[<strong><em>$1</em></strong>]")
      .replace(/\[\*([^*]*?)\*\]/g, "[<strong><em>$1</em></strong>]")
      .replace(/\*\*([^*]+)\*\*/g, "<strong><em>$1</em></strong>")
      .replace(/\*([^*]+)\*/g, "<strong><em>$1</em></strong>");
  }

  function isCompositeTasksPage() {
    const path = window.location.pathname.replace(/\\/g, "/");
    return path.endsWith("/tasks/composite_tasks.html") || path.endsWith("/tasks/composite_tasks/");
  }

  function isDatasetsOverviewPage() {
    const path = window.location.pathname.replace(/\\/g, "/");
    return path.endsWith("/datasets/datasets_overview.html") || path.endsWith("/datasets/datasets_overview/");
  }

  function isObjectsPage() {
    const path = window.location.pathname.replace(/\\/g, "/");
    return path.endsWith("/assets/objects.html") || path.endsWith("/assets/objects/");
  }

  function isFoundationModelLearningPage() {
    const path = window.location.pathname.replace(/\\/g, "/");
    return (
      path.endsWith("/benchmarking/foundation_model_learning.html") ||
      path.endsWith("/benchmarking/foundation_model_learning/")
    );
  }

  function isLifelongLearningPage() {
    const path = window.location.pathname.replace(/\\/g, "/");
    return (
      path.endsWith("/benchmarking/lifelong_learning.html") ||
      path.endsWith("/benchmarking/lifelong_learning/")
    );
  }

  function initObjectsTableSorting() {
    const table = document.querySelector("table.rc-objects-table");
    if (!table) return;

    // Page-scoped styling hooks
    document.body.classList.add("rc-objects-page");

    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");
    if (!thead || !tbody) return;

    const ths = Array.from(thead.querySelectorAll("th"));
    if (ths.length === 0) return;

    function normalizeGroupsCell(td) {
      if (!td) return "";
      const tags = Array.from(td.querySelectorAll(".group-tag"));
      if (tags.length === 0) {
        const raw = (td.textContent || "").trim().replace(/\s+/g, " ");
        return raw;
      }
      const names = tags.map((el) => (el.textContent || "").trim()).filter(Boolean).sort((a, b) => a.localeCompare(b));
      // Reorder DOM to match alphabetical precedence
      if (tags.length > 1) {
        td.innerHTML = "";
        const wrap = document.createElement("span");
        wrap.className = "rc-groups-wrap";
        if (names.length === 2) wrap.classList.add("rc-groups-wrap-2");
        for (let i = 0; i < names.length; i++) {
          const span = document.createElement("span");
          span.className = "group-tag";
          span.textContent = names[i];
          wrap.appendChild(span);
        }
        td.appendChild(wrap);
      }
      return names.join(" ");
    }

    // Ensure multi-group rows display groups in alphabetical order
    for (const tr of Array.from(tbody.querySelectorAll("tr"))) {
      normalizeGroupsCell(tr.children?.[1]);
    }

    // 0: Category (string), 1: Groups (string), 2-4: numeric
    function cellKey(tr, idx) {
      const td = tr.children?.[idx];
      if (idx === 1) {
        return normalizeGroupsCell(td).toLowerCase();
      }
      const raw = (td?.textContent || "").trim().replace(/\\s+/g, " ");
      if (idx >= 2) {
        const n = Number.parseFloat(raw);
        return Number.isFinite(n) ? n : -Infinity;
      }
      return raw.toLowerCase();
    }

    let active = { idx: -1, dir: "asc" }; // dir: 'asc' | 'desc'

    const indicators = new Map(); // th -> span

    function applySort(idx, dir) {
      active = { idx, dir };

      const rows = Array.from(tbody.querySelectorAll("tr"));
      const keyed = rows.map((tr, origIdx) => ({ tr, origIdx, key: cellKey(tr, idx) }));
      keyed.sort((a, b) => {
        const ka = a.key;
        const kb = b.key;
        let cmp = 0;
        if (typeof ka === "number" && typeof kb === "number") cmp = ka - kb;
        else cmp = String(ka).localeCompare(String(kb));
        if (cmp === 0) cmp = a.origIdx - b.origIdx; // stable
        return dir === "asc" ? cmp : -cmp;
      });

      for (const { tr } of keyed) tbody.appendChild(tr);

      // Update indicators + aria-sort
      for (let j = 0; j < ths.length; j++) {
        const thj = ths[j];
        const indj = indicators.get(thj);
        if (!indj) continue;
        if (j === idx) {
          thj.setAttribute("aria-sort", dir === "asc" ? "ascending" : "descending");
          indj.textContent = dir === "asc" ? "▲" : "▼";
        } else {
          thj.setAttribute("aria-sort", "none");
          indj.textContent = "↕";
        }
      }
    }

    for (let i = 0; i < ths.length; i++) {
      const th = ths[i];
      th.setAttribute("aria-sort", "none");

      const label = document.createElement("span");
      label.className = "rc-sort-label";
      // Preserve any existing HTML in header cell
      while (th.firstChild) label.appendChild(th.firstChild);

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "rc-table-sort-btn";
      btn.setAttribute("aria-label", `Sort by ${(th.textContent || "").trim()}`);

      const ind = document.createElement("span");
      ind.className = "rc-table-sort-indicator";
      ind.textContent = "↕";
      btn.appendChild(ind);

      th.appendChild(label);
      th.appendChild(btn);
      indicators.set(th, ind);

      btn.addEventListener("click", () => {
        const idx = i;
        const dir = active.idx === idx ? (active.dir === "asc" ? "desc" : "asc") : "asc";
        applySort(idx, dir);
      });
    }

    // Default sort: Category A→Z
    applySort(0, "asc");
  }

  function getTaskAttributesMap() {
    // Prefer the pre-bundled map (works on file:// too)
    const obj = window.ROBOCASA_TASK_ATTRIBUTES;
    if (obj && typeof obj === "object") {
      return new Map(Object.entries(obj));
    }
    return null;
  }

  function getEpisodeLengthMap() {
    // Load episode length data from pre-bundled object or fetch from JSON
    const obj = window.ROBOCASA_EPISODE_LENGTHS;
    if (obj && typeof obj === "object" && obj.tasks) {
      const map = new Map();
      for (const [taskName, data] of Object.entries(obj.tasks)) {
        if (data && typeof data.mean_seconds === "number") {
          map.set(taskName, data.mean_seconds);
        }
      }
      return map;
    }
    return null;
  }

  function getAtomicEpisodeLengthMap() {
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

  let CACHED_TASK_ATTRIBUTES_JSON = null;
  async function loadTaskAttributesJson() {
    if (CACHED_TASK_ATTRIBUTES_JSON) return CACHED_TASK_ATTRIBUTES_JSON;
    // In this docs build, static assets are served from `_static/`.
    // (`html_static_path` merges several directories into `_static/` root.)
    // Append version to avoid stale cache when task list / activities change
    const url = `${getContentRoot()}_static/task_attributes.json?v=2`;
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) {
      throw new Error(`Failed to load ${url}: ${res.status} ${res.statusText}`);
    }
    const data = await res.json();
    CACHED_TASK_ATTRIBUTES_JSON = data;
    return data;
  }

  function getHeadingTitle(headingEl) {
    // Sphinx adds a permalink anchor like: <a class="headerlink"...>#</a>
    // We want the visible title without that "#".
    const clone = headingEl.cloneNode(true);
    const headerlink = clone.querySelector("a.headerlink");
    if (headerlink) headerlink.remove();
    return (clone.textContent || "").trim();
  }

  function anchorIdFromActivityTitle(title) {
    // Roughly matches Sphinx's default slugging for section IDs.
    return (title || "")
      .toLowerCase()
      .replace(/&/g, " and ")
      .replace(/[^a-z0-9\\s-]/g, "")
      .trim()
      .replace(/\\s+/g, "-")
      .replace(/-+/g, "-");
  }

  function normalizeActivityKey(activity) {
    return (activity || "").trim().replace(/\\s+/g, " ").toLowerCase();
  }

  function titleCaseActivity(activity) {
    const s = (activity || "").trim().replace(/\\s+/g, " ");
    if (!s) return s;
    const lowerWords = new Set(["a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with"]);
    return s
      .split(" ")
      .map((w, i) => {
        if (!w) return w;
        // Preserve all-caps acronyms
        if (w.length > 1 && w === w.toUpperCase()) return w;
        const lw = w.toLowerCase();
        if (i !== 0 && lowerWords.has(lw)) return lw;
        return lw.charAt(0).toUpperCase() + lw.slice(1);
      })
      .join(" ");
  }

  function folderTitleFromMultiStageFolder(folder) {
    // "adding_ice_to_beverages" -> "Adding Ice to Beverages"
    return titleCaseActivity((folder || "").replace(/_/g, " "));
  }

  function compactAlphaNumKey(s) {
    return (s || "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]/g, "");
  }

  // Source of truth: the actual directory names under robocasa/environments/kitchen/composite/
  // (Excluded: "__pycache__")
  const MULTI_STAGE_FOLDERS = [
    "adding_ice_to_beverages",
    "arranging_buffet",
    "arranging_cabinets",
    "arranging_condiments",
    "baking",
    "boiling",
    "brewing",
    "broiling_fish",
    "chopping_food",
    "chopping_vegetables",
    "cleaning_appliances",
    "cleaning_sink",
    "clearing_table",
    "defrosting_food",
    "filling_serving_dishes",
    "frying",
    "garnishing_dishes",
    "loading_dishwasher",
    "loading_fridge",
    "making_juice",
    "making_salads",
    "making_smoothies",
    "making_tea",
    "making_toast",
    "managing_freezer_space",
    "measuring_ingredients",
    "meat_preparation",
    "microwaving_food",
    "mixing_and_blending",
    "mixing_ingredients",
    "organizing_dishes_and_containers",
    "organizing_recycling",
    "organizing_utensils",
    "packing_lunches",
    "plating_food",
    "portioning_meals",
    "preparing_hot_chocolate",
    "preparing_marinade",
    "preparing_sandwiches",
    "reheating_food",
    "restocking_supplies",
    "sanitizing_surface",
    "sanitizing_cutting_board",
    "sauteing_vegetables",
    "seasoning_food",
    "serving_beverages",
    "serving_food",
    "setting_the_table",
    "simmering_sauces",
    "slicing_meat",
    "slow_cooking",
    "snack_preparation",
    "sorting_ingredients",
    "steaming_food",
    "steaming_vegetables",
    "storing_leftovers",
    "tidying_cabinets_and_drawers",
    "toasting_bread",
    "washing_dishes",
    "washing_fruits_and_vegetables",
  ];

  const MULTI_STAGE_FOLDER_SET = new Set(MULTI_STAGE_FOLDERS);

  const MULTI_STAGE_COMPACT_TO_FOLDER = (() => {
    const map = new Map();
    for (const folder of MULTI_STAGE_FOLDERS) {
      const title = folderTitleFromMultiStageFolder(folder);
      const candidates = new Set([compactAlphaNumKey(folder), compactAlphaNumKey(title)]);
      // Common broken format: words concatenated and sometimes singularized ("cabinets" -> "cabinet").
      for (const k of Array.from(candidates)) {
        if (k && k.endsWith("s")) candidates.add(k.slice(0, -1));
      }
      for (const k of candidates) {
        if (!k) continue;
        if (!map.has(k)) map.set(k, folder);
      }
    }
    return map;
  })();

  const ACTIVITY_TO_FOLDER_ALIASES = new Map([
    // Use the actual directory names under robocasa/environments/kitchen/composite/
    ["boiling water", "boiling"],
    ["brewing coffee", "brewing"],
    ["frying foods", "frying"],

    ["microwaving foods", "microwaving_food"],

    ["preparing marinades", "preparing_marinade"],
    ["preparing marinade", "preparing_marinade"],

    ["sanitizing surfaces", "sanitizing_surface"],
    ["Sanitizing Surface", "sanitizing_surface"],

    ["sanitizing cutting boards", "sanitizing_cutting_board"],
    ["sanitizing cutting board", "sanitizing_cutting_board"],

    // Typos seen in the wild
    ["serving begerages", "serving_beverages"],

    // More typos / legacy labels present in task_attributes.json
    ["boiilng water", "boiling"],
    ["cleaning appliacnes", "cleaning_appliances"],
    ["loading dishwasher", "loading_dishwasher"],
    ["washing produce", "washing_fruits_and_vegetables"],

    ["organizing dishes and containers", "organizing_dishes_and_containers"],
  ]);

  function folderForActivityTitle(activityTitle) {
    const key = normalizeActivityKey(activityTitle);
    const aliased = ACTIVITY_TO_FOLDER_ALIASES.get(key);
    if (aliased) return aliased;

    // If the incoming activity is already the (possibly broken) title, try to map it
    // to a known composite folder by compact matching.
    const compactFromTitle = compactAlphaNumKey(key);
    const fromCompact = MULTI_STAGE_COMPACT_TO_FOLDER.get(compactFromTitle);
    if (fromCompact) return fromCompact;

    // Default: "Adding ice to beverages" -> "adding_ice_to_beverages"
    const folder = (key || "")
      .replace(/&/g, " and ")
      .replace(/[^a-z0-9\\s-]/g, "")
      .trim()
      .replace(/[\\s-]+/g, "_")
      .replace(/_+/g, "_");

    // If the default slug doesn't match an actual folder, try compact matching before giving up.
    if (MULTI_STAGE_FOLDER_SET.has(folder)) return folder;
    const fromFolderCompact = MULTI_STAGE_COMPACT_TO_FOLDER.get(compactAlphaNumKey(folder));
    return fromFolderCompact || folder;
  }

  function groupTasksByActivity(tasks) {
    const map = new Map();
    for (const t of tasks || []) {
      const activityRaw = (t && t.activity) || "";
      const name = (t && t.name) || "";
      const description = (t && t.description) || "";
      if (!activityRaw || !name) continue;
      if (activityRaw === "Atomic") continue;

      const folder = folderForActivityTitle(activityRaw);
      if (!folder) continue;

      const prettyTitle = folderTitleFromMultiStageFolder(folder);
      if (!map.has(folder)) map.set(folder, { folder, title: prettyTitle, tasks: [] });

      const entry = map.get(folder);
      entry.title = prettyTitle;
      entry.tasks.push({ name, description });
    }
    // Sort tasks alphabetically by name within each activity
    for (const entry of map.values()) {
      if (Array.isArray(entry.tasks)) {
        entry.tasks.sort((a, b) => (a.name || "").localeCompare(b.name || ""));
      }
    }
    return map;
  }

  function buildTaskToSourceMapFromLegacy(rootSection) {
    const map = new Map();
    if (!rootSection) return map;
    // Legacy Sphinx content structure: <section id="..."><table>...</table></section>
    const rows = Array.from(rootSection.querySelectorAll(":scope > section[id] table tbody tr"));
    for (const tr of rows) {
      const tds = Array.from(tr.querySelectorAll("td"));
      if (tds.length < 3) continue;
      const taskName = (tds[0].querySelector("code")?.textContent || tds[0].textContent || "").trim();
      const href = tds[2].querySelector("a[href]")?.getAttribute("href") || "";
      if (taskName && href) map.set(taskName, href);
    }
    return map;
  }

  const META_BY_ACTIVITY = new Map([
    // From the provided Activity -> Meta-category sheet (normalized keys)
    ["washing produce", "food prep"],
    ["defrosting food", "food prep"],
    ["preparing sandwiches", "food prep"],
    ["mixing ingredients", "food prep"],
    ["seasoning food", "food prep"],
    ["making salads", "food prep"],
    ["preparing marinades", "food prep"],
    ["measuring ingredients", "food prep"],
    ["chopping vegetables", "food prep"],
    ["slicing meat", "food prep"],

    ["boiling water", "cooking"],
    ["sauteing vegetables", "cooking"],
    ["frying foods", "cooking"],
    ["steaming vegetables", "cooking"],
    ["microwaving foods", "cooking"],
    ["toasting bread", "cooking"],
    ["slow cooking", "cooking"],
    ["baking", "cooking"],
    ["broiling fish", "cooking"],
    ["simmering sauces", "cooking"],

    ["brewing coffee", "beverage preparation"],
    ["making tea", "beverage preparation"],
    ["making smoothies", "beverage preparation"],
    ["making juice", "beverage preparation"],
    ["preparing hot chocolate", "beverage preparation"],
    ["mixing drinks", "beverage preparation"],
    ["adding ice to beverages", "beverage preparation"],

    ["arranging cabinets", "organizing and storage"],
    ["stocking supplies", "organizing and storage"],
    ["organizing dishes and containers", "organizing and storage"],
    ["sorting ingredients", "organizing and storage"],
    ["organizing utensils", "organizing and storage"],
    ["loading refrigerator", "organizing and storage"],
    ["storing leftovers", "organizing and storage"],
    ["managing freezer space", "organizing and storage"],

    ["setting the table", "serving"],
    ["plating food", "serving"],
    ["portioning meals", "serving"],
    ["filling serving dishes", "serving"],
    ["serving beverages", "serving"],
    ["arranging buffet", "serving"],
    ["packing lunches", "serving"],
    ["arranging condiments", "serving"],
    ["garnishing dishes", "serving"],

    ["washing dishes", "cleaning and sanitizing"],
    ["loading dishwasher", "cleaning and sanitizing"],
    ["sanitizing surfaces", "cleaning and sanitizing"],
    ["cleaning appliances", "cleaning and sanitizing"],
    ["organizing recycling", "cleaning and sanitizing"],
    ["cleaning sink", "cleaning and sanitizing"],
    ["sanitizing cutting boards", "cleaning and sanitizing"],
  ]);

  function normalizeActivityName(s) {
    return (s || "")
      .toLowerCase()
      .normalize("NFD")
      // remove diacritics (e.g. sautéing)
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function metaCategoryForActivityTitle(title) {
    const n = normalizeActivityName(title);
    if (META_BY_ACTIVITY.has(n)) return META_BY_ACTIVITY.get(n);

    // Common aliases / pluralization variants vs our auto-generated headings
    const aliases = [
      [n, n.replace("washing fruits and vegetables", "washing produce")],
      [n, n.replace("preparing marinade", "preparing marinades")],
      [n, n.replace("loading fridge", "loading refrigerator")],
      [n, n.replace("restocking supplies", "stocking supplies")],
      [n, n.replace("organizing dishes and containers", "organizing dishes and containers")],
      [n, n.replace("Sanitizing Surface", "sanitizing surfaces")],
      [n, n.replace("sanitizing cutting board", "sanitizing cutting boards")],
      [n, n.replace("broiling fish", "broiling fish")],
      [n, n.replace("brewing", "brewing coffee")],
      [n, n.replace("frying", "frying foods")],
      [n, n.replace("boiling", "boiling water")],
    ];
    for (const [, cand] of aliases) {
      if (META_BY_ACTIVITY.has(cand)) return META_BY_ACTIVITY.get(cand);
    }

    // Heuristic fallback so every activity gets a tag and participates in filtering
    if (/\b(beverage|drink|tea|coffee|juice|smoothie|hot chocolate|ice)\b/.test(n)) return "beverage preparation";
    if (/\b(wash|clean|sanitize|sanitiz|recycl|dishwasher|sink|tidy)\b/.test(n)) return "cleaning and sanitizing";
    if (/\b(fridge|freezer|cabinet|drawer|organizing|arranging|sorting|stock|store|storing|pack)\b/.test(n))
      return "organizing and storage";
    if (/\b(serv|plating|portion|setting the table|buffet)\b/.test(n)) return "serving";
    if (/\b(boil|saute|fry|steam|microwave|toast|bake|broil|simmer|slow cook|reheat)\b/.test(n))
      return "cooking";
    return "food prep";
  }

  const CATEGORY_LABELS = new Map([
    ["food prep", "Food Preparation"],
    ["cooking", "Cooking"],
    ["beverage preparation", "Beverage Preparation"],
    ["organizing and storage", "Organizing and Storage"],
    ["serving", "Serving"],
    ["cleaning and sanitizing", "Cleaning and Sanitizing"],
  ]);

  function categoryLabel(key) {
    return CATEGORY_LABELS.get(key) || key;
  }

  function countTasks(sectionEl) {
    // Count <tbody><tr> rows in the first table under the section, if present.
    const table = sectionEl.querySelector("table");
    if (!table) return null;
    const tbody = table.querySelector("tbody");
    if (!tbody) return null;
    return tbody.querySelectorAll("tr").length;
  }

  function ensureColumns(table) {
    // Add header columns if missing
    const theadRow = table.querySelector("thead tr");
    if (theadRow) {
      let thEls = Array.from(theadRow.querySelectorAll("th"));
      let ths = thEls.map((th) => (th.textContent || "").trim());
      const horizonHeader = "Horizon";
      const skillsHeader = "Skills Involved";
      const hasSubtasks = ths.includes("Subtasks");
      const legacyLengthIdx = ths.indexOf("Length");
      const horizonIdx = ths.indexOf(horizonHeader);
      const hasEpisodeLength = legacyLengthIdx >= 0 || horizonIdx >= 0;
      const hasSkills = ths.includes(skillsHeader);
      const hasVideo = ths.includes("Video");

      // Backwards compat: if a pre-existing column is labeled "Length", rename it to "Horizon"
      if (legacyLengthIdx >= 0 && horizonIdx < 0) {
        const legacyTh = thEls[legacyLengthIdx];
        if (legacyTh) legacyTh.textContent = horizonHeader;
      }

      // Remove the old "Navigation" column if present (we now render Navigation as a tag)
      const navIdx = ths.indexOf("Navigation");
      if (navIdx >= 0) {
        thEls[navIdx]?.remove();
        const bodyRows = Array.from(table.querySelectorAll("tbody tr"));
        for (const tr of bodyRows) {
          const cells = Array.from(tr.children);
          cells[navIdx]?.remove();
        }
        // Recompute after mutation
        thEls = Array.from(theadRow.querySelectorAll("th"));
        ths = thEls.map((th) => (th.textContent || "").trim());
      }

      // Insert Task Skills after Description (we want to show Description before Skills)
      const taskIdx = ths.indexOf("Task");
      const descIdx = ths.indexOf("Description");
      if (!hasSkills && taskIdx >= 0 && descIdx >= 0) {
        const th = document.createElement("th");
        th.textContent = skillsHeader;
        th.style.textAlign = "left";
        theadRow.insertBefore(th, theadRow.children[descIdx].nextSibling);
        // Recompute after mutation
        thEls = Array.from(theadRow.querySelectorAll("th"));
        ths = thEls.map((th2) => (th2.textContent || "").trim());
      }

      // Insert Subtasks/Horizon after Task Skills (if present), otherwise after Description
      const skillsIdx2 = ths.indexOf(skillsHeader);
      const descIdx2 = ths.indexOf("Description");
      const insertAfter =
        skillsIdx2 >= 0 ? theadRow.children[skillsIdx2] : descIdx2 >= 0 ? theadRow.children[descIdx2] : null;
      let refNode = insertAfter ? insertAfter.nextSibling : null;

      // Insert in desired order right after Description
      if (!hasSubtasks) {
        const th = document.createElement("th");
        th.textContent = "Subtasks";
        th.style.textAlign = "left";
        theadRow.insertBefore(th, refNode);
        refNode = th.nextSibling;
      }
      if (!hasEpisodeLength) {
        const th = document.createElement("th");
        th.textContent = horizonHeader;
        th.style.textAlign = "left";
        theadRow.insertBefore(th, refNode);
        refNode = th.nextSibling;
      }

      // Far-right column
      if (!hasVideo) {
        const th = document.createElement("th");
        th.textContent = "Video";
        th.style.textAlign = "left";
        theadRow.appendChild(th);
      }
    }
  }

  function linkifyTaskAndRemoveClassFile(table) {
    const theadRow = table.querySelector("thead tr");
    if (!theadRow) return;
    const ths = Array.from(theadRow.querySelectorAll("th")).map((th) => (th.textContent || "").trim());
    const taskIdx = ths.indexOf("Task");
    const classIdx = ths.indexOf("Class File");
    if (taskIdx < 0) return;

    const rows = Array.from(table.querySelectorAll("tbody tr"));

    function ensureLinkedTaskCell(taskTd, href) {
      if (!taskTd || !href) return;
      if (taskTd.querySelector("a[href]")) return;

      const code = taskTd.querySelector("code");
      const a = document.createElement("a");
      a.href = href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";

      if (code) {
        // Insert the link right where the code element is, then move the code into it.
        taskTd.insertBefore(a, code);
        a.appendChild(code);
        return;
      }

      // Fallback: wrap plain text content (best-effort).
      const txt = (taskTd.textContent || "").trim();
      if (!txt) return;
      a.textContent = txt;
      taskTd.textContent = "";
      taskTd.appendChild(a);
    }

    if (classIdx >= 0) {
      // Legacy tables: "Class File" column contains the canonical source link.
      for (const tr of rows) {
        const tds = Array.from(tr.children);
        const taskTd = tds[taskIdx];
        const classTd = tds[classIdx];
        if (!taskTd || !classTd) continue;

        const srcLink = classTd.querySelector("a[href]");
        if (srcLink) ensureLinkedTaskCell(taskTd, srcLink.href);
      }

      // Remove header + cells (delete column)
      theadRow.children[classIdx]?.remove();
      for (const tr of rows) {
        tr.children[classIdx]?.remove();
      }
      return;
    }

    // JSON-rendered / auto-generated tables: no "Class File" column.
    // Synthesize a link from (activity title, task name).
    const details = table.closest("details.rc-activity");
    const activityTitle = details ? getActivityTitleFromDetails(details) : "";
    for (const tr of rows) {
      const taskTd = tr.children?.[taskIdx] || null;
      const taskName = taskNameFromRow(tr);
      if (!taskTd || !taskName) continue;
      const href = sourceUrlForTask(activityTitle, taskName);
      if (href) ensureLinkedTaskCell(taskTd, href);
    }
  }

  function taskNameFromRow(tr) {
    // First cell contains code element with the task name
    const firstCell = tr.querySelector("td");
    if (!firstCell) return null;
    const code = firstCell.querySelector("code");
    const name = (code ? code.textContent : firstCell.textContent) || "";
    // Strip any UI-only suffix (e.g. composite asterisk marker)
    const cleaned = name.replace(/\s*[*★]\s*$/, "").trim();
    return cleaned || null;
  }

  function getActivityTitleFromDetails(detailsEl) {
    const t = detailsEl.querySelector("summary .rc-activity-title");
    return (t ? t.textContent : detailsEl.id || "").trim();
  }

  function spacedLowerFromIdentifier(name) {
    // "PastryDisplay" -> "pastry display", "restock_canned_food" -> "restock canned food"
    return (name || "")
      .replace(/_/g, " ")
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .toLowerCase()
      .trim();
  }

  function activityFolderFromTitle(title) {
    // "Adding Ice to Beverages" -> "adding_ice_to_beverages"
    return (title || "")
      .toLowerCase()
      .replace(/&/g, " and ")
      // allow underscores if we're handed a folder-like string
      .replace(/[^a-z0-9_\\s-]/g, "")
      .trim()
      .replace(/[\\s-]+/g, "_")
      .replace(/_+/g, "_");
  }

  function snakeFromTaskName(taskName) {
    // "MakeIceLemonade" -> "make_ice_lemonade"
    return (taskName || "")
      .replace(/-/g, "_")
      .replace(/\\s+/g, "_")
      .replace(/([A-Z]+)([A-Z][a-z0-9])/g, "$1_$2")
      .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
      .toLowerCase()
      .replace(/_+/g, "_")
      .trim();
  }

  const TASK_FILE_BASE_OVERRIDES = new Map([
    // Fix local filename typos / variants in robocasa/environments/kitchen/composite/
    ["AddSweetener", "add_sweetner"],
    ["AirDryFruit", "airdry_fruit"],
    ["OrganizeMetallicUtensils", "organize_metalic_utensils"],
    ["SetUpSpiceStation", "setup_spice_station"],
    ["StackBowlsInSink", "stack_bowls"],
  ]);

  function sourceUrlForTask(activityTitle, taskName) {
    // Derive folder from actual composite directory naming conventions.
    const activityFolder = folderForActivityTitle(activityTitle) || activityFolderFromTitle(activityTitle);
    const base = TASK_FILE_BASE_OVERRIDES.get(taskName) || snakeFromTaskName(taskName);
    if (!activityFolder || !base) return null;
    return `https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/composite/${activityFolder}/${base}.py`;
  }

  function uniqueTokens(query) {
    const raw = (query || "")
      .toLowerCase()
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    return Array.from(new Set(raw));
  }

  function buildTaskIndex(contentRoot) {
    const items = [];
    const details = Array.from(contentRoot.querySelectorAll("details.rc-activity"));
    for (const d of details) {
      const activityId = d.id;
      const activityTitle = getActivityTitleFromDetails(d);
      const metaCategory = d.dataset.metaCategory || null;
      const activityTitleLower = activityTitle.toLowerCase();
      const rows = Array.from(d.querySelectorAll("table tbody tr"));
      for (const tr of rows) {
        const task = taskNameFromRow(tr);
        if (!task) continue;
        const taskLower = task.toLowerCase();
        const taskSpacedLower = spacedLowerFromIdentifier(task);
        // Search text includes task, spaced task, and activity
        const searchText = `${taskLower} ${taskSpacedLower} ${activityTitleLower}`;
        const tds = tr.querySelectorAll("td");
        // With Task Skills column after Description, Description is the 2nd cell
        const descText = (tds.length >= 2 ? tds[1]?.textContent : "") || "";
        // Prefer dataset values (fast path) so filtering/indexing doesn't depend on JS-inserted columns.
        const mobileTd = tr.querySelector("td.rc-mobile");
        const subtasksTd = tr.querySelector("td.rc-subtasks");
        const episodeLengthTd = tr.querySelector("td.rc-episode-length");
        const mobile = (tr.dataset.rcMobile || (mobileTd ? mobileTd.textContent : "") || "").trim();
        const subtasksText = (tr.dataset.rcSubtasks || (subtasksTd ? subtasksTd.textContent : "") || "").trim();
        const subtasks = Number.parseInt(subtasksText, 10);
        const lengthText = (tr.dataset.rcLength || (episodeLengthTd ? episodeLengthTd.textContent : "") || "").trim();
        // Extract numeric value from "32" or "32s"
        let length = null;
        if (lengthText && lengthText !== "—") {
          const match = lengthText.match(/^(\d+)s?$/);
          if (match) {
            length = Number.parseInt(match[1], 10);
            if (!Number.isFinite(length)) length = null;
          }
        }
        items.push({
          task,
          taskLower,
          taskSpacedLower,
          searchText,
          mobile: mobile === "Yes" || mobile === "No" ? mobile : null,
          tagKeys: getTaskTagKeys(task, mobile === "Yes" || mobile === "No" ? mobile : null, descText),
          subtasks: Number.isFinite(subtasks) ? subtasks : null,
          length: Number.isFinite(length) ? length : null,
          metaCategory,
          activityId,
          activityTitle,
          detailsEl: d,
          rowEl: tr,
        });
      }
    }
    return items;
  }

  // Optional per-task tags shown under the task name (Composite Tasks page only).
  // Default applies to all tasks, unless overridden in TASK_TAGS.
  const DEFAULT_TASK_TAGS = ["PickPlace"];
  // Format: taskName -> array of tag keys (rendered as pills).
  // Use [] to explicitly disable tags for a specific task.
  const TASK_TAGS = new Map([
    // Manual skill overrides (additive alongside inferred tags)
    ["MakeIceLemonade", ["PickPlace", "door_open"]],
    ["ArrangeBuffetDessert", ["PickPlace", "door_open"]],
    ["CutBuffetPizza", ["PickPlace", "drawer_open"]],
    ["DivideBuffetTrays", ["PickPlace", "door_open"]],
    ["TongBuffetSetup", ["PickPlace", "drawer_open"]],
    ["CoolBakedCake", ["PickPlace", "rack_slide", "door_open"]],
    ["CoolBakedCookies", ["PickPlace", "rack_slide"]],
    ["MixCakeFrosting", ["PickPlace", "knob_twist", "stand_mixer_close"]],
    ["OvenBroilFish", ["PickPlace", "rack_slide"]],
    ["RemoveBroiledFish", ["PickPlace", "door_open", "rack_slide"]],
    ["ToasterOvenBroilFish", ["PickPlace", "rack_slide", "knob_twist", "door_close"]],
    // Remove Pick & Place from this task specifically, but keep the desired skill
    ["StartElectricKettle", ["button_press", "kettle_lid_close"]],
    ["MeatTransfer", ["PickPlace", "door_open"]],
    ["CuttingToolSelection", ["PickPlace", "drawer_open"]],
    ["ChooseRipeFruit", ["PickPlace", "blender_lid_open"]],
    ["ClearReceptaclesForCleaning", ["PickPlace", "lever_turn"]],
    ["MicrowaveThawing", ["PickPlace", "door_open", "door_close", "button_press"]],
    ["MicrowaveThawingFridge", ["PickPlace", "door_close", "button_press"]],
    ["ThawInSink", ["PickPlace", "lever_turn"]],
    ["AddLemonToFish", ["PickPlace", "door_open"]],
    ["GarnishPancake", ["PickPlace", "door_open"]],
    ["FillBlenderJug", ["PickPlace", "lever_turn"]],
    ["WashLettuce", ["PickPlace", "lever_turn"]],
    ["ReturnHeatedFood", ["PickPlace", "door_open"]],
    ["LoadDishwasher", ["PickPlace", "rack_slide"]],
    ["AfterwashSorting", ["PickPlace", "lever_turn"]],
    ["AirDryFruit", ["PickPlace", "lever_turn"]],
    ["BlendIngredients", ["PickPlace", "button_press", "blender_lid_open", "blender_lid_close"]],
    ["BlendSalsaMix", ["PickPlace", "blender_lid_open", "blender_lid_close"]],
    ["BlendVegetableSauce", ["PickPlace", "button_press", "blender_lid_close"]],
    ["PrepareVeggieDip", ["PickPlace", "button_press", "blender_lid_close"]],
    ["ClusterUtensilsInDrawer", ["PickPlace", "drawer_open"]],
    ["PlateStoreDinner", ["PickPlace", "door_open"]],
    ["BlendMarinade", ["PickPlace", "button_press", "blender_lid_close"]],
    ["PlaceMeatInMarinade", ["PickPlace", "door_open"]],
    ["PlaceStraw", ["PickPlace", "drawer_open"]],
    ["HeatKebabSandwich", ["PickPlace", "knob_twist", "rack_slide", "door_close"]],
    ["HotDogSetup", ["PickPlace", "door_open"]],
    ["PrepareSausageCheese", ["PickPlace", "door_open"]],
    ["ToastHeatableIngredients", ["PickPlace", "rack_slide", "door_open"]],
    ["MakeLoadedPotato", ["PickPlace", "door_open"]],
    ["WaffleReheat", ["PickPlace", "button_press"]],
    ["RefillCondimentStation", ["PickPlace", "door_open"]],
    ["RestockSinkSupplies", ["PickPlace", "drawer_open"]],
    ["CleanMicrowave", ["PickPlace", "door_open"]],
    ["PrepForSanitizing", ["PickPlace", "door_open"]],
    ["RinseCuttingBoard", ["lever_turn"]],
    ["LemonSeasoningFish", ["PickPlace", "door_open"]],
    ["DeliverStraw", ["PickPlace", "drawer_open"]],
    ["PrepareCocktailStation", ["PickPlace", "door_open"]],
    ["PrepareDishwasher", ["rack_slide"]],
    ["PreSoakPan", ["PickPlace", "lever_turn"]],
    ["RinseBowls", ["PickPlace", "lever_turn"]],
    ["SoakSponge", ["PickPlace", "lever_turn"]],
    ["DrainVeggies", ["PickPlace", "lever_turn"]],
    ["PrewashFoodAssembly", ["PickPlace", "lever_turn"]],
    ["WashFruitColander", ["PickPlace", "lever_turn"]],
    ["ArrangeBreadBowl", ["PickPlace", "rack_slide"]],
    ["DateNight", ["PickPlace", "door_open"]],
    ["SeasoningSpiceSetup", ["PickPlace", "door_open"]],
    ["SetBowlsForSoup", ["PickPlace", "door_open"]],
    ["SetupButterPlate", ["PickPlace", "door_open"]],
    ["SetupFruitBowl", ["PickPlace", "door_open"]],
    ["SetUpCuttingStation", ["PickPlace", "drawer_open"]],
    ["BeginSlowCooking", ["PickPlace", "knob_twist"]],
    ["MultistepSteaming", ["PickPlace", "lever_turn"]],
    ["ServeWarmCroissant", ["PickPlace", "rack_slide", "door_open"]],
    ["ToastBagel", ["PickPlace", "rack_slide", "knob_twist", "door_close"]],
    ["ToastBaguette", ["PickPlace", "rack_slide", "knob_twist", "door_close"]],
    ["ToastOnCorrectRack", ["PickPlace", "rack_slide", "door_open"]],
    ["GetToastedBread", ["PickPlace", "start_toaster"]],
    ["ToastOneSlotPair", ["PickPlace", "start_toaster"]],
    ["MakeCheesecakeFilling", ["PickPlace", "knob_twist", "stand_mixer_close"]],
    ["ClearClutter", ["PickPlace", "lever_turn"]],
  ]);

  // Per-task tag removals applied after auto-inference.
  // Format: taskName -> array of tag keys to remove.
  const TASK_TAG_REMOVALS = new Map([
    // Reset tasks mention "open" in the description, but we don't want to show Open Door here.
    ["ResetCabinetDoors", ["door_open", "PickPlace"]],
    ["CandleCleanup", ["door_open"]],
    ["DrinkwareConsolidation", ["door_open"]],
    // "Press" here is not a button press; avoid incorrect Press Button tag.
    ["PressChicken", ["button_press"]],
    // Prefer lever turning over knob twisting for this task.
    ["ThawInSink", ["knob_twist"]],
    ["FillBlenderJug", ["knob_twist"]],
    ["AfterwashSorting", ["knob_twist"]],
    ["AirDryFruit", ["knob_twist"]],
    ["ClearClutter", ["knob_twist"]],
    ["WashFruitColander", ["knob_twist"]],
    ["EmptyDishRack", ["door_open"]],
    ["StackBowlsCabinet", ["door_open"]],
    ["BlendMarinade", ["door_close"]],
    ["CountertopCleanup", ["drawer_open"]],
    ["RinseCuttingBoard", ["knob_twist", "button_press", "PickPlace"]],
    ["MultistepSteaming", ["knob_twist"]],
    ["PlaceBreakfastItemsAway", ["door_open"]],
    ["UtensilShuffle", ["drawer_open"]],
    ["SortingCleanup", ["door_open"]],
    ["ToastOneSlotPair", ["button_press"]],
    ["RinseSinkBasin", ["PickPlace"]],
    ["TurnOffSimmeredSauceHeat", ["PickPlace"]],
    // Explicit: this task should be Pick & Place only (no inferred open/close door or navigation).
    ["OrganizeMugsByHandle", ["nav", "door_open", "door_close"]],
  ]);

  // Tag display order (should match the Task Attributes dropdown order)
  const TAG_ORDER = [
    "comp_seen_target",
    "comp_unseen_target",
    "nav",
    "door_open",
    "door_close",
    "drawer_open",
    "drawer_close",
    "start_toaster",
    "stand_mixer_close",
    "blender_lid_open",
    "blender_lid_close",
    "kettle_lid_close",
    "rack_slide",
    "knob_twist",
    "lever_turn",
    "button_press",
    "PickPlace",
  ];

  // Target task lists (from robocasa/utils/dataset_registry.py)
  const COMP_SEEN_TARGET_TASKS = new Set([
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WashLettuce",
  ]);

  const COMP_UNSEEN_TARGET_TASKS = new Set([
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CategorizeCondiments",
    "CuttingToolSelection",
    "GarnishPancake",
    "GatherTableware",
    "HeatKebabSandwich",
    "MakeIceLemonade",
    "PanTransfer",
    "PortionHotDogs",
    "RecycleBottlesByType",
    "SeparateFreezerRack",
    "WaffleReheat",
    "WashFruitColander",
    "WeighIngredients",
  ]);

  function inferSkillTagKeys(taskName, descText) {
    const t = String(taskName || "").toLowerCase();
    const d = String(descText || "").toLowerCase();
    const out = [];

    const isOpen = /\bopen\b/.test(d) || t.startsWith("open");
    const isClose = /\b(close|shut)\b/.test(d) || t.startsWith("close") || t.startsWith("reset");

    const isDrawer = /\bdrawer\b/.test(d) || t.includes("drawer");
    const isDoorish =
      /\b(door|cabinet|fridge|freezer|dishwasher|oven|microwave)\b/.test(d) ||
      /(door|cabinet|fridge|freezer|dishwasher|oven|microwave)/.test(t);

    if (isOpen) {
      if (isDrawer) out.push("drawer_open");
      else if (isDoorish) out.push("door_open");
    }
    if (isClose) {
      if (isDrawer) out.push("drawer_close");
      else if (isDoorish) out.push("door_close");
    }

    const hasTurn = /\b(turn|twist|rotate|adjust)\b/.test(d) || t.startsWith("turn") || t.includes("adjust");
    const knobCtx =
      /\b(knob|dial|temperature|stove|burner|faucet)\b/.test(d) ||
      /(temperature|stove|toaster|oven)/.test(t);
    const leverCtx = /\b(spout|lever)\b/.test(d) || t.includes("spout") || t.includes("lever");

    if (hasTurn && knobCtx) out.push("knob_twist");
    if (hasTurn && leverCtx) out.push("lever_turn");

    const presses =
      /\b(press|push)\b/.test(d) ||
      /\bbutton\b/.test(d) ||
      t.includes("press") ||
      t.includes("button");
    if (presses) out.push("button_press");

    return out;
  }

  function uniquePush(arr, key) {
    if (!arr.includes(key)) arr.push(key);
  }

  function formatCount(n) {
    if (!Number.isFinite(n) || n <= 0) return "";
    return `(${n})`;
  }

  function getTaskTagKeys(taskName, navigationVal, descText) {
    const override = TASK_TAGS.get(taskName);
    const tags = override != null ? override.slice() : DEFAULT_TASK_TAGS.slice();
    if (!tags || tags.length === 0) return [];

    // Prepend target-task tags (must appear before any other tags)
    if (COMP_SEEN_TARGET_TASKS.has(taskName) && !tags.includes("comp_seen_target")) {
      tags.unshift("comp_seen_target");
    }
    if (COMP_UNSEEN_TARGET_TASKS.has(taskName) && !tags.includes("comp_unseen_target")) {
      tags.unshift("comp_unseen_target");
    }

    // Auto-add Navigation tag based on task attributes ("Yes" / "No")
    if (navigationVal === "Yes" && !tags.includes("nav")) tags.push("nav");

    // Skill tags inferred from task name + description
    for (const s of inferSkillTagKeys(taskName, descText)) uniquePush(tags, s);

    // Apply per-task removals (if any)
    const removals = TASK_TAG_REMOVALS.get(taskName);
    if (removals && removals.length) {
      for (const r of removals) {
        for (let i = tags.length - 1; i >= 0; i -= 1) {
          if (tags[i] === r) tags.splice(i, 1);
        }
      }
    }

    // Ensure skill precedence matches the Task Attributes dropdown (without injecting tags).
    const out = [];
    const seen = new Set();
    function pushOnce(t) {
      if (!t || seen.has(t)) return;
      seen.add(t);
      out.push(t);
    }
    for (const t of TAG_ORDER) if (tags.includes(t)) pushOnce(t);
    for (const t of tags) pushOnce(t);
    return out;
  }

  function renderCompositeTargetTagIntoTaskCell(taskName, taskTd) {
    if (!taskName || !taskTd) return;

    // Idempotent: remove any existing indicators first (legacy stars + current tags)
    for (const old of Array.from(taskTd.querySelectorAll(".rc-comp-star"))) old.remove();
    taskTd.querySelector(".rc-comp-target-tagline")?.remove();

    // Datasets overview page already separates Composite Seen/Unseen tables,
    // so showing the tag under each task name is redundant.
    if (document.body.classList.contains("rc-datasets-overview")) return;

    const isSeen = COMP_SEEN_TARGET_TASKS.has(taskName);
    const isUnseen = COMP_UNSEEN_TARGET_TASKS.has(taskName);
    if (!isSeen && !isUnseen) return;

    const line = document.createElement("div");
    line.className = "rc-comp-target-tagline";

    const pill = document.createElement("span");
    pill.className = `rc-task-tag ${isSeen ? "rc-task-tag-comp-seen" : "rc-task-tag-comp-unseen"}`;
    pill.textContent = isSeen ? "Composite-Seen" : "Composite-Unseen";
    pill.setAttribute("aria-label", pill.textContent);

    line.appendChild(pill);
    // Place the tag under the task name (below the link / code)
    taskTd.appendChild(line);
  }

  function rcCanMeasureSkillPack(el) {
    // If the activity <details> is closed, the table isn't laid out yet and
    // offsetTop measurements will all be 0. Defer packing until visible.
    try {
      return !!(el && el.isConnected && el.getClientRects && el.getClientRects().length > 0 && el.offsetWidth > 0);
    } catch {
      return false;
    }
  }

  function rcPackSkillTagsInWrap(tagWrap) {
    if (!tagWrap) return;
    // Fast path: 0-2 tags can't benefit from reordering.
    const pills = Array.from(tagWrap.children || []).filter((n) => n && n.nodeType === 1);
    if (pills.length <= 2) return;

    function render(order) {
      tagWrap.innerHTML = "";
      for (const p of order) tagWrap.appendChild(p);
    }

    function layoutInfo(order) {
      const tops = [];
      for (const p of order) tops.push(p.offsetTop);
      const uniqueTops = Array.from(new Set(tops)).sort((a, b) => a - b);
      const lineByIdx = tops.map((t) => uniqueTops.indexOf(t));
      const lineCounts = new Array(uniqueTops.length).fill(0);
      for (const li of lineByIdx) if (li >= 0) lineCounts[li] += 1;
      const lineCount = uniqueTops.length;
      const orphanLines = lineCounts.reduce((acc, c) => acc + (c === 1 ? 1 : 0), 0);
      const firstLineCount = lineCounts.length ? lineCounts[0] : 0;
      return { lineCount, orphanLines, firstLineCount, lineByIdx, lineCounts };
    }

    function sameLineForClasses(info, ord, classA, classB) {
      const a = ord.find((p) => p.classList && p.classList.contains(classA));
      const b = ord.find((p) => p.classList && p.classList.contains(classB));
      if (!a || !b) return null;
      const ai = ord.indexOf(a);
      const bi = ord.indexOf(b);
      const al = info.lineByIdx[ai];
      const bl = info.lineByIdx[bi];
      if (!Number.isFinite(al) || !Number.isFinite(bl)) return null;
      return al === bl;
    }

    function isBetter(aInfo, aOrder, bInfo, bOrder) {
      // Prefer fewer total lines.
      if (aInfo.lineCount !== bInfo.lineCount) return aInfo.lineCount < bInfo.lineCount;
      // If lines tie, prefer coupling "Navigation" + "Twist Knob" when possible.
      const aCoupled = sameLineForClasses(aInfo, aOrder, "rc-task-tag-nav", "rc-task-tag-knob-twist");
      const bCoupled = sameLineForClasses(bInfo, bOrder, "rc-task-tag-nav", "rc-task-tag-knob-twist");
      if (aCoupled != null && bCoupled != null && aCoupled !== bCoupled) return aCoupled === true;
      // Then fewer 1-tag lines, then more tags on first line.
      if (aInfo.orphanLines !== bInfo.orphanLines) return aInfo.orphanLines < bInfo.orphanLines;
      if (aInfo.firstLineCount !== bInfo.firstLineCount) return aInfo.firstLineCount > bInfo.firstLineCount;
      return false;
    }

    function moveItem(arr, fromIdx, toIdx) {
      if (fromIdx === toIdx) return arr.slice();
      const out = arr.slice();
      const [item] = out.splice(fromIdx, 1);
      out.splice(toIdx, 0, item);
      return out;
    }

    // Start with the current (canonical) order.
    let order = pills.slice();
    render(order);
    let best = layoutInfo(order);
    let bestOrder = order;
    if (best.lineCount <= 1) return;

    // Record baseline Navigation line index (if present); keep Navigation from
    // moving to a later line than it started on.
    let navBaselineLine = null;
    const navEl = order.find((p) => p.classList && p.classList.contains("rc-task-tag-nav"));
    if (navEl) navBaselineLine = layoutInfo(order).lineByIdx[order.indexOf(navEl)] ?? null;

    function navOk(info, ord) {
      if (navBaselineLine == null) return true;
      const nav = ord.find((p) => p.classList && p.classList.contains("rc-task-tag-nav"));
      if (!nav) return true;
      const idx = ord.indexOf(nav);
      const line = info.lineByIdx[idx];
      return Number.isFinite(line) ? line <= navBaselineLine : true;
    }

    // Iteratively improve by pulling a later tag upward to fill earlier lines.
    const maxIters = Math.max(8, order.length * order.length);
    let iter = 0;
    while (iter < maxIters) {
      iter += 1;
      render(order);
      const cur = layoutInfo(order);
      if (cur.lineCount <= 1) break;

      const lineToIdxs = new Map();
      for (let i = 0; i < order.length; i += 1) {
        const li = cur.lineByIdx[i];
        if (li < 0) continue;
        if (!lineToIdxs.has(li)) lineToIdxs.set(li, []);
        lineToIdxs.get(li).push(i);
      }

      let improved = false;
      let bestCandidate = null;
      for (let line = 0; line < cur.lineCount - 1; line += 1) {
        const idxs = lineToIdxs.get(line) || [];
        if (idxs.length === 0) continue;
        const insertAfter = idxs[idxs.length - 1];
        const insertPos = insertAfter + 1;
        for (let from = insertPos; from < order.length; from += 1) {
          const trialOrder = moveItem(order, from, insertPos);
          render(trialOrder);
          const info = layoutInfo(trialOrder);
          if (!navOk(info, trialOrder)) continue;
          if (info.lineCount > best.lineCount) continue;
          if (isBetter(info, trialOrder, best, bestOrder)) {
            bestCandidate = { order: trialOrder, info };
            best = info;
            bestOrder = trialOrder;
            improved = true;
          }
        }
      }

      if (improved && bestCandidate) order = bestCandidate.order;
      else break;
    }

    render(order);
  }

  function renderTaskTagsIntoCell(taskName, containerTd, navigationVal, descText) {
    if (!taskName || !containerTd) return;
    if (containerTd.querySelector(".rc-task-tags")) return; // idempotent

    const tags = getTaskTagKeys(taskName, navigationVal, descText);
    if (!tags || tags.length === 0) return;

    const wrap = document.createElement("div");
    wrap.className = "rc-task-tags";

    function pillForTag(t) {
      const span = document.createElement("span");
      span.className = "rc-task-tag";
      if (t === "PickPlace") {
        span.classList.add("rc-task-tag-PickPlace");
        span.textContent = "Pick & Place";
        return span;
      }
      // Composite target (seen/unseen) is shown under the task name (not in Task Skills)
      if (t === "comp_seen_target" || t === "comp_unseen_target") return null;
      if (t === "nav") {
        span.classList.add("rc-task-tag-nav");
        span.textContent = "Navigation";
        return span;
      }
      if (t === "door_open") {
        span.classList.add("rc-task-tag-door-open");
        span.textContent = "Open Door";
        return span;
      }
      if (t === "door_close") {
        span.classList.add("rc-task-tag-door-close");
        span.textContent = "Close Door";
        return span;
      }
      if (t === "drawer_open") {
        span.classList.add("rc-task-tag-drawer-open");
        span.textContent = "Open Drawer";
        return span;
      }
      if (t === "drawer_close") {
        span.classList.add("rc-task-tag-drawer-close");
        span.textContent = "Close Drawer";
        return span;
      }
      // Deprecated toaster-oven tags: render as generic Open/Close Door instead
      if (t === "toaster_oven_open") {
        span.classList.add("rc-task-tag-door-open");
        span.textContent = "Open Door";
        return span;
      }
      if (t === "toaster_oven_close") {
        span.classList.add("rc-task-tag-door-close");
        span.textContent = "Close Door";
        return span;
      }
      if (t === "start_toaster") {
        span.classList.add("rc-task-tag-start-toaster");
        span.textContent = "Start Toaster";
        return span;
      }
      if (t === "stand_mixer_close") {
        span.classList.add("rc-task-tag-stand-mixer-close");
        span.textContent = "Close Stand Mixer";
        return span;
      }
      if (t === "blender_lid_open") {
        span.classList.add("rc-task-tag-blender-lid-open");
        span.textContent = "Open Blender Lid";
        return span;
      }
      if (t === "blender_lid_close") {
        span.classList.add("rc-task-tag-blender-lid-close");
        span.textContent = "Close Blender Lid";
        return span;
      }
      if (t === "kettle_lid_close") {
        span.classList.add("rc-task-tag-kettle-lid-close");
        span.textContent = "Close Kettle Lid";
        return span;
      }
      if (t === "knob_twist") {
        span.classList.add("rc-task-tag-knob-twist");
        span.textContent = "Twist Knob";
        return span;
      }
      if (t === "lever_turn") {
        span.classList.add("rc-task-tag-lever-turn");
        span.textContent = "Turn Lever";
        return span;
      }
      if (t === "button_press") {
        span.classList.add("rc-task-tag-button-press");
        span.textContent = "Press Button";
        return span;
      }
      if (t === "rack_slide") {
        span.classList.add("rc-task-tag-rack-slide");
        span.textContent = "Slide Rack";
        return span;
      }
      return null;
    }

    const pills = [];
    for (const t of tags) {
      const pill = pillForTag(t);
      if (pill) pills.push(pill);
    }

    if (pills.length === 0) return;

    // Insert into the provided container cell (needed for layout measurements)
    containerTd.appendChild(wrap);
    // First render in canonical order.
    for (const p of pills) wrap.appendChild(p);

    // Then, if the activity is visible (table laid out), pack tags by only pulling later tags upward.
    // If not visible (closed <details>), defer until it is opened.
    if (rcCanMeasureSkillPack(wrap)) {
      // Defer one frame so the browser computes the initial line wraps.
      window.requestAnimationFrame(() => rcPackSkillTagsInWrap(wrap));
    } else {
      wrap.dataset.rcNeedsSkillPack = "1";
    }
  }

  function reorderActivityDropdownOptions(selectEl, detailsEls) {
    const current = selectEl.value;
    const defaultOpt = selectEl.querySelector('option[value=""]');
    const optById = new Map();
    for (const opt of Array.from(selectEl.querySelectorAll("option"))) {
      if (!opt.value) continue;
      optById.set(opt.value, opt);
    }
    // Rebuild options in new order
    selectEl.innerHTML = "";
    if (defaultOpt) selectEl.appendChild(defaultOpt);
    for (const d of detailsEls) {
      const id = d.id;
      const opt = optById.get(id);
      if (opt) selectEl.appendChild(opt);
    }
    // Restore selection if possible
    if (current && optById.has(current)) {
      selectEl.value = current;
    } else {
      // Force placeholder when no valid selection (some browsers restore form state
      // for dynamically-created selects, which can make the first option appear selected).
      selectEl.value = "";
      const def = selectEl.querySelector('option[value=""]');
      if (def) def.selected = true;
    }
  }

  function sortActivities(rootSectionEl, mode, selectEl) {
    const detailsEls = Array.from(rootSectionEl.querySelectorAll(":scope > details.rc-activity"));
    if (mode === "category") {
      detailsEls.sort((a, b) => {
        const ac = categoryLabel(a.dataset.metaCategory || "").toLowerCase();
        const bc = categoryLabel(b.dataset.metaCategory || "").toLowerCase();
        if (ac !== bc) return ac.localeCompare(bc);
        const at = getActivityTitleFromDetails(a).toLowerCase();
        const bt = getActivityTitleFromDetails(b).toLowerCase();
        return at.localeCompare(bt);
      });
    } else {
      // activity: alphabetical by activity title
      detailsEls.sort((a, b) => {
        const at = getActivityTitleFromDetails(a).toLowerCase();
        const bt = getActivityTitleFromDetails(b).toLowerCase();
        return at.localeCompare(bt);
      });
    }

    for (const d of detailsEls) rootSectionEl.appendChild(d);
    reorderActivityDropdownOptions(selectEl, detailsEls);
  }

  function highlightRow(rowEl) {
    rowEl.classList.remove("rc-task-highlight");
    // force reflow so re-adding retriggers animation
    void rowEl.offsetWidth; // eslint-disable-line no-unused-expressions
    rowEl.classList.add("rc-task-highlight");
    window.setTimeout(() => rowEl.classList.remove("rc-task-highlight"), 1800);
  }

  function getContentRoot() {
    const root = document.documentElement.getAttribute("data-content_root") || "../";
    return root.endsWith("/") ? root : `${root}/`;
  }

  /* const VIDEO_BY_TASK = new Map([
    ["AddIceCubes", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AddIceCubes.mp4` }],
    ["AddLemonToFish", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AddLemonToFish.mp4` }],
    ["AddMarshmallow", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AddMarshmallow.mp4` }],
    ["AddSugarCubes", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AddSugarCubes.mp4` }],
    ["AddSweetener", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AddSweetener.mp4` }],
    ["AddToSoupPot", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AddToSoupPot.mp4` }],
    ["AdjustHeat", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AdjustHeat.mp4` }],
    ["AfterwashSorting", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AfterwashSorting.mp4` }],
    ["AirDryFruit", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AirDryFruit.mp4` }],
    ["AlcoholServingPrep", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AlcoholServingPrep.mp4` }],
    ["AlignSilverware", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AlignSilverware.mp4` }],
    ["ArrangeBreadBasket", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ArrangeBreadBasket.mp4` }],
    ["ArrangeBreadBowl", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ArrangeBreadBowl.mp4` }],
    ["ArrangeBuffetDessert", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ArrangeBuffetDessert.mp4` }],
    ["ArrangeCuttingFruits", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ArrangeCuttingFruits.mp4` }],
    ["ArrangeDrinkware", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ArrangeDrinkware.mp4` }],
    ["ArrangeSinkSanitization", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ArrangeSinkSanitization.mp4` }],
    ["ArrangeTeaAccompaniments", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ArrangeTeaAccompaniments.mp4` }],
    ["ArrangeUtensilsByType", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ArrangeUtensilsByType.mp4` }],
    ["ArrangeVegetables", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ArrangeVegetables.mp4` }],
    ["AssembleCookingArray", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/AssembleCookingArray.mp4` }],
    ["BalancedMealPrep", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BalancedMealPrep.mp4` }],
    ["BeginSlowCooking", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BeginSlowCooking.mp4` }],
    ["BeverageOrganization", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BeverageOrganization.mp4` }],
    ["BeverageSorting", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BeverageSorting.mp4` }],
    ["BlendIngredients", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BlendIngredients.mp4` }],
    ["BlendMarinade", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BlendMarinade.mp4` }],
    ["BlendSalsaMix", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BlendSalsaMix.mp4` }],
    ["BlendVegetableSauce", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BlendVegetableSauce.mp4` }],
    ["BoilCorn", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BoilCorn.mp4` }],
    ["BoilEggs", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BoilEggs.mp4` }],
    ["BoilPot", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BoilPot.mp4` }],
    ["BowlAndCup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BowlAndCup.mp4` }],
    ["BreadAndCheese", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BreadAndCheese.mp4` }],
    ["BreadSelection", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BreadSelection.mp4` }],
    ["BreadSetupSlicing", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BreadSetupSlicing.mp4` }],
    ["BuildAppetizerPlate", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/BuildAppetizerPlate.mp4` }],
    ["ButterOnPan", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ButterOnPan.mp4` }],
    ["CandleCleanup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CandleCleanup.mp4` }],
    ["CategorizeCondiments", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CategorizeCondiments.mp4` }],
    ["CerealAndBowl", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CerealAndBowl.mp4` }],
    ["CheeseMixing", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CheeseMixing.mp4` }],
    ["CheesyBread", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CheesyBread.mp4` }],
    ["ChooseMeasuringCup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ChooseMeasuringCup.mp4` }],
    ["ChooseRipeFruit", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ChooseRipeFruit.mp4` }],
    ["CleanBlenderJug", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CleanBlenderJug.mp4` }],
    ["CleanBoard", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CleanBoard.mp4` }],
    ["CleanMicrowave", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CleanMicrowave.mp4` }],
    ["ClearClutter", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClearClutter.mp4` }],
    ["ClearCuttingBoard", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClearCuttingBoard.mp4` }],
    ["ClearFoodWaste", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClearFoodWaste.mp4` }],
    ["ClearFreezer", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClearFreezer.mp4` }],
    ["ClearReceptaclesForCleaning", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClearReceptaclesForCleaning.mp4` }],
    ["ClearSink", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClearSink.mp4` }],
    ["ClearSinkArea", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClearSinkArea.mp4` }],
    ["ClearSinkSpace", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClearSinkSpace.mp4` }],
    ["ClusterItemsForClearing", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClusterItemsForClearing.mp4` }],
    ["ClusterUtensilsInDrawer", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ClusterUtensilsInDrawer.mp4` }],
    ["CollectWashingSupplies", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CollectWashingSupplies.mp4` }],
    ["ColorfulSalsa", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ColorfulSalsa.mp4` }],
    ["CondimentCollection", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CondimentCollection.mp4` }],
    ["CookieDoughPrep", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CookieDoughPrep.mp4` }],
    ["CoolBakedCake", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CoolBakedCake.mp4` }],
    ["CoolBakedCookies", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CoolBakedCookies.mp4` }],
    ["CoolKettle", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CoolKettle.mp4` }],
    ["CountertopCleanup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CountertopCleanup.mp4` }],
    ["CreateChildFriendlyFridge", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CreateChildFriendlyFridge.mp4` }],
    ["CupcakeCleanup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CupcakeCleanup.mp4` }],
    ["CutBuffetPizza", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CutBuffetPizza.mp4` }],
    ["CuttingToolSelection", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/CuttingToolSelection.mp4` }],
    ["DateNight", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DateNight.mp4` }],
    ["DefrostByCategory", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DefrostByCategory.mp4` }],
    ["DeliverBrewedCoffee", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DeliverBrewedCoffee.mp4` }],
    ["DeliverStraw", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DeliverStraw.mp4` }],
    ["DessertAssembly", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DessertAssembly.mp4` }],
    ["DessertUpgrade", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DessertUpgrade.mp4` }],
    ["DisplayMeatVariety", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DisplayMeatVariety.mp4` }],
    ["DistributeChicken", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DistributeChicken.mp4` }],
    ["DistributeSteakOnPans", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DistributeSteakOnPans.mp4` }],
    ["DivideBasins", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DivideBasins.mp4` }],
    ["DivideBuffetTrays", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DivideBuffetTrays.mp4` }],
    ["DrainVeggies", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DrainVeggies.mp4` }],
    ["DrawerUtensilSort", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DrawerUtensilSort.mp4` }],
    ["DrinkwareConsolidation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DrinkwareConsolidation.mp4` }],
    ["DryDishes", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DryDishes.mp4` }],
    ["DryDrinkware", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DryDrinkware.mp4` }],
    ["DumpLeftovers", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/DumpLeftovers.mp4` }],
    ["EmptyDishRack", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/EmptyDishRack.mp4` }],
    ["FillBlenderJug", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/FillBlenderJug.mp4` }],
    ["FillKettle", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/FillKettle.mp4` }],
    ["FilterMicrowavableItem", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/FilterMicrowavableItem.mp4` }],
    ["FreezeBottledWaters", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/FreezeBottledWaters.mp4` }],
    ["FreezeCookedFood", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/FreezeCookedFood.mp4` }],
    ["FreezeIceTray", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/FreezeIceTray.mp4` }],
    ["FreshProduceOrganization", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/FreshProduceOrganization.mp4` }],
    ["FryingPanAdjustment", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/FryingPanAdjustment.mp4` }],
    ["GarnishCake", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/GarnishCake.mp4` }],
    ["GarnishCupcake", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/GarnishCupcake.mp4` }],
    ["GarnishPancake", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/GarnishPancake.mp4` }],
    ["GatherCuttingTools", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/GatherCuttingTools.mp4` }],
    ["GatherMarinadeIngredients", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/GatherMarinadeIngredients.mp4` }],
    ["GatherProduceWashing", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/GatherProduceWashing.mp4` }],
    ["GatherTableware", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/GatherTableware.mp4` }],
    ["GatherVegetables", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/GatherVegetables.mp4` }],
    ["GetToastedBread", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/GetToastedBread.mp4` }],
    ["HeatKebabSandwich", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/HeatKebabSandwich.mp4` }],
    ["HeatMug", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/HeatMug.mp4` }],
    ["HeatMultipleWater", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/HeatMultipleWater.mp4` }],
    ["HotDogSetup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/HotDogSetup.mp4` }],
    ["JuiceFruitReamer", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/JuiceFruitReamer.mp4` }],
    ["KettleBoiling", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/KettleBoiling.mp4` }],
    ["LemonSeasoningFish", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/LemonSeasoningFish.mp4` }],
    ["LineUpCondiments", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/LineUpCondiments.mp4` }],
    ["LoadCondimentsInFridge", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/LoadCondimentsInFridge.mp4` }],
    ["LoadDishwasher", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/LoadDishwasher.mp4` }],
    ["LoadFridgeByType", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/LoadFridgeByType.mp4` }],
    ["LoadFridgeFifo", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/LoadFridgeFifo.mp4` }],
    ["LoadPreparedFood", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/LoadPreparedFood.mp4` }],
    ["LowerHeat", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/LowerHeat.mp4` }],
    ["MakeBananaMilkshake", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MakeBananaMilkshake.mp4` }],
    ["MakeCheesecakeFilling", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MakeCheesecakeFilling.mp4` }],
    ["MakeChocolateMilk", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MakeChocolateMilk.mp4` }],
    ["MakeFruitBowl", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MakeFruitBowl.mp4` }],
    ["MakeIcedCoffee", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MakeIcedCoffee.mp4` }],
    ["MakeIceLemonade", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MakeIceLemonade.mp4` }],
    ["MakeLoadedPotato", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MakeLoadedPotato.mp4` }],
    ["MatchCupAndDrink", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MatchCupAndDrink.mp4` }],
    ["MaximizeFreezerSpace", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MaximizeFreezerSpace.mp4` }],
    ["MealPrepStaging", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MealPrepStaging.mp4` }],
    ["MeatSkewerAssembly", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MeatSkewerAssembly.mp4` }],
    ["MeatTransfer", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MeatTransfer.mp4` }],
    ["MicrowaveCorrectMeal", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MicrowaveCorrectMeal.mp4` }],
    ["MicrowaveDefrostMeat", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MicrowaveDefrostMeat.mp4` }],
    ["MicrowaveThawing", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MicrowaveThawing.mp4` }],
    ["MicrowaveThawingFridge", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MicrowaveThawingFridge.mp4` }],
    ["MixCakeFrosting", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MixCakeFrosting.mp4` }],
    ["MixedFruitPlatter", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MixedFruitPlatter.mp4` }],
    ["MoveFreezerToFridge", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MoveFreezerToFridge.mp4` }],
    ["MoveFridgeToFreezer", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MoveFridgeToFreezer.mp4` }],
    ["MoveToCounter", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MoveToCounter.mp4` }],
    ["MoveToFreezerDrawer", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MoveToFreezerDrawer.mp4` }],
    ["MultistepSteaming", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/MultistepSteaming.mp4` }],
    ["OrganizeBakingIngredients", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/OrganizeBakingIngredients.mp4` }],
    ["OrganizeCleaningSupplies", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/OrganizeCleaningSupplies.mp4` }],
    ["OrganizeCoffeeCondiments", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/OrganizeCoffeeCondiments.mp4` }],
    ["OrganizeCondiments", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/OrganizeCondiments.mp4` }],
    ["OrganizeMeasuringCups", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/OrganizeMeasuringCups.mp4` }],
    ["OrganizeMetallicUtensils", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/OrganizeMetallicUtensils.mp4` }],
    ["OrganizeMugsByHandle", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/OrganizeMugsByHandle.mp4` }],
    ["OrganizeVegetables", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/OrganizeVegetables.mp4` }],
    ["OvenBroilFish", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/OvenBroilFish.mp4` }],
    ["PackDessert", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PackDessert.mp4` }],
    ["PackFoodByTemp", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PackFoodByTemp.mp4` }],
    ["PackFruitContainer", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PackFruitContainer.mp4` }],
    ["PackIdenticalLunches", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PackIdenticalLunches.mp4` }],
    ["PackSnack", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PackSnack.mp4` }],
    ["PanTransfer", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PanTransfer.mp4` }],
    ["PastryDisplay", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PastryDisplay.mp4` }],
    ["PJSandwichPrep", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PJSandwichPrep.mp4` }],
    ["PlaceBeveragesTogether", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceBeveragesTogether.mp4` }],
    ["PlaceBreakfastItemsAway", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceBreakfastItemsAway.mp4` }],
    ["PlaceDishesBySink", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceDishesBySink.mp4` }],
    ["PlaceEqualIceCubes", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceEqualIceCubes.mp4` }],
    ["PlaceFoodInBowls", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceFoodInBowls.mp4` }],
    ["PlaceIceInCup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceIceInCup.mp4` }],
    ["PlaceLidToBoil", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceLidToBoil.mp4` }],
    ["PlaceMeatInMarinade", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceMeatInMarinade.mp4` }],
    ["PlaceMicrowaveSafeItem", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceMicrowaveSafeItem.mp4` }],
    ["PlaceOnDishRack", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceOnDishRack.mp4` }],
    ["PlaceStraw", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceStraw.mp4` }],
    ["PlaceVegetablesEvenly", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceVegetablesEvenly.mp4` }],
    ["PlaceVeggiesInDrawer", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlaceVeggiesInDrawer.mp4` }],
    ["PlateSteakMeal", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlateSteakMeal.mp4` }],
    ["PlateStoreDinner", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PlateStoreDinner.mp4` }],
    ["PortionFruitBowl", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PortionFruitBowl.mp4` }],
    ["PortionHotDogs", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PortionHotDogs.mp4` }],
    ["PortionInTupperware", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PortionInTupperware.mp4` }],
    ["PortionOnSize", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PortionOnSize.mp4` }],
    ["PortionYogurt", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PortionYogurt.mp4` }],
    ["PreheatPot", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PreheatPot.mp4` }],
    ["PrepareBroilingStation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareBroilingStation.mp4` }],
    ["PrepareCheeseStation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareCheeseStation.mp4` }],
    ["PrepareCocktailStation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareCocktailStation.mp4` }],
    ["PrepareCoffee", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareCoffee.mp4` }],
    ["PrepareDishwasher", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareDishwasher.mp4` }],
    ["PrepareDrinkStation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareDrinkStation.mp4` }],
    ["PrepareSandwichStation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareSandwichStation.mp4` }],
    ["PrepareSausageCheese", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareSausageCheese.mp4` }],
    ["PrepareSmoothie", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareSmoothie.mp4` }],
    ["PrepareSoupServing", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareSoupServing.mp4` }],
    ["PrepareStoringLeftovers", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareStoringLeftovers.mp4` }],
    ["PrepareToast", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareToast.mp4` }],
    ["PrepareVegetableRoasting", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareVegetableRoasting.mp4` }],
    ["PrepareVeggieDip", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareVeggieDip.mp4` }],
    ["PrepareVeggiesForSteaming", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepareVeggiesForSteaming.mp4` }],
    ["PrepForSanitizing", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepForSanitizing.mp4` }],
    ["PrepForTenderizing", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepForTenderizing.mp4` }],
    ["PrepFridgeForCleaning", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepFridgeForCleaning.mp4` }],
    ["PrepMarinatingMeat", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepMarinatingMeat.mp4` }],
    ["PrepSinkForCleaning", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrepSinkForCleaning.mp4` }],
    ["PreRinseStation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PreRinseStation.mp4` }],
    ["PreSoakPan", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PreSoakPan.mp4` }],
    ["PressChicken", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PressChicken.mp4` }],
    ["PrewashFoodAssembly", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrewashFoodAssembly.mp4` }],
    ["PrewashFoodSorting", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/PrewashFoodSorting.mp4` }],
    ["QuickThaw", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/QuickThaw.mp4` }],
    ["RearrangeFridgeItems", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RearrangeFridgeItems.mp4` }],
    ["RecycleBottlesBySize", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RecycleBottlesBySize.mp4` }],
    ["RecycleBottlesByType", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RecycleBottlesByType.mp4` }],
    ["RecycleSodaCans", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RecycleSodaCans.mp4` }],
    ["RecycleStackedYogurt", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RecycleStackedYogurt.mp4` }],
    ["RefillCondimentStation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RefillCondimentStation.mp4` }],
    ["ReheatMeal", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ReheatMeal.mp4` }],
    ["ReheatMeatOnStove", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ReheatMeatOnStove.mp4` }],
    ["RemoveBroiledFish", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RemoveBroiledFish.mp4` }],
    ["RemoveCuttingBoardItems", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RemoveCuttingBoardItems.mp4` }],
    ["RemoveSteamedVegetables", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RemoveSteamedVegetables.mp4` }],
    ["ReorganizeFrozenVegetables", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ReorganizeFrozenVegetables.mp4` }],
    ["ResetCabinetDoors", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ResetCabinetDoors.mp4` }],
    ["RestockBowls", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RestockBowls.mp4` }],
    ["RestockCannedFood", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RestockCannedFood.mp4` }],
    ["RestockPantry", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RestockPantry.mp4` }],
    ["RestockSinkSupplies", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RestockSinkSupplies.mp4` }],
    ["RetrieveIceTray", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RetrieveIceTray.mp4` }],
    ["RetrieveMeat", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RetrieveMeat.mp4` }],
    ["ReturnHeatedFood", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ReturnHeatedFood.mp4` }],
    ["ReturnWashingSupplies", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ReturnWashingSupplies.mp4` }],
    ["RinseBowls", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RinseBowls.mp4` }],
    ["RinseCuttingBoard", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RinseCuttingBoard.mp4` }],
    ["RinseFragileItem", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RinseFragileItem.mp4` }],
    ["RinseSinkBasin", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RinseSinkBasin.mp4` }],
    ["RotatePan", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/RotatePan.mp4` }],
    ["SanitizePrepCuttingBoard", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SanitizePrepCuttingBoard.mp4` }],
    ["SanitizeSink", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SanitizeSink.mp4` }],
    ["ScalePortioning", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ScalePortioning.mp4` }],
    ["ScrubBowl", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ScrubBowl.mp4` }],
    ["ScrubCuttingBoard", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ScrubCuttingBoard.mp4` }],
    ["SearingMeat", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SearingMeat.mp4` }],
    ["SeasoningSpiceSetup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SeasoningSpiceSetup.mp4` }],
    ["SeasoningSteak", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SeasoningSteak.mp4` }],
    ["SeparateFreezerRack", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SeparateFreezerRack.mp4` }],
    ["SeparateRawIngredients", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SeparateRawIngredients.mp4` }],
    ["ServeMealJuice", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ServeMealJuice.mp4` }],
    ["ServeSteak", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ServeSteak.mp4` }],
    ["ServeTea", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ServeTea.mp4` }],
    ["ServeWarmCroissant", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ServeWarmCroissant.mp4` }],
    ["SetBowlsForSoup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SetBowlsForSoup.mp4` }],
    ["SetupBowls", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SetupBowls.mp4` }],
    ["SetupButterPlate", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SetupButterPlate.mp4` }],
    ["SetUpCuttingStation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SetUpCuttingStation.mp4` }],
    ["SetupFruitBowl", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SetupFruitBowl.mp4` }],
    ["SetupFrying", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SetupFrying.mp4` }],
    ["SetupSodaBowl", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SetupSodaBowl.mp4` }],
    ["SetUpSpiceStation", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SetUpSpiceStation.mp4` }],
    ["SetupWineGlasses", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SetupWineGlasses.mp4` }],
    ["ShakePan", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ShakePan.mp4` }],
    ["SimmeringSauce", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SimmeringSauce.mp4` }],
    ["SizeSorting", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SizeSorting.mp4` }],
    ["SnackSorting", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SnackSorting.mp4` }],
    ["SoakSponge", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SoakSponge.mp4` }],
    ["SortBreakfastIngredients", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SortBreakfastIngredients.mp4` }],
    ["SortingCleanup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SortingCleanup.mp4` }],
    ["SpicyMarinade", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SpicyMarinade.mp4` }],
    ["StackBowlsCabinet", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StackBowlsCabinet.mp4` }],
    ["StackBowlsInSink", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StackBowlsInSink.mp4` }],
    ["StackCans", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StackCans.mp4` }],
    ["StartElectricKettle", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StartElectricKettle.mp4` }],
    ["SteamFish", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SteamFish.mp4` }],
    ["SteamInMicrowave", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SteamInMicrowave.mp4` }],
    ["SteamVeggiesWithWater", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SteamVeggiesWithWater.mp4` }],
    ["StirVegetables", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StirVegetables.mp4` }],
    ["StockingBreakfastFoods", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StockingBreakfastFoods.mp4` }],
    ["StopSlowCooking", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StopSlowCooking.mp4` }],
    ["StoreDumplings", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StoreDumplings.mp4` }],
    ["StoreLeftoversByType", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StoreLeftoversByType.mp4` }],
    ["StoreLeftoversInBowl", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StoreLeftoversInBowl.mp4` }],
    ["StrainerSetup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/StrainerSetup.mp4` }],
    ["SweetenCoffee", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SweetenCoffee.mp4` }],
    ["SweetenHotChocolate", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SweetenHotChocolate.mp4` }],
    ["SweetSavoryToastSetup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/SweetSavoryToastSetup.mp4` }],
    ["ThawInSink", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ThawInSink.mp4` }],
    ["TiltPan", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/TiltPan.mp4` }],
    ["ToastBagel", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ToastBagel.mp4` }],
    ["ToastBaguette", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ToastBaguette.mp4` }],
    ["ToasterOvenBroilFish", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ToasterOvenBroilFish.mp4` }],
    ["ToastHeatableIngredients", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ToastHeatableIngredients.mp4` }],
    ["ToastOnCorrectRack", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ToastOnCorrectRack.mp4` }],
    ["ToastOneSlotPair", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/ToastOneSlotPair.mp4` }],
    ["TongBuffetSetup", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/TongBuffetSetup.mp4` }],
    ["TransportCookware", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/TransportCookware.mp4` }],
    ["TurnOffSimmeredSauceHeat", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/TurnOffSimmeredSauceHeat.mp4` }],
    ["UtensilShuffle", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/UtensilShuffle.mp4` }],
    ["VeggieDipPrep", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/VeggieDipPrep.mp4` }],
    ["WaffleReheat", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/WaffleReheat.mp4` }],
    ["WarmCroissant", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/WarmCroissant.mp4` }],
    ["WashFish", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/WashFish.mp4` }],
    ["WashFruitColander", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/WashFruitColander.mp4` }],
    ["WashInSaucepan", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/WashInSaucepan.mp4` }],
    ["WashLettuce", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/WashLettuce.mp4` }],
    ["WeighIngredients", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/WeighIngredients.mp4` }],
    ["WipeTable", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/WipeTable.mp4` }],
    ["YogurtDelightPrep", { label: "Demo", src: () => `${getContentRoot()}_static/composite_task_videos/YogurtDelightPrep.mp4` }],
  ]); */

  function getVideoForTask(taskName) {
    // You can override by setting:
    //   window.ROBOCASA_VIDEO_BASE_URL = "https://your-host/videos"
    // which will be used as: `${base}/${TaskName}.mp4`
    if (!taskName) return null;
    // Task display names may include asterisks (e.g. "Seen/Unseen" markers). Strip these for file lookup.
    const normalizedTaskName = String(taskName).replace(/\*/g, "").trim();
    const safe = encodeURIComponent(normalizedTaskName);
    const sources = [];

    // Default public hosting (Cloudflare R2), overrideable at runtime via:
    //   window.ROBOCASA_VIDEO_BASE_URL = "https://your-host/videos"
    // Used as: `${base}/${TaskName}.mp4`
    //
    // Public R2 bucket. Some uploads live at bucket root, some under a prefix.
    // We try both layouts so videos play regardless of how the objects were keyed.
    const DEFAULT_PUBLIC_VIDEO_BASE_URL = "https://pub-4433dcd10060475196ea5832312785f9.r2.dev";
    const DEFAULT_PUBLIC_VIDEO_PREFIX = "robocasa365-videos";

    // Optional user-configured base URL (e.g., R2 bucket)
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

    let video = document.createElement("video");
    video.className = "rc-video-modal-player";
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";
    // NOTE: Do NOT set video.crossOrigin by default.
    // The public R2 host may not emit CORS headers; forcing CORS can make playback fail in all browsers.

    const error = document.createElement("div");
    error.className = "rc-video-modal-error";
    error.hidden = true;

    const instruction = document.createElement("div");
    instruction.className = "rc-video-modal-instruction";
    instruction.hidden = true;
    const instructionLabel = document.createElement("strong");
    const instructionText = document.createElement("span");
    instructionText.className = "rc-video-modal-instruction-text";
    instruction.appendChild(instructionLabel);
    instruction.appendChild(instructionText);

    // Optional debug logging.
    // Enable in DevTools console:
    //   window.ROBOCASA_DEBUG_VIDEO_LOGS = true
    //   window.ROBOCASA_DEBUG_VIDEO_LOGS_TO_SERVER = true
    // When enabled, this will:
    // - console.debug(...) a detailed timeline, and
    // - "print" to a simple Python static server by requesting:
    //     {content_root}/__rc_video_log?...  (will typically 404, but still logs the request)
    function rcVideoDebugEnabled() {
      return !!window.ROBOCASA_DEBUG_VIDEO_LOGS || window.ROBOCASA_DEBUG_VIDEO_LOGS === "server";
    }
    function rcVideoDebugServerEnabled() {
      return !!window.ROBOCASA_DEBUG_VIDEO_LOGS_TO_SERVER || window.ROBOCASA_DEBUG_VIDEO_LOGS === "server";
    }
    function rcVideoDebugPing(params) {
      if (!rcVideoDebugServerEnabled()) return;
      try {
        const q = new URLSearchParams();
        for (const [k, v] of Object.entries(params || {})) {
          if (v == null) continue;
          q.set(k, String(v).slice(0, 500));
        }
        // Use content_root so this works for local hosting under subpaths / versioned docs.
        const url = `${getContentRoot()}__rc_video_log?${q.toString()}`;
        const img = new Image();
        img.src = url;
      } catch {}
    }
    function rcVideoDebugLog(event, params) {
      if (!rcVideoDebugEnabled()) return;
      const payload = { event, t: Date.now(), ...(params || {}) };
      try {
        // eslint-disable-next-line no-console
        console.debug("[robocasa video]", payload);
      } catch {}
      rcVideoDebugPing(payload);
    }

    function rcVideoSnapshot() {
      try {
        const snap = {
          readyState: video.readyState,
          networkState: video.networkState,
          currentSrc: video.currentSrc || "",
          currentTime: Number.isFinite(video.currentTime) ? Number(video.currentTime.toFixed(3)) : "",
          duration: Number.isFinite(video.duration) ? Number(video.duration.toFixed(3)) : "",
          errorCode: video.error ? video.error.code : "",
        };
        try {
          const b = video.buffered;
          if (b && typeof b.length === "number") {
            // Keep this small: only the last buffered range.
            if (b.length > 0) {
              snap.buffered = `${b.start(b.length - 1).toFixed(3)}-${b.end(b.length - 1).toFixed(3)}`;
            } else {
              snap.buffered = "0";
            }
          }
        } catch {}
        return snap;
      } catch {
        return {};
      }
    }

    function rcAttachVideoDebugEvents(session, taskName) {
      if (!rcVideoDebugEnabled()) return () => {};
      const events = [
        "loadstart",
        "loadedmetadata",
        "loadeddata",
        "canplay",
        "canplaythrough",
        "progress",
        "stalled",
        "suspend",
        "waiting",
        "playing",
        "pause",
        "abort",
        "emptied",
        "error",
      ];

      const handlers = new Map();
      for (const ev of events) {
        const handler = () => {
          if (overlay._rcVideoSession !== session) return;
          rcVideoDebugLog(`video_${ev}`, { task: taskName || "", ...rcVideoSnapshot() });
        };
        handlers.set(ev, handler);
        video.addEventListener(ev, handler);
      }

      return () => {
        for (const [ev, handler] of handlers.entries()) {
          video.removeEventListener(ev, handler);
        }
      };
    }

    // Monotonically increasing token; used to ignore stale events from prior opens.
    overlay._rcVideoSession = 0;

      function doClose() {
      overlay.hidden = true;
      document.body.classList.remove("rc-modal-open");
      try {
        video.pause();
      } catch {}
      // Swallow any pending play() rejection when we abort the load (avoids "Uncaught (in promise) DOMException").
      if (overlay._rcLastPlayPromise) {
        overlay._rcLastPlayPromise.catch(function () {});
        overlay._rcLastPlayPromise = null;
      }
      if (overlay._rcLoadTimeout != null) {
        clearTimeout(overlay._rcLoadTimeout);
        overlay._rcLoadTimeout = null;
      }
      instruction.hidden = true;
      instructionLabel.textContent = "";
      instructionText.textContent = "";
      error.hidden = true;
      error.textContent = "";
      titleCode.textContent = "";
      try {
        video.removeAttribute("src");
        video.load();
      } catch (_) {}
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
    modal.appendChild(error);
    modal.appendChild(instruction);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    function replaceVideoElement() {
      // Firefox in particular can get "stuck" after many sequential MP4 loads.
      // Replacing the <video> element forces a full reset of the media pipeline.
      const next = document.createElement("video");
      next.className = "rc-video-modal-player";
      next.controls = true;
      next.playsInline = true;
      next.preload = "metadata";

      try {
        video.pause();
      } catch {}
      if (overlay._rcLastPlayPromise) {
        overlay._rcLastPlayPromise.catch(function () {});
        overlay._rcLastPlayPromise = null;
      }
      try {
        video.removeAttribute("src");
        video.load();
      } catch {}

      modal.replaceChild(next, video);
      video = next;
    }

    overlay._rcOpen = (sources, taskName, taskDescription) => {
      overlay._rcVideoSession = (overlay._rcVideoSession || 0) + 1;
      const session = overlay._rcVideoSession;
      // Some cross-origin video requests can get stuck in an edge/cache hiccup state.
      // If that happens, we retry the *same* URL once with a cache-busting query param.
      let didCacheBustRetry = false;
      if (overlay._rcLoadTimeout != null) {
        clearTimeout(overlay._rcLoadTimeout);
        overlay._rcLoadTimeout = null;
      }
      // Hard reset media element (prevents Firefox "stuck" loads).
      replaceVideoElement();
      if (typeof overlay._rcVideoDebugCleanup === "function") {
        try {
          overlay._rcVideoDebugCleanup();
        } catch {}
      }
      // Attach debug listeners to the *current* video element.
      overlay._rcVideoDebugCleanup = rcAttachVideoDebugEvents(session, taskName);

      rcVideoDebugLog("open", {
        task: taskName || "",
        sources: Array.isArray(sources) ? sources.join(" | ") : String(sources || ""),
      });

      overlay.hidden = false;
      document.body.classList.add("rc-modal-open");
      error.hidden = true;
      error.textContent = "";
      titleCode.textContent = taskName ? `${taskName}` : "";

      const srcsRaw = Array.isArray(sources) ? sources : [sources];
      const srcs = srcsRaw.map((s) => String(s || "").trim()).filter(Boolean);
      let attempt = 0;

      function showUnavailable() {
        video.removeAttribute("src");
        video.load();

        rcVideoDebugLog("unavailable", {
          task: taskName || "",
          attempt,
          readyState: video.readyState,
          networkState: video.networkState,
          currentSrc: video.currentSrc || "",
          errorCode: video.error ? video.error.code : "",
        });

        error.hidden = false;
        const hasCustomBase = typeof window.ROBOCASA_VIDEO_BASE_URL === "string" && window.ROBOCASA_VIDEO_BASE_URL.trim();
        if (hasCustomBase) {
          error.textContent =
            "Demo video unavailable. The video could not be loaded from the configured base URL. Please verify the URL is correct and points to where the .mp4 files are stored (some setups use a /robocasa365-videos prefix).";
        } else {
          error.textContent =
            "Demo video unavailable. Could not load the demo from the public R2 host. If your videos are hosted elsewhere, set " +
            'window.ROBOCASA_VIDEO_BASE_URL to a base like "https://pub-4433dcd10060475196ea5832312785f9.r2.dev" (or ".../robocasa365-videos" if your objects use that prefix).';
        }
      }

      let loadTimeout = null;
      let hasLoadedMetadata = false;

      function tryNextSource() {
        if (attempt >= srcs.length) {
          showUnavailable();
          return;
        }
        const srcIndex = attempt;
        const src = srcs[attempt++];
        hasLoadedMetadata = false;
        overlay._rcActiveVideoSrc = src;

        rcVideoDebugLog("try_source", {
          task: taskName || "",
          srcIndex,
          src,
          ...rcVideoSnapshot(),
        });
        
        // Clear any existing timeout (and stale timeouts from a previous open)
        if (overlay._rcLoadTimeout != null) {
          clearTimeout(overlay._rcLoadTimeout);
          overlay._rcLoadTimeout = null;
        }
        if (loadTimeout) {
          clearTimeout(loadTimeout);
          loadTimeout = null;
        }

        // Safari: use canplay or loadedmetadata to detect successful load
        const onCanPlay = () => {
          if (overlay._rcVideoSession !== session) return;
          hasLoadedMetadata = true;
          if (overlay._rcLoadTimeout != null) {
            clearTimeout(overlay._rcLoadTimeout);
            overlay._rcLoadTimeout = null;
          }
          if (loadTimeout) {
            clearTimeout(loadTimeout);
            loadTimeout = null;
          }
          video.removeEventListener("canplay", onCanPlay);
          video.removeEventListener("loadedmetadata", onCanPlay);

          rcVideoDebugLog("loaded_metadata", {
            task: taskName || "",
            readyState: video.readyState,
            networkState: video.networkState,
            currentSrc: video.currentSrc || "",
          });
        };
        video.addEventListener("canplay", onCanPlay);
        video.addEventListener("loadedmetadata", onCanPlay);

        // Set and attempt playback (may require user gesture; that's fine)
        video.src = src;
        video.load();
        rcVideoDebugLog("set_src", { task: taskName || "", src, ...rcVideoSnapshot() });
        
        // Safari: give it more time before assuming error (especially for R2/CDN)
        // Increased timeout for Safari which can be slower with cross-origin video loading
        // 5s is too aggressive for cross-origin video metadata on some networks / browsers.
        // A longer timeout avoids false "unavailable" errors when R2 is slow to respond.
        loadTimeout = window.setTimeout(() => {
          if (overlay._rcVideoSession !== session) return;
          if (!hasLoadedMetadata) {
            rcVideoDebugLog("timeout", {
              task: taskName || "",
              src,
              ...rcVideoSnapshot(),
            });

            // If the primary URL is valid but stuck, retry once with cache-busting.
            // This can recover from transient Cloudflare edge/cache issues.
            if (!didCacheBustRetry && srcIndex === 0 && !String(src).includes("cb=")) {
              didCacheBustRetry = true;
              const cbSrc = `${src}${String(src).includes("?") ? "&" : "?"}cb=${Date.now()}`;
              overlay._rcActiveVideoSrc = cbSrc;
              rcVideoDebugLog("retry_cache_bust", { task: taskName || "", from: src, to: cbSrc, ...rcVideoSnapshot() });

              // Reset listeners + src and try again.
              video.removeEventListener("canplay", onCanPlay);
              video.removeEventListener("loadedmetadata", onCanPlay);
              try {
                video.pause();
              } catch {}
              video.removeAttribute("src");
              video.load();
              video.src = cbSrc;
              video.load();
              rcVideoDebugLog("set_src", { task: taskName || "", src: cbSrc, ...rcVideoSnapshot() });

              // Arm a fresh timeout for the retry.
              if (overlay._rcLoadTimeout != null) {
                clearTimeout(overlay._rcLoadTimeout);
                overlay._rcLoadTimeout = null;
              }
              loadTimeout = window.setTimeout(() => {
                if (overlay._rcVideoSession !== session) return;
                if (!hasLoadedMetadata) {
                  rcVideoDebugLog("timeout_retry", { task: taskName || "", src: cbSrc, ...rcVideoSnapshot() });
                }
              }, 20000);
              overlay._rcLoadTimeout = loadTimeout;

              const p2 = video.play();
              overlay._rcLastPlayPromise = p2;
              if (p2 && typeof p2.catch === "function") p2.catch(() => {});
              return;
            }
            // Do NOT fail over on timeout. For R2, the "prefix" layout is often a 404,
            // so timing out and switching sources can create a false "unavailable"
            // even when the primary URL would have loaded if given more time.
            //
            // We only try the next source on a real `error` event.
          }
        }, 20000);
        overlay._rcLoadTimeout = loadTimeout;

        const p = video.play();
        overlay._rcLastPlayPromise = p;
        if (p && typeof p.catch === "function") p.catch(() => {});
      }

      // If a source fails to load, try the next one.
      // Safari: only treat as error if we haven't loaded metadata yet
      video.onerror = () => {
        if (overlay._rcVideoSession !== session) return;
        rcVideoDebugLog("error", {
          task: taskName || "",
          readyState: video.readyState,
          networkState: video.networkState,
          currentSrc: video.currentSrc || "",
          errorCode: video.error ? video.error.code : "",
        });
        if (overlay._rcLoadTimeout != null) {
          clearTimeout(overlay._rcLoadTimeout);
          overlay._rcLoadTimeout = null;
        }
        if (loadTimeout) {
          clearTimeout(loadTimeout);
          loadTimeout = null;
        }
        // Only fail over if the error is for the currently active src.
        if (!hasLoadedMetadata && (!overlay._rcActiveVideoSrc || video.currentSrc === overlay._rcActiveVideoSrc)) {
          tryNextSource();
        }
      };

      const tName = (taskName || "").trim();
      const tDesc = (taskDescription || "").trim();
      if (tName || tDesc) {
        // Title already shows task name; bottom should only show the description.
        instructionLabel.textContent = "";
        let htmlDesc = tDesc || "";
        // Sphinx renders [*text*] as [<em>text</em>] - keep brackets but ensure italics
        htmlDesc = htmlDesc.replace(/\[<em>([^<]+)<\/em>\]/g, "[<em>$1</em>]");
        // Also handle plain markdown [*text*] if not yet rendered - convert to [<em>text</em>]
        htmlDesc = htmlDesc.replace(/\[\*([^\*]+)\*\]/g, "[<em>$1</em>]");
        instructionText.innerHTML = htmlDesc;
        instruction.hidden = false;
      } else {
        instruction.hidden = true;
        instructionLabel.textContent = "";
        instructionText.textContent = "";
      }

      tryNextSource();
    };
    overlay._rcClose = doClose;

    return overlay;
  }

  function openVideo(src, taskName, taskDescription) {
    const overlay = ensureVideoModal();
    overlay._rcOpen(src, taskName, taskDescription);
  }

  function scrollToTask(item) {
    if (!item) return;
    if (item.detailsEl && item.detailsEl.tagName.toLowerCase() === "details") item.detailsEl.open = true;
    // Scroll activity into view first (so layout expands)
    const anchor = document.getElementById(item.activityId) || item.detailsEl;
    if (anchor) anchor.scrollIntoView({ behavior: "smooth", block: "start" });
    // Then scroll row into view and highlight after scroll completes
    window.setTimeout(() => {
      item.rowEl.scrollIntoView({ behavior: "smooth", block: "center" });
      // Wait for smooth scroll to complete before highlighting
      // Smooth scrolls typically take 300-1000ms depending on distance
      // Use IntersectionObserver to detect when element is actually visible
      let highlighted = false;
      const doHighlight = () => {
        if (highlighted) return;
        highlighted = true;
        highlightRow(item.rowEl);
      };
      const observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting && entry.intersectionRatio > 0.5) {
              observer.disconnect();
              // Small delay to ensure element is fully in view
              window.setTimeout(doHighlight, 100);
              return;
            }
          }
        },
        { threshold: 0.5 }
      );
      observer.observe(item.rowEl);
      // Fallback: if observer doesn't fire within 2s, highlight anyway
      window.setTimeout(() => {
        observer.disconnect();
        doHighlight();
      }, 2000);
    }, 350);
  }

  function addAttributesToTable(table, attrsMap, episodeLengthMap) {
    ensureColumns(table);
    const rows = Array.from(table.querySelectorAll("tbody tr"));
    for (const tr of rows) {
      // Avoid double-inserting
      if (
        tr.querySelector("td.rc-task-skills") ||
        tr.querySelector("td.rc-subtasks") ||
        tr.querySelector("td.rc-episode-length")
      )
        continue;

      const name = taskNameFromRow(tr);
      const attrs = name ? attrsMap.get(name) : null;

      // Base columns at this point: Task (0), Description (1)
      const tds0 = Array.from(tr.querySelectorAll("td"));
      const taskTd = tds0[0] || null;
      const descTd0 = tds0[1] || null;
      const descText = (descTd0?.textContent || "").trim();

      // Remove any legacy tag container under task name
      taskTd?.querySelector(".rc-task-tags")?.remove();

      // Insert Task Skills after Description
      const skillsTd = document.createElement("td");
      skillsTd.className = "rc-task-skills";
      skillsTd.style.textAlign = "left";
      if (descTd0) tr.insertBefore(skillsTd, descTd0.nextSibling);
      else tr.appendChild(skillsTd);

      if (name) renderTaskTagsIntoCell(name, skillsTd, attrs?.mobile, descText);
      if (name && taskTd) renderCompositeTargetTagIntoTaskCell(name, taskTd);

      const mobileVal = attrs?.mobile;
      tr.dataset.rcMobile = mobileVal === "Yes" || mobileVal === "No" ? mobileVal : "";
      // If an old Navigation column exists from a cached script run, remove it.
      tr.querySelector("td.rc-mobile")?.remove();

      const subtasksTd = document.createElement("td");
      subtasksTd.className = "rc-subtasks";
      subtasksTd.style.textAlign = "left";
      const n = attrs?.subtasks;
      subtasksTd.textContent = Number.isFinite(n) ? String(n) : "—";

      const episodeLengthTd = document.createElement("td");
      episodeLengthTd.className = "rc-episode-length";
      episodeLengthTd.style.textAlign = "left";
      const episodeLength = name ? episodeLengthMap?.get(name) : null;
      if (episodeLength != null && Number.isFinite(episodeLength)) {
        // Format as seconds rounded to nearest integer
        episodeLengthTd.textContent = `${Math.round(episodeLength)}s`;
      } else {
        episodeLengthTd.textContent = "—";
      }

      const videoTd = document.createElement("td");
      videoTd.className = "rc-video";
      videoTd.style.textAlign = "left";
      const v = name ? getVideoForTask(name) : null;
      if (v) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "rc-atomic-video-btn";
        btn.textContent = "Demo";
        btn.addEventListener("click", () => {
          // Get description as HTML (Sphinx may have already rendered markdown) or as text
          const desc = descTd0
            ? (descTd0.innerHTML || descTd0.textContent || "").trim()
            : "";
          openVideo(v.sources, name || "", desc);
        });
        videoTd.appendChild(btn);
      } else {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "rc-atomic-video-btn";
        btn.textContent = "Demo";
        btn.disabled = true;
        videoTd.appendChild(btn);
      }

      // Insert after Task Skills column (Task, Description, Task Skills)
      const tds = tr.querySelectorAll("td");
      const skillsTd2 = tds.length >= 3 ? tds[2] : null;
      if (skillsTd2 && skillsTd2.parentNode) {
        tr.insertBefore(subtasksTd, skillsTd2.nextSibling);
        tr.insertBefore(episodeLengthTd, subtasksTd.nextSibling);
      } else {
        tr.appendChild(subtasksTd);
        tr.appendChild(episodeLengthTd);
      }

      // Always append Video at the far right
      tr.appendChild(videoTd);
    }
  }

  function addAtomicAttributesToTable(table, atomicEpisodeLengthMap) {
    // Reuse Composite column layout, but fill Atomic-specific horizon + video.
    ensureColumns(table);
    const rows = Array.from(table.querySelectorAll("tbody tr"));
    for (const tr of rows) {
      // Avoid double-inserting
      if (
        tr.querySelector("td.rc-task-skills") ||
        tr.querySelector("td.rc-subtasks") ||
        tr.querySelector("td.rc-episode-length") ||
        tr.querySelector("td.rc-video")
      )
        continue;

      const name = taskNameFromRow(tr);
      const tds0 = Array.from(tr.querySelectorAll("td"));
      const descTd0 = tds0[1] || null;

      // Insert Task Skills after Description (blank for Atomic)
      const skillsTd = document.createElement("td");
      skillsTd.className = "rc-task-skills";
      skillsTd.style.textAlign = "left";
      skillsTd.textContent = "—";
      if (descTd0) tr.insertBefore(skillsTd, descTd0.nextSibling);
      else tr.appendChild(skillsTd);

      const subtasksTd = document.createElement("td");
      subtasksTd.className = "rc-subtasks";
      subtasksTd.style.textAlign = "left";
      subtasksTd.textContent = "—";

      const episodeLengthTd = document.createElement("td");
      episodeLengthTd.className = "rc-episode-length";
      episodeLengthTd.style.textAlign = "left";
      const episodeLength = name ? atomicEpisodeLengthMap?.get(name) : null;
      if (episodeLength != null && Number.isFinite(episodeLength)) {
        episodeLengthTd.textContent = `${Math.round(episodeLength)}s`;
      } else {
        episodeLengthTd.textContent = "—";
      }

      const videoTd = document.createElement("td");
      videoTd.className = "rc-video";
      videoTd.style.textAlign = "left";
      const v = name ? getVideoForTask(name) : null;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "rc-atomic-video-btn";
      btn.textContent = "Demo";
      if (v) {
        btn.addEventListener("click", () => {
          const desc = descTd0 ? (descTd0.innerHTML || descTd0.textContent || "").trim() : "";
          openVideo(v.sources, name || "", desc);
        });
      } else {
        btn.disabled = true;
      }
      videoTd.appendChild(btn);

      // Insert after Task Skills column (Task, Description, Task Skills)
      const tds = tr.querySelectorAll("td");
      const skillsTd2 = tds.length >= 3 ? tds[2] : null;
      if (skillsTd2 && skillsTd2.parentNode) {
        tr.insertBefore(subtasksTd, skillsTd2.nextSibling);
        tr.insertBefore(episodeLengthTd, subtasksTd.nextSibling);
      } else {
        tr.appendChild(subtasksTd);
        tr.appendChild(episodeLengthTd);
      }
      tr.appendChild(videoTd);
    }
  }

  function ensureBackToTopButton() {
    if (!(isCompositeTasksPage() || isObjectsPage())) return;
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

  onReady(async () => {
    if (!isCompositeTasksPage()) return;
    // Hard guard: prevent double-initialization (duplicate activity cards)
    if (window.__rcCompositeTasksInit) return;
    window.__rcCompositeTasksInit = true;

    // Page-scoped styling hooks
    document.body.classList.add("rc-composite-tasks");

    // Pack "Skills Involved" tags when their <details> becomes visible.
    // (When <details> is closed, we can't measure wrap layout, so initial packing is deferred.)
    function packDeferredSkillTags(scopeEl) {
      const scope = scopeEl || document;
      const wraps = Array.from(scope.querySelectorAll('.rc-task-tags[data-rc-needs-skill-pack="1"]'));
      for (const w of wraps) {
        if (!rcCanMeasureSkillPack(w)) continue;
        try {
          delete w.dataset.rcNeedsSkillPack;
        } catch {}
        rcPackSkillTagsInWrap(w);
      }
    }
    document.addEventListener(
      "toggle",
      (e) => {
        const d = e && e.target;
        if (!d || d.tagName !== "DETAILS" || !d.classList.contains("rc-activity") || !d.open) return;
        // Lazily enrich the activity table (adds columns/tags) when opened.
        ensureDetailsTableEnriched(d);
        window.requestAnimationFrame(() => packDeferredSkillTags(d));
      },
      true
    );
    let packResizeTimer = null;
    window.addEventListener("resize", () => {
      if (packResizeTimer) window.clearTimeout(packResizeTimer);
      packResizeTimer = window.setTimeout(() => packDeferredSkillTags(document), 120);
    });

    // Sphinx Book Theme main content wrapper
    const content =
      document.querySelector("main .bd-article") ||
      document.querySelector("main") ||
      document.body;
    if (!content) return;

    // Activities are sections nested under the main "Composite Tasks" section,
    // OR are generated from JSON if a placeholder root is present.
    const rootSection = content.querySelector("section#composite-tasks");
    if (!rootSection) return;

    const attrsMap = getTaskAttributesMap();
    const episodeLengthMap = getEpisodeLengthMap();
    const taskToSourceMap = buildTaskToSourceMapFromLegacy(rootSection);

    const activities = [];

    const placeholderRoot = rootSection.querySelector("#rc-composite-tasks-root");
    const existingDetails = Array.from(rootSection.querySelectorAll(":scope > details.rc-activity"));

    if (existingDetails.length > 0) {
      // Build activities list from existing <details> (pre-rendered at build time).
      for (const details of existingDetails) {
        const title = getActivityTitleFromDetails(details);
        if (!details.id) {
          const anchorId = anchorIdFromActivityTitle(title);
          if (anchorId) details.id = anchorId;
        }

        // Ensure meta pill exists for consistent UI
        const summaryLeft = details.querySelector("summary .rc-activity-summary-left");
        const hasMeta = !!details.querySelector("summary .rc-activity-meta");
        const metaCategory = metaCategoryForActivityTitle(title);
        details.dataset.metaCategory = metaCategory;
        if (summaryLeft && !hasMeta) {
          const metaTag = document.createElement("span");
          metaTag.className = "rc-activity-meta";
          metaTag.dataset.meta = metaCategory;
          metaTag.textContent = categoryLabel(metaCategory);
          summaryLeft.appendChild(metaTag);
        }

        activities.push({ title, id: details.id });
      }
    } else if (placeholderRoot) {
      try {
        const data = await loadTaskAttributesJson();
        const grouped = groupTasksByActivity(data && data.tasks ? data.tasks : []);
        const activityEntries = Array.from(grouped.values()).sort((a, b) => (a.title || "").localeCompare(b.title || ""));

        const detailsEls = [];
        for (const entry of activityEntries) {
          const title = entry.title || "";
          const folder = entry.folder || folderForActivityTitle(title);
          const tasks = entry.tasks || [];

          const details = document.createElement("details");
          details.className = "rc-activity";

          const summary = document.createElement("summary");
          const summaryLeft = document.createElement("div");
          summaryLeft.className = "rc-activity-summary-left";

          const summaryText = document.createElement("span");
          summaryText.className = "rc-activity-title";
          summaryText.textContent = title;
          summaryLeft.appendChild(summaryText);

          const metaCategory = metaCategoryForActivityTitle(title);
          details.dataset.metaCategory = metaCategory;
          const metaTag = document.createElement("span");
          metaTag.className = "rc-activity-meta";
          metaTag.dataset.meta = metaCategory;
          metaTag.textContent = categoryLabel(metaCategory);
          summaryLeft.appendChild(metaTag);

          summary.appendChild(summaryLeft);

          const nTasks = tasks.length;
          const badge = document.createElement("span");
          badge.className = "rc-activity-badge";
          badge.textContent = `${nTasks} task${nTasks === 1 ? "" : "s"}`;
          summary.appendChild(badge);

          details.appendChild(summary);

          const body = document.createElement("div");
          body.className = "rc-activity-body";

          const table = document.createElement("table");
          table.className = "docutils";
          table.setAttribute("border", "1");
          const thead = document.createElement("thead");
          const theadTr = document.createElement("tr");
          const thTask = document.createElement("th");
          thTask.textContent = "Task";
          const thDesc = document.createElement("th");
          thDesc.textContent = "Description";
          thTask.style.textAlign = "left";
          thDesc.style.textAlign = "left";
          theadTr.appendChild(thTask);
          theadTr.appendChild(thDesc);
          thead.appendChild(theadTr);
          table.appendChild(thead);

          const tbody = document.createElement("tbody");
          for (const t of tasks) {
            const tr = document.createElement("tr");
            const tdTask = document.createElement("td");
            const code = document.createElement("code");
            code.textContent = t.name;
            const srcUrl = taskToSourceMap.get(t.name) || sourceUrlForTask(folder || title, t.name);
            if (srcUrl) {
              const a = document.createElement("a");
              a.href = srcUrl;
              a.target = "_blank";
              a.rel = "noopener noreferrer";
              a.appendChild(code);
              tdTask.appendChild(a);
            } else {
              tdTask.appendChild(code);
            }
            const tdDesc = document.createElement("td");
            tdDesc.innerHTML = formatDescriptionToHtml(t.description || "");
            tdTask.style.textAlign = "left";
            tdDesc.style.textAlign = "left";
            tr.appendChild(tdTask);
            tr.appendChild(tdDesc);
            tbody.appendChild(tr);
          }
          table.appendChild(tbody);

          body.appendChild(table);

          const anchorId = anchorIdFromActivityTitle(title);
          details.id = anchorId || title;
          details.appendChild(body);

          activities.push({ title, id: details.id });
          detailsEls.push(details);
        }

        // Replace the placeholder root with the generated activities, so <details> are direct children
        placeholderRoot.replaceWith(...detailsEls);
      } catch (e) {
        // If JSON can't be loaded, fall back to existing static HTML (if any).
        console.error(e);
      }
    } else {
      const activitySections = Array.from(rootSection.querySelectorAll(":scope > section[id]"));
      if (activitySections.length === 0) return;

      for (const sec of activitySections) {
        const h2 = sec.querySelector(":scope > h2");
        if (!h2) continue;

        const title = getHeadingTitle(h2);
        if (!title) continue;

        const details = document.createElement("details");
        details.className = "rc-activity";

        const summary = document.createElement("summary");
        const summaryLeft = document.createElement("div");
        summaryLeft.className = "rc-activity-summary-left";

        const summaryText = document.createElement("span");
        summaryText.className = "rc-activity-title";
        summaryText.textContent = title;
        summaryLeft.appendChild(summaryText);

        const metaCategory = metaCategoryForActivityTitle(title);
        details.dataset.metaCategory = metaCategory;
        const metaTag = document.createElement("span");
        metaTag.className = "rc-activity-meta";
        metaTag.dataset.meta = metaCategory;
        metaTag.textContent = categoryLabel(metaCategory);
        summaryLeft.appendChild(metaTag);

        summary.appendChild(summaryLeft);

        const nTasks = countTasks(sec);
        if (typeof nTasks === "number") {
          const badge = document.createElement("span");
          badge.className = "rc-activity-badge";
          badge.textContent = `${nTasks} task${nTasks === 1 ? "" : "s"}`;
          summary.appendChild(badge);
        }
        details.appendChild(summary);

        const body = document.createElement("div");
        body.className = "rc-activity-body";

        // Move everything except the heading into the details body
        for (const child of Array.from(sec.childNodes)) {
          if (child === h2) continue;
          body.appendChild(child);
        }

        // Preserve existing anchor IDs (so #baking links still work)
        const anchorId = sec.id;
        details.id = anchorId;
        details.appendChild(body);

        activities.push({ title, id: anchorId });

        // Replace the entire section with details
        sec.parentNode.replaceChild(details, sec);
      }
    }

    // Fill the intro blurb activity count (if present)
    const activityCountEl = content.querySelector("#rc-composite-activity-count");
    if (activityCountEl) activityCountEl.textContent = String(activities.length);

    // Convert "Class File" column into a link on the Task name, then remove the column
    for (const d of Array.from(content.querySelectorAll("details.rc-activity"))) {
      const table = d.querySelector("table");
      if (table) linkifyTaskAndRemoveClassFile(table);
    }

    // Replace templated variable braces in descriptions:
    // "{x}" and [*x*] -> [<strong><em>x</em></strong>]; *x* or **x** -> <strong><em>x</em></strong>
    function formatCompositeDescriptionInPlace(rootEl) {
      if (!rootEl) return;
      const textNodes = [];
      const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) textNodes.push(n);

      const re = /\{([^}]+)\}|\[\*([^*]*?)\*\]|\*\*([^*]+)\*\*|\*([^*]+)\*/g;
      for (const node of textNodes) {
        const s = node.nodeValue || "";
        if (!s.includes("{") && !s.includes("*")) continue;
        re.lastIndex = 0;

        const frag = document.createDocumentFragment();
        let last = 0;
        let m;
        while ((m = re.exec(s)) !== null) {
          if (m.index > last) frag.appendChild(document.createTextNode(s.slice(last, m.index)));
          const text = m[1] !== undefined ? m[1] : (m[2] !== undefined ? m[2] : (m[3] !== undefined ? m[3] : m[4]));
          const withBrackets = m[1] !== undefined || m[2] !== undefined;
          if (withBrackets) frag.appendChild(document.createTextNode("["));
          const strong = document.createElement("strong");
          const em = document.createElement("em");
          em.textContent = text;
          strong.appendChild(em);
          frag.appendChild(strong);
          if (withBrackets) frag.appendChild(document.createTextNode("]"));
          last = re.lastIndex;
        }
        if (last === 0) continue; // no matches
        if (last < s.length) frag.appendChild(document.createTextNode(s.slice(last)));
        node.parentNode.replaceChild(frag, node);
      }
    }

    // Lazily enrich a table the first time its activity opens.
    function ensureDetailsTableEnriched(detailsEl) {
      if (!detailsEl || detailsEl.dataset.rcEnriched === "1") return;
      const table = detailsEl.querySelector("table");
      if (!table) return;

      // Apply description formatting (braces/asterisks) once
      const ths = Array.from(table.querySelectorAll("thead tr th")).map((th) => (th.textContent || "").trim());
      const descIdx = ths.indexOf("Description");
      const idx = descIdx >= 0 ? descIdx : 1;
      for (const tr of Array.from(table.querySelectorAll("tbody tr"))) {
        const td = tr.children?.[idx];
        if (td) formatCompositeDescriptionInPlace(td);
      }

      // Add computed attribute columns once (Skills/Subtasks/Horizon/Video)
      if (attrsMap || episodeLengthMap) {
        addAttributesToTable(table, attrsMap, episodeLengthMap);
      }

      detailsEl.dataset.rcEnriched = "1";
    }

    // Insert a selector at the top
    const selectorWrap = document.createElement("div");
    selectorWrap.className = "rc-activity-selector";

    const activityGroup = document.createElement("div");
    activityGroup.className = "rc-activity-group";

    const label = document.createElement("label");
    label.textContent = "Activity:";
    label.setAttribute("for", "rc-activity-select");

    const select = document.createElement("select");
    select.id = "rc-activity-select";
    // Enable :invalid styling for placeholder text
    select.required = true;

    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.disabled = true;
    defaultOpt.selected = true;
    defaultOpt.textContent = "Select an activity…";
    select.appendChild(defaultOpt);

    for (const a of activities) {
      const opt = document.createElement("option");
      opt.value = a.id;
      opt.textContent = a.title;
      select.appendChild(opt);
    }

    select.addEventListener("change", () => {
      const id = select.value;
      if (!id) return;
      const el = document.getElementById(id);
      if (!el) return;
      if (el.tagName.toLowerCase() === "details") el.open = true;
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    activityGroup.appendChild(label);
    activityGroup.appendChild(select);
    selectorWrap.appendChild(activityGroup);

    // Filters
    const filtersWrap = document.createElement("div");
    filtersWrap.className = "rc-filters";

    // Sort UI (goes BELOW the filters row)
    const sortWrap = document.createElement("div");
    sortWrap.className = "rc-sort-row";
    const sortLabel = document.createElement("label");
    sortLabel.textContent = "Sort by:";
    sortLabel.setAttribute("for", "rc-sort-by");
    const sortSelect = document.createElement("select");
    sortSelect.id = "rc-sort-by";
    for (const [val, text] of [
      ["activity", "A-Z"],
      ["category", "Category"],
    ]) {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = text;
      sortSelect.appendChild(opt);
    }
    sortWrap.appendChild(sortLabel);
    sortWrap.appendChild(sortSelect);

    const taskCount = document.createElement("div");
    taskCount.className = "rc-task-count";
    taskCount.textContent = "Showing 0 tasks";
    sortWrap.appendChild(taskCount);

    const subWrap = document.createElement("div");
    subWrap.className = "rc-filter rc-filter-subtasks";
    const subRow = document.createElement("div");
    subRow.className = "rc-subtasks-row";

    const subLabel = document.createElement("label");
    subLabel.textContent = "Subtasks:";
    subLabel.setAttribute("for", "rc-filter-sub-min");
    const minInput = document.createElement("input");
    minInput.id = "rc-filter-sub-min";
    // Use number input so we get the up/down steppers.
    // We still allow free typing (including clearing); we validate ourselves.
    minInput.type = "number";
    minInput.min = "2";
    minInput.max = "16";
    minInput.step = "1";
    // Use placeholders so defaults look like "suggested" values
    minInput.placeholder = "2";
    minInput.value = "";
    minInput.ariaLabel = "Minimum subtasks";

    const maxInput = document.createElement("input");
    maxInput.id = "rc-filter-sub-max";
    maxInput.type = "number";
    maxInput.min = "2";
    maxInput.max = "16";
    maxInput.step = "1";
    maxInput.placeholder = "16";
    maxInput.value = "";
    maxInput.ariaLabel = "Maximum subtasks";

    const dash = document.createElement("span");
    dash.className = "rc-filter-range-sep";
    dash.textContent = "–";

    subRow.appendChild(subLabel);
    subRow.appendChild(minInput);
    subRow.appendChild(dash);
    subRow.appendChild(maxInput);
    subWrap.appendChild(subRow);

    filtersWrap.appendChild(subWrap);

    const note = document.createElement("div");
    note.className = "rc-filter-note rc-subtasks-note";
    note.textContent = "Subtasks must be between 2 and 16";
    note.hidden = true;
    subWrap.appendChild(note);

    // Length filter (multi-select dropdown with counts)
    const lengthWrap = document.createElement("div");
    lengthWrap.className = "rc-filter rc-filter-length";

    const lengthBtn = document.createElement("button");
    lengthBtn.type = "button";
    lengthBtn.id = "rc-length-btn";
    lengthBtn.className = "rc-length-dropdown-btn";
    // No separate visible label (match Categories). Keep accessible name.
    lengthBtn.setAttribute("aria-label", "Horizon");

    const lengthMenu = document.createElement("div");
    lengthMenu.className = "rc-length-dropdown";
    lengthMenu.hidden = true;

    // Generate 10-second intervals from 10 to 120 (half-open), then 120-170 inclusive
    const LENGTH_INTERVALS = [];
    for (let min = 10; min < 120; min += 10) {
      const max = min + 10;
      LENGTH_INTERVALS.push({ key: `${min}-${max}`, label: `${min}-${max}s`, min, max, inclusiveMax: false });
    }
    LENGTH_INTERVALS.push({ key: "120-170", label: "120-170s", min: 120, max: 170, inclusiveMax: true });

    const lengthChecks = new Map(); // key -> checkbox
    const lengthMeta = new Map(); // key -> { min, max, inclusiveMax, label, countEl, rowEl }

    function updateLengthButton() {
      const total = LENGTH_INTERVALS.length;
      let checked = 0;
      for (const cb of lengthChecks.values()) if (cb.checked) checked += 1;
      if (checked === total) lengthBtn.textContent = "Horizon (All)";
      else if (checked === 0) lengthBtn.textContent = "Horizon (None)";
      else lengthBtn.textContent = `Horizon (${checked}/${total})`;
    }

    // Header: "All" toggle row
    const lengthHeader = document.createElement("div");
    lengthHeader.className = "rc-length-header";

    const allRow = document.createElement("label");
    allRow.className = "rc-length-all";

    const allCb = document.createElement("input");
    allCb.type = "checkbox";
    allCb.className = "rc-length-all-cb";
    allCb.checked = true;
    allCb.setAttribute("aria-label", "All horizons");

    const allText = document.createElement("span");
    allText.textContent = "All";

    allRow.appendChild(allCb);
    allRow.appendChild(allText);

    lengthHeader.appendChild(allRow);
    lengthMenu.appendChild(lengthHeader);

    function syncAllCheckbox() {
      const total = LENGTH_INTERVALS.length;
      let checked = 0;
      for (const cb of lengthChecks.values()) if (cb.checked) checked += 1;
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
      const total = LENGTH_INTERVALS.length;
      let checked = 0;
      for (const cb of lengthChecks.values()) if (cb.checked) checked += 1;
      const shouldSelectAll = checked !== total; // if not all selected (including none), select all
      for (const cb of lengthChecks.values()) cb.checked = shouldSelectAll;
      syncAllCheckbox();
      updateLengthButton();
      applyFilters();
    });

    for (const it of LENGTH_INTERVALS) {
      const row = document.createElement("label");
      row.className = "rc-length-item";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = true; // default: All
      cb.value = it.key;
      cb.setAttribute("aria-label", it.label);

      const labelSpan = document.createElement("span");
      labelSpan.className = "rc-length-item-label";
      labelSpan.textContent = it.label;

      const countSpan = document.createElement("span");
      countSpan.className = "rc-length-item-count";
      countSpan.textContent = "";

      const rightWrap = document.createElement("span");
      rightWrap.className = "rc-length-item-right";
      rightWrap.appendChild(labelSpan);

      row.appendChild(cb);
      row.appendChild(rightWrap);
      row.appendChild(countSpan);
      lengthMenu.appendChild(row);

      lengthChecks.set(it.key, cb);
      lengthMeta.set(it.key, { min: it.min, max: it.max, inclusiveMax: it.inclusiveMax, label: it.label, countEl: countSpan, rowEl: row });

      cb.addEventListener("change", () => {
        syncAllCheckbox();
        updateLengthButton();
        applyFilters();
      });
    }

    syncAllCheckbox();
    updateLengthButton();

    lengthBtn.addEventListener("click", () => {
      lengthMenu.hidden = !lengthMenu.hidden;
    });
    document.addEventListener("click", (e) => {
      if (lengthWrap.contains(e.target)) return;
      lengthMenu.hidden = true;
    });

    lengthWrap.appendChild(lengthBtn);
    lengthWrap.appendChild(lengthMenu);
    filtersWrap.appendChild(lengthWrap);

    // Meta-category filter (dropdown with multi-select)
    const categoryWrap = document.createElement("div");
    categoryWrap.className = "rc-filter rc-filter-categories";

    const catBtn = document.createElement("button");
    catBtn.type = "button";
    catBtn.className = "rc-category-dropdown-btn";
    catBtn.textContent = "Categories";

    const catMenu = document.createElement("div");
    catMenu.className = "rc-category-dropdown";
    catMenu.hidden = true;

    const CATEGORIES = [
      { key: "food prep", label: "Food Preparation" },
      { key: "cooking", label: "Cooking" },
      { key: "beverage preparation", label: "Beverage Preparation" },
      { key: "organizing and storage", label: "Organizing and Storage" },
      { key: "serving", label: "Serving" },
      { key: "cleaning and sanitizing", label: "Cleaning and Sanitizing" },
    ];
    CATEGORIES.sort((a, b) => a.label.toLowerCase().localeCompare(b.label.toLowerCase()));

    const categoryChecks = new Map();
    const categoryCountEls = new Map(); // key -> span
    const categoryRows = new Map(); // key -> row
    for (const { key, label: lab } of CATEGORIES) {
      const item = document.createElement("label");
      item.className = "rc-category-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = true;
      cb.value = key;
      cb.setAttribute("aria-label", lab);
      const txt = document.createElement("span");
      // Render label as a tag (same style/color as activity meta tags)
      txt.className = "rc-activity-meta rc-category-tag";
      txt.dataset.meta = key;
      txt.textContent = lab;
      const countSpan = document.createElement("span");
      countSpan.className = "rc-category-item-count";
      countSpan.textContent = "";
      item.appendChild(cb);
      const rightWrap = document.createElement("span");
      rightWrap.className = "rc-category-item-right";
      rightWrap.appendChild(txt);
      item.appendChild(rightWrap);
      item.appendChild(countSpan);
      catMenu.appendChild(item);
      categoryChecks.set(key, cb);
      categoryCountEls.set(key, countSpan);
      categoryRows.set(key, item);
    }

    function updateCategoriesButton() {
      const total = CATEGORIES.length;
      let checked = 0;
      for (const cb of categoryChecks.values()) if (cb.checked) checked += 1;
      if (checked === total) catBtn.textContent = "Categories (All)";
      else if (checked === 0) catBtn.textContent = "Categories (None)";
      else catBtn.textContent = `Categories (${checked}/${total})`;
    }
    updateCategoriesButton();

    catBtn.addEventListener("click", () => {
      catMenu.hidden = !catMenu.hidden;
    });
    document.addEventListener("click", (e) => {
      if (categoryWrap.contains(e.target)) return;
      catMenu.hidden = true;
    });

    categoryWrap.appendChild(catBtn);
    categoryWrap.appendChild(catMenu);
    filtersWrap.appendChild(categoryWrap);

    // Skills Involved filter (dropdown with multi-select + counts)
    const attrWrap = document.createElement("div");
    attrWrap.className = "rc-filter rc-filter-task-attrs";

    const attrBtn = document.createElement("button");
    attrBtn.type = "button";
    attrBtn.className = "rc-task-attr-dropdown-btn";
    attrBtn.setAttribute("aria-label", "Skills Involved");

    const attrMenu = document.createElement("div");
    attrMenu.className = "rc-task-attr-dropdown";
    attrMenu.hidden = true;

    const TASK_ATTRS = [
      { key: "nav", label: "Navigation", pillClass: "rc-task-tag rc-task-tag-nav" },
      { key: "door_open", label: "Open Door", pillClass: "rc-task-tag rc-task-tag-door-open" },
      { key: "door_close", label: "Close Door", pillClass: "rc-task-tag rc-task-tag-door-close" },
      { key: "drawer_open", label: "Open Drawer", pillClass: "rc-task-tag rc-task-tag-drawer-open" },
      { key: "drawer_close", label: "Close Drawer", pillClass: "rc-task-tag rc-task-tag-drawer-close" },
      { key: "start_toaster", label: "Start Toaster", pillClass: "rc-task-tag rc-task-tag-start-toaster" },
      { key: "stand_mixer_close", label: "Close Stand Mixer", pillClass: "rc-task-tag rc-task-tag-stand-mixer-close" },
      { key: "blender_lid_open", label: "Open Blender Lid", pillClass: "rc-task-tag rc-task-tag-blender-lid-open" },
      { key: "blender_lid_close", label: "Close Blender Lid", pillClass: "rc-task-tag rc-task-tag-blender-lid-close" },
      { key: "kettle_lid_close", label: "Close Kettle Lid", pillClass: "rc-task-tag rc-task-tag-kettle-lid-close" },
      { key: "rack_slide", label: "Slide Rack", pillClass: "rc-task-tag rc-task-tag-rack-slide" },
      { key: "knob_twist", label: "Twist Knob", pillClass: "rc-task-tag rc-task-tag-knob-twist" },
      { key: "lever_turn", label: "Turn Lever", pillClass: "rc-task-tag rc-task-tag-lever-turn" },
      { key: "button_press", label: "Press Button", pillClass: "rc-task-tag rc-task-tag-button-press" },
      { key: "PickPlace", label: "Pick & Place", pillClass: "rc-task-tag rc-task-tag-PickPlace" },
    ];

    const attrChecks = new Map(); // key -> checkbox
    const attrCountEls = new Map(); // key -> span
    const attrRows = new Map(); // key -> row

    function getVisibleAttrKeys() {
      const keys = [];
      for (const it of TASK_ATTRS) {
        const row = attrRows.get(it.key);
        if (row && !row.hidden) keys.push(it.key);
      }
      return keys;
    }

    function updateAttrButton() {
      const visibleKeys = getVisibleAttrKeys();
      const total = visibleKeys.length;
      let checked = 0;
      for (const k of visibleKeys) if (attrChecks.get(k)?.checked) checked += 1;
      if (total === 0) attrBtn.textContent = "Skills Involved (None)";
      else if (checked === total) attrBtn.textContent = "Skills Involved (All)";
      else if (checked === 0) attrBtn.textContent = "Skills Involved (None)";
      else attrBtn.textContent = `Skills Involved (${checked}/${total})`;
    }

    const attrHeader = document.createElement("div");
    attrHeader.className = "rc-task-attr-header";
    const attrAllRow = document.createElement("label");
    attrAllRow.className = "rc-task-attr-all";
    const attrAllCb = document.createElement("input");
    attrAllCb.type = "checkbox";
    attrAllCb.className = "rc-task-attr-all-cb";
    attrAllCb.checked = true;
    attrAllCb.setAttribute("aria-label", "All skills involved");
    const attrAllText = document.createElement("span");
    attrAllText.textContent = "All";
    attrAllRow.appendChild(attrAllCb);
    attrAllRow.appendChild(attrAllText);
    attrHeader.appendChild(attrAllRow);
    attrMenu.appendChild(attrHeader);

    function syncAttrAllCheckbox() {
      const visibleKeys = getVisibleAttrKeys();
      const total = visibleKeys.length;
      let checked = 0;
      for (const k of visibleKeys) if (attrChecks.get(k)?.checked) checked += 1;
      if (total === 0 || checked === 0) {
        attrAllCb.checked = false;
        attrAllCb.indeterminate = false;
      } else if (checked === total) {
        attrAllCb.checked = true;
        attrAllCb.indeterminate = false;
      } else {
        attrAllCb.checked = false;
        attrAllCb.indeterminate = true;
      }
    }

    attrAllCb.addEventListener("change", () => {
      const visibleKeys = getVisibleAttrKeys();
      const total = visibleKeys.length;
      let checked = 0;
      for (const k of visibleKeys) if (attrChecks.get(k)?.checked) checked += 1;
      const shouldSelectAll = total > 0 && checked !== total;
      for (const k of visibleKeys) {
        const cb = attrChecks.get(k);
        if (cb) cb.checked = shouldSelectAll;
      }
      syncAttrAllCheckbox();
      updateAttrButton();
      applyFilters();
    });

    for (const it of TASK_ATTRS) {
      const row = document.createElement("label");
      row.className = "rc-task-attr-item";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = true;
      cb.value = it.key;
      cb.setAttribute("aria-label", it.label);

      const pill = document.createElement("span");
      pill.className = it.pillClass;
      pill.textContent = it.label;

      const countSpan = document.createElement("span");
      countSpan.className = "rc-task-attr-count";
      countSpan.textContent = "";

      row.appendChild(cb);
      const rightWrap = document.createElement("span");
      rightWrap.className = "rc-task-attr-item-right";
      rightWrap.appendChild(pill);
      row.appendChild(rightWrap);
      row.appendChild(countSpan);
      attrMenu.appendChild(row);

      attrChecks.set(it.key, cb);
      attrCountEls.set(it.key, countSpan);
      attrRows.set(it.key, row);

      cb.addEventListener("change", () => {
        syncAttrAllCheckbox();
        updateAttrButton();
        applyFilters();
      });
    }

    syncAttrAllCheckbox();
    updateAttrButton();

    attrBtn.addEventListener("click", () => {
      attrMenu.hidden = !attrMenu.hidden;
    });
    document.addEventListener("click", (e) => {
      if (attrWrap.contains(e.target)) return;
      attrMenu.hidden = true;
    });

    attrWrap.appendChild(attrBtn);
    attrWrap.appendChild(attrMenu);
    filtersWrap.appendChild(attrWrap);

    // Target Tasks filter (All Tasks vs target-only)
    const targetWrap = document.createElement("div");
    targetWrap.className = "rc-filter rc-filter-target-tasks";

    const targetBtn = document.createElement("button");
    targetBtn.type = "button";
    targetBtn.className = "rc-task-attr-dropdown-btn";
    targetBtn.setAttribute("aria-label", "Target Tasks");

    const targetMenu = document.createElement("div");
    targetMenu.className = "rc-task-attr-dropdown";
    targetMenu.hidden = true;

    const TARGET_TASKS = [
      { key: "comp_seen_target", label: "Composite-Seen", pillClass: "rc-task-tag rc-task-tag-comp-seen" },
      { key: "comp_unseen_target", label: "Composite-Unseen", pillClass: "rc-task-tag rc-task-tag-comp-unseen" },
    ];

    // Button label should be stable
    targetBtn.textContent = "Target Tasks";

    const targetChecks = new Map(); // key -> checkbox
    const targetCountEls = new Map(); // key -> span

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

    // Divider (matches other dropdowns visually)
    const targetDivider = document.createElement("div");
    targetDivider.style.height = "1px";
    targetDivider.style.background = "var(--pst-color-border, #e0e0e0)";
    targetDivider.style.margin = "0.25rem 0.15rem 0.35rem 0";
    targetMenu.appendChild(targetDivider);

    function syncFromAllTasks() {
      if (!allTasksCb.checked) return;
      // If "All Tasks" is selected, other two are selected by default.
      for (const cb of targetChecks.values()) cb.checked = true;
    }

    for (const it of TARGET_TASKS) {
      const row = document.createElement("label");
      row.className = "rc-task-attr-item";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = true;
      cb.value = it.key;
      cb.setAttribute("aria-label", it.label);

      const pill = document.createElement("span");
      pill.className = it.pillClass;
      pill.textContent = it.label;

      const countSpan = document.createElement("span");
      countSpan.className = "rc-task-attr-count";
      countSpan.textContent = "";

      row.appendChild(cb);
      const rightWrap = document.createElement("span");
      rightWrap.className = "rc-task-attr-item-right";
      rightWrap.appendChild(pill);
      row.appendChild(rightWrap);
      row.appendChild(countSpan);
      targetMenu.appendChild(row);

      targetChecks.set(it.key, cb);
      targetCountEls.set(it.key, countSpan);

      cb.addEventListener("change", () => {
        // If user customizes seen/unseen, they are no longer in "All Tasks" mode.
        allTasksCb.checked = false;
        applyFilters();
      });
    }

    allTasksCb.addEventListener("change", () => {
      syncFromAllTasks();
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
    filtersWrap.appendChild(targetWrap);

    // Filter order: Subtasks, Categories, Skills Involved, Horizon, Target Tasks
    // Re-append nodes to deterministically enforce order.
    filtersWrap.appendChild(categoryWrap);
    filtersWrap.appendChild(attrWrap);
    filtersWrap.appendChild(lengthWrap);
    filtersWrap.appendChild(targetWrap);

    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "rc-reset-filters-btn";
    resetBtn.textContent = "Reset";
    resetBtn.setAttribute("aria-label", "Reset filters");
    resetBtn.addEventListener("click", () => {
      // Reset filter inputs to defaults
      minInput.value = "";
      maxInput.value = "";
      lastValidSubtasks = { minV: 2, maxV: 16 };
      for (const cb of lengthChecks.values()) cb.checked = true;
      updateLengthButton();
      lengthMenu.hidden = true;
      // sync the All checkbox state
      // (it will be checked since all are selected)
      // eslint-disable-next-line no-use-before-define
      syncAllCheckbox();
      for (const cb of categoryChecks.values()) cb.checked = true;
      updateCategoriesButton();
      updateSubtasksValidityAndNote();
      catMenu.hidden = true;
      for (const cb of attrChecks.values()) cb.checked = true;
      // eslint-disable-next-line no-use-before-define
      syncAttrAllCheckbox();
      updateAttrButton();
      attrMenu.hidden = true;
      allTasksCb.checked = true;
      for (const cb of targetChecks.values()) cb.checked = true;
      targetMenu.hidden = true;
      applyFilters();
    });
    filtersWrap.appendChild(resetBtn);

    // Pre-fill per-row datasets so indexing/filtering doesn't depend on JS-inserted columns.
    if (attrsMap || episodeLengthMap) {
      for (const d of Array.from(content.querySelectorAll("details.rc-activity"))) {
        for (const tr of Array.from(d.querySelectorAll("table tbody tr"))) {
          const name = taskNameFromRow(tr);
          if (!name) continue;
          const attrs = attrsMap ? attrsMap.get(name) : null;
          const mobileVal = attrs?.mobile;
          tr.dataset.rcMobile = mobileVal === "Yes" || mobileVal === "No" ? mobileVal : "";
          const st = attrs?.subtasks;
          if (Number.isFinite(st)) tr.dataset.rcSubtasks = String(st);
          const len = episodeLengthMap ? episodeLengthMap.get(name) : null;
          if (len != null && Number.isFinite(len)) tr.dataset.rcLength = String(Math.round(len));
        }
      }
    }

    // Build task search + suggestions
    const taskIndex = buildTaskIndex(content);

    // Sort Skills Involved filter popup by frequency (most to least frequent)
    // This only affects the filter popup order, not the tag display order in tables
    function sortAttrFilterByFrequency() {
      // Calculate frequency of each skill
      const skillFreq = new Map();
      for (const attr of TASK_ATTRS) {
        skillFreq.set(attr.key, 0);
      }
      for (const item of taskIndex) {
        const keys = item.tagKeys || [];
        if (Array.isArray(keys)) {
          for (const key of keys) {
            if (skillFreq.has(key)) {
              skillFreq.set(key, skillFreq.get(key) + 1);
            }
          }
        }
      }

      // Sort TASK_ATTRS by frequency (descending)
      const sortedAttrs = [...TASK_ATTRS].sort((a, b) => {
        const freqA = skillFreq.get(a.key) || 0;
        const freqB = skillFreq.get(b.key) || 0;
        return freqB - freqA; // Descending order
      });

      // Reorder the rows in the filter popup (keep header, reorder item rows)
      const header = attrMenu.querySelector('.rc-task-attr-header');
      // Remove all item rows (but keep header)
      for (const it of TASK_ATTRS) {
        const row = attrRows.get(it.key);
        if (row && row.parentNode === attrMenu) {
          attrMenu.removeChild(row);
        }
      }

      // Re-append rows in sorted order (after header)
      for (const it of sortedAttrs) {
        const row = attrRows.get(it.key);
        if (row) {
          attrMenu.appendChild(row);
        }
      }
    }
    sortAttrFilterByFrequency();

    // "Show all tasks" toggle (opens all activity accordions)
    const showAllWrap = document.createElement("div");
    showAllWrap.className = "rc-show-all";
    const showAllLabel = document.createElement("label");
    showAllLabel.className = "rc-show-all-label";
    const showAllCb = document.createElement("input");
    showAllCb.type = "checkbox";
    showAllCb.id = "rc-show-all";
    const showAllText = document.createElement("span");
    showAllText.textContent = "Show all tasks";
    showAllLabel.appendChild(showAllCb);
    showAllLabel.appendChild(showAllText);
    showAllWrap.appendChild(showAllLabel);

    let prevOpenIds = new Set();
    function getVisibleActivities() {
      return Array.from(content.querySelectorAll("details.rc-activity")).filter(
        (d) => d.style.display !== "none"
      );
    }

    function applyShowAllState() {
      const visible = getVisibleActivities();
      if (showAllCb.checked) {
        for (const d of visible) d.open = true;
      } else {
        for (const d of visible) d.open = prevOpenIds.has(d.id);
      }
    }

    showAllCb.addEventListener("change", () => {
      if (showAllCb.checked) {
        prevOpenIds = new Set(
          Array.from(content.querySelectorAll("details.rc-activity"))
            .filter((d) => d.open)
            .map((d) => d.id)
        );
      }
      applyShowAllState();
    });

    let lastValidSubtasks = { minV: 2, maxV: 16 };

    function parseBound(s) {
      const raw = String(s ?? "").trim();
      if (raw === "") return { ok: true, value: null };
      const v = Number.parseInt(raw, 10);
      if (!Number.isFinite(v)) return { ok: false, value: null };
      return { ok: true, value: v };
    }

    function setInvalid(el, isInvalid) {
      el.classList.toggle("rc-invalid", Boolean(isInvalid));
    }

    /** Restrict to 1–2 positive digits only (no minus, no 100+). */
    function sanitizeSubtasksInput(val) {
      return String(val ?? "").replace(/[^0-9]/g, "").slice(0, 2);
    }

    function syncMinMaxIfCrossed(changed) {
      const minP = parseBound(minInput.value);
      const maxP = parseBound(maxInput.value);
      if (!minP.ok || !maxP.ok) return;
      if (minP.value == null || maxP.value == null) return;
      if (minP.value <= maxP.value) return;
      // On blur only: fix the field that was edited so the other value is preserved
      // (e.g. 5-1 after editing max → 5-5, so user's min is kept).
      if (changed === "min") maxInput.value = String(minP.value);
      else maxInput.value = String(minP.value);
    }

    function getActiveLengthIntervals() {
      try {
        const total = LENGTH_INTERVALS.length;
        let checked = 0;
        const active = [];
        for (const it of LENGTH_INTERVALS) {
          const cb = lengthChecks.get(it.key);
          if (cb && cb.checked) {
            checked += 1;
            active.push(it);
          }
        }
        // All checked => no length constraint
        if (checked === total) return null;
        // None checked => impossible filter
        if (checked === 0) return [];
        return active;
      } catch (e) {
        console.error("Error getting active length intervals:", e);
      }
      return null;
    }

    function lengthMatchesInterval(lengthSeconds, interval) {
      if (!Number.isFinite(lengthSeconds)) return false;
      if (interval.inclusiveMax) return lengthSeconds >= interval.min && lengthSeconds <= interval.max;
      return lengthSeconds >= interval.min && lengthSeconds < interval.max;
    }

    function getActiveTaskAttrKeys() {
      const total = TASK_ATTRS.length;
      let checked = 0;
      const active = [];
      for (const it of TASK_ATTRS) {
        const cb = attrChecks.get(it.key);
        if (cb && cb.checked) {
          checked += 1;
          active.push(it.key);
        }
      }
      if (checked === total) return null; // no constraint
      if (checked === 0) return []; // impossible
      return active;
    }

    function getActiveTargetTaskKeys() {
      // If "All Tasks" is selected, we do not filter by target tasks.
      if (allTasksCb.checked) return null;
      const total = TARGET_TASKS.length;
      let checked = 0;
      const active = [];
      for (const it of TARGET_TASKS) {
        const cb = targetChecks.get(it.key);
        if (cb && cb.checked) {
          checked += 1;
          active.push(it.key);
        }
      }
      if (checked === 0) return []; // impossible
      return active;
    }

    function taskHasAnyAttr(item, activeKeys) {
      if (activeKeys == null) return true;
      if (!Array.isArray(activeKeys) || activeKeys.length === 0) return false;
      const s = new Set(activeKeys);
      const keys = item?.tagKeys || [];
      return Array.isArray(keys) && keys.some((k) => s.has(k));
    }

    function updateLengthDropdownCounts() {
      try {
        // Get current filters excluding length
        const minP = parseBound(minInput.value);
        const maxP = parseBound(maxInput.value);
        const minV = minP.value == null ? 2 : minP.value;
        const maxV = maxP.value == null ? 16 : maxP.value;
        const minOut = minP.ok && minP.value != null && (minP.value < 2 || minP.value > 16);
        const maxOut = maxP.ok && maxP.value != null && (maxP.value < 2 || maxP.value > 16);
        const inRange = !minOut && !maxOut && minP.ok && maxP.ok;
        const effectiveSubtasks = inRange ? { minV, maxV } : lastValidSubtasks;

        const cats = new Set();
        for (const [cat, cb] of categoryChecks.entries()) {
          if (cb.checked) cats.add(cat);
        }

        const attrKeys = getActiveTaskAttrKeys();
        const targetKeys = getActiveTargetTaskKeys();

        // Count tasks for each interval (excluding length filter)
        for (const it of LENGTH_INTERVALS) {
          let count = 0;
          for (const item of taskIndex) {
            // Check other filters (nav, subtasks, categories)
            if (cats.size > 0) {
              if (!item.metaCategory || !cats.has(item.metaCategory)) continue;
            } else if (cats.size === 0) {
              continue;
            }
            if (Array.isArray(targetKeys)) {
              if (targetKeys.length === 0) continue;
              const keys = item.tagKeys || [];
              const set = new Set(targetKeys);
              if (!Array.isArray(keys) || !keys.some((k) => set.has(k))) continue;
            }
            if (!taskHasAnyAttr(item, attrKeys)) continue;
            if (item.subtasks == null) continue;
            if (item.subtasks < effectiveSubtasks.minV || item.subtasks > effectiveSubtasks.maxV) continue;

            if (item.length == null) continue;
            if (!lengthMatchesInterval(item.length, it)) continue;
            count += 1;
          }

          const meta = lengthMeta.get(it.key);
          if (meta?.countEl) meta.countEl.textContent = formatCount(count);
          if (meta?.rowEl) {
            meta.rowEl.hidden = count === 0;
          }
        }

        // Refresh header + button state (checkboxes are not mutated here)
        syncAllCheckbox();
        updateLengthButton();
      } catch (e) {
        console.error("Error updating length dropdown counts:", e);
      }
    }

    function updateTaskAttrDropdownCounts() {
      try {
        // Get current filters excluding task attributes
        const minP = parseBound(minInput.value);
        const maxP = parseBound(maxInput.value);
        const minV = minP.value == null ? 2 : minP.value;
        const maxV = maxP.value == null ? 16 : maxP.value;
        const minOut = minP.ok && minP.value != null && (minP.value < 2 || minP.value > 16);
        const maxOut = maxP.ok && maxP.value != null && (maxP.value < 2 || maxP.value > 16);
        const inRange = !minOut && !maxOut && minP.ok && maxP.ok;
        const effectiveSubtasks = inRange ? { minV, maxV } : lastValidSubtasks;

        const cats = new Set();
        for (const [cat, cb] of categoryChecks.entries()) {
          if (cb.checked) cats.add(cat);
        }

        const lengthIntervalsActive = getActiveLengthIntervals();
        const targetKeys = getActiveTargetTaskKeys();

        for (const ta of TASK_ATTRS) {
          let count = 0;
          for (const item of taskIndex) {
            if (cats.size > 0) {
              if (!item.metaCategory || !cats.has(item.metaCategory)) continue;
            } else if (cats.size === 0) {
              continue;
            }
            if (Array.isArray(targetKeys)) {
              if (targetKeys.length === 0) continue;
              const keys = item.tagKeys || [];
              const set = new Set(targetKeys);
              if (!Array.isArray(keys) || !keys.some((k) => set.has(k))) continue;
            }

            if (item.subtasks == null) continue;
            if (item.subtasks < effectiveSubtasks.minV || item.subtasks > effectiveSubtasks.maxV) continue;

            if (item.length == null) continue;
            if (Array.isArray(lengthIntervalsActive)) {
              if (lengthIntervalsActive.length === 0) continue;
              let ok = false;
              for (const it of lengthIntervalsActive) {
                if (lengthMatchesInterval(item.length, it)) {
                  ok = true;
                  break;
                }
              }
              if (!ok) continue;
            }

            const keys = item.tagKeys || [];
            if (!Array.isArray(keys) || !keys.includes(ta.key)) continue;
            count += 1;
          }

          const el = attrCountEls.get(ta.key);
          if (el) el.textContent = formatCount(count);
          const row = attrRows.get(ta.key);
          if (row) row.hidden = count === 0;
        }

        // Refresh header + button state (checkboxes are not mutated here)
        syncAttrAllCheckbox();
        updateAttrButton();
      } catch (e) {
        console.error("Error updating task attribute dropdown counts:", e);
      }
    }

    function updateCategoryDropdownCounts() {
      try {
        // Get current filters excluding categories
        const minP = parseBound(minInput.value);
        const maxP = parseBound(maxInput.value);
        const minV = minP.value == null ? 2 : minP.value;
        const maxV = maxP.value == null ? 16 : maxP.value;
        const minOut = minP.ok && minP.value != null && (minP.value < 2 || minP.value > 16);
        const maxOut = maxP.ok && maxP.value != null && (maxP.value < 2 || maxP.value > 16);
        const inRange = !minOut && !maxOut && minP.ok && maxP.ok;
        const effectiveSubtasks = inRange ? { minV, maxV } : lastValidSubtasks;

        const lengthIntervalsActive = getActiveLengthIntervals();
        const attrKeys = getActiveTaskAttrKeys();
        const targetKeys = getActiveTargetTaskKeys();

        for (const { key } of CATEGORIES) {
          let count = 0;
          for (const item of taskIndex) {
            if (!item.metaCategory || item.metaCategory !== key) continue;
            if (Array.isArray(targetKeys)) {
              if (targetKeys.length === 0) continue;
              const keys = item.tagKeys || [];
              const set = new Set(targetKeys);
              if (!Array.isArray(keys) || !keys.some((k) => set.has(k))) continue;
            }
            if (!taskHasAnyAttr(item, attrKeys)) continue;

            if (item.subtasks == null) continue;
            if (item.subtasks < effectiveSubtasks.minV || item.subtasks > effectiveSubtasks.maxV) continue;

            if (item.length == null) continue;
            if (Array.isArray(lengthIntervalsActive)) {
              if (lengthIntervalsActive.length === 0) continue;
              let ok = false;
              for (const it of lengthIntervalsActive) {
                if (lengthMatchesInterval(item.length, it)) {
                  ok = true;
                  break;
                }
              }
              if (!ok) continue;
            }

            count += 1;
          }
          const el = categoryCountEls.get(key);
          if (el) el.textContent = formatCount(count);
          const row = categoryRows.get(key);
          if (row) row.hidden = count === 0;
        }

        // Refresh button state (checkboxes are not mutated here)
        updateCategoriesButton();
      } catch (e) {
        console.error("Error updating category dropdown counts:", e);
      }
    }

    function updateTargetTaskDropdownCounts() {
      try {
        // Get current filters excluding target tasks
        const minP = parseBound(minInput.value);
        const maxP = parseBound(maxInput.value);
        const minV = minP.value == null ? 2 : minP.value;
        const maxV = maxP.value == null ? 16 : maxP.value;
        const minOut = minP.ok && minP.value != null && (minP.value < 2 || minP.value > 16);
        const maxOut = maxP.ok && maxP.value != null && (maxP.value < 2 || maxP.value > 16);
        const inRange = !minOut && !maxOut && minP.ok && maxP.ok;
        const effectiveSubtasks = inRange ? { minV, maxV } : lastValidSubtasks;

        const cats = new Set();
        for (const [cat, cb] of categoryChecks.entries()) {
          if (cb.checked) cats.add(cat);
        }

        const lengthIntervalsActive = getActiveLengthIntervals();
        const attrKeys = getActiveTaskAttrKeys();

        // All Tasks count (total tasks in table)
        allTasksCount.textContent = formatCount(taskIndex.length);

        for (const tt of TARGET_TASKS) {
          let count = 0;
          for (const item of taskIndex) {
            if (cats.size > 0) {
              if (!item.metaCategory || !cats.has(item.metaCategory)) continue;
            } else if (cats.size === 0) {
              continue;
            }
            if (!taskHasAnyAttr(item, attrKeys)) continue;

            if (item.subtasks == null) continue;
            if (item.subtasks < effectiveSubtasks.minV || item.subtasks > effectiveSubtasks.maxV) continue;

            if (item.length == null) continue;
            if (Array.isArray(lengthIntervalsActive)) {
              if (lengthIntervalsActive.length === 0) continue;
              let ok = false;
              for (const it of lengthIntervalsActive) {
                if (lengthMatchesInterval(item.length, it)) {
                  ok = true;
                  break;
                }
              }
              if (!ok) continue;
            }

            const keys = item.tagKeys || [];
            if (!Array.isArray(keys) || !keys.includes(tt.key)) continue;
            count += 1;
          }

          const el = targetCountEls.get(tt.key);
          if (el) el.textContent = formatCount(count);
        }
      } catch (e) {
        console.error("Error updating target task dropdown counts:", e);
      }
    }

    function updateSubtasksValidityAndNote() {
      const minP = parseBound(minInput.value);
      const maxP = parseBound(maxInput.value);

      const minOut = minP.ok && minP.value != null && (minP.value < 2 || minP.value > 16);
      const maxOut = maxP.ok && maxP.value != null && (maxP.value < 2 || maxP.value > 16);

      setInvalid(minInput, minOut);
      setInvalid(maxInput, maxOut);

      const show = minOut || maxOut;
      note.hidden = !show;
    }

    function getActiveFilters() {
      const minP = parseBound(minInput.value);
      const maxP = parseBound(maxInput.value);

      // Empty input is allowed; it means "use default boundary"
      const minV = minP.value == null ? 2 : minP.value;
      const maxV = maxP.value == null ? 16 : maxP.value;

      const minOut = minP.ok && minP.value != null && (minP.value < 2 || minP.value > 16);
      const maxOut = maxP.ok && maxP.value != null && (maxP.value < 2 || maxP.value > 16);

      const inRange = !minOut && !maxOut && minP.ok && maxP.ok;
      if (inRange) lastValidSubtasks = { minV, maxV };

      const effective = inRange ? { minV, maxV } : lastValidSubtasks;

      // Length filter (multi-select)
      const lengthIntervalsActive = getActiveLengthIntervals();
      const taskAttrKeys = getActiveTaskAttrKeys();
      const targetTaskKeys = getActiveTargetTaskKeys();

      const cats = new Set();
      for (const [cat, cb] of categoryChecks.entries()) {
        if (cb.checked) cats.add(cat);
      }
      return {
        minV: effective.minV,
        maxV: effective.maxV,
        hasError: !inRange,
        lengthIntervals: lengthIntervalsActive,
        lengthHasError: false, // Dropdown always has valid values
        taskAttrs: taskAttrKeys,
        targetTasks: targetTaskKeys,
        cats,
      };
    }

    function passesFilters(item, f) {
      if (f.cats && f.cats.size > 0) {
        if (!item.metaCategory || !f.cats.has(item.metaCategory)) return false;
      } else if (f.cats && f.cats.size === 0) {
        return false;
      }
      // Target Tasks filter
      if (Array.isArray(f.targetTasks)) {
        if (f.targetTasks.length === 0) return false;
        const keys = item.tagKeys || [];
        const set = new Set(f.targetTasks);
        if (!Array.isArray(keys) || !keys.some((k) => set.has(k))) return false;
      }
      if (Array.isArray(f.taskAttrs)) {
        if (f.taskAttrs.length === 0) return false;
        const keys = item.tagKeys || [];
        const set = new Set(f.taskAttrs);
        if (!Array.isArray(keys) || !keys.some((k) => set.has(k))) return false;
      }
      if (item.subtasks == null) return false;
      if (item.subtasks < f.minV || item.subtasks > f.maxV) return false;
      // Length filter
      if (item.length == null) return false;
      if (Array.isArray(f.lengthIntervals)) {
        if (f.lengthIntervals.length === 0) return false;
        let ok = false;
        for (const it of f.lengthIntervals) {
          if (lengthMatchesInterval(item.length, it)) {
            ok = true;
            break;
          }
        }
        if (!ok) return false;
      }
      return true;
    }

    function updateActivityBadgesAndVisibility(visibleByActivity) {
      for (const d of Array.from(content.querySelectorAll("details.rc-activity"))) {
        const id = d.id;
        const badge = d.querySelector("summary .rc-activity-badge");
        const total = d.querySelectorAll("table tbody tr").length;
        const visible = visibleByActivity.get(id) ?? 0;
        if (badge) badge.textContent = `${visible} task${visible === 1 ? "" : "s"}`;
        d.style.display = visible === 0 ? "none" : "";
      }

      // Disable activity dropdown options with 0 visible tasks
      const map = new Map();
      for (const [id, n] of visibleByActivity.entries()) map.set(id, n);
      for (const opt of Array.from(select.querySelectorAll("option"))) {
        if (!opt.value) continue;
        const n = map.get(opt.value) ?? 0;
        opt.disabled = n === 0;
        opt.hidden = n === 0;
      }

      if (select.value && (visibleByActivity.get(select.value) ?? 0) === 0) {
        select.value = "";
      }
    }

    function applyFilters() {
      const f = getActiveFilters();

      const visibleByActivity = new Map();
      let visibleTotal = 0;
      for (const it of taskIndex) {
        const ok = passesFilters(it, f);
        it.rowEl.style.display = ok ? "" : "none";
        if (ok) {
          visibleTotal += 1;
          visibleByActivity.set(it.activityId, (visibleByActivity.get(it.activityId) ?? 0) + 1);
        }
      }
      updateActivityBadgesAndVisibility(visibleByActivity);
      taskCount.textContent = `Showing ${visibleTotal} task${visibleTotal === 1 ? "" : "s"}`;

      // If "show all" is enabled, keep all visible activities open
      if (showAllCb.checked) {
        for (const d of getVisibleActivities()) d.open = true;
      }

      // refresh suggestions if open
      if (!suggest.hidden && input.value.trim()) {
        lastMatches = renderSuggestions(input.value);
      }
      
      // Update length dropdown counts (shows counts based on current other filters)
      updateLengthDropdownCounts();
      // Update task-attribute dropdown counts (shows counts based on current other filters)
      updateTaskAttrDropdownCounts();
      // Update category dropdown counts (shows counts based on current other filters)
      updateCategoryDropdownCounts();
      // Update target tasks dropdown counts (shows counts based on current other filters)
      updateTargetTaskDropdownCounts();
    }

    const searchWrap = document.createElement("div");
    searchWrap.className = "rc-task-search";

    const searchLabel = document.createElement("label");
    searchLabel.textContent = "Task:";
    searchLabel.setAttribute("for", "rc-task-input");

    const input = document.createElement("input");
    input.id = "rc-task-input";
    input.type = "search";
    input.placeholder = "Search a task…";
    input.autocomplete = "off";
    input.spellcheck = false;

    const suggest = document.createElement("div");
    suggest.className = "rc-task-suggest";
    suggest.hidden = true;
    suggest.setAttribute("role", "listbox");

    function renderSuggestions(q) {
      const queryRaw = (q || "").trim();
      const query = queryRaw.toLowerCase();
      suggest.innerHTML = "";
      const tokens = uniqueTokens(queryRaw);
      if (!tokens.length) {
        suggest.hidden = true;
        return [];
      }
      const f = getActiveFilters();
      const matches = taskIndex
        .filter((it) => passesFilters(it, f) && tokens.every((t) => it.searchText.includes(t)))
        .sort((a, b) => {
          // lower score = better
          // Prefer matches in the task name itself over matches only in the activity name.
          const aTaskText = `${a.taskLower} ${a.taskSpacedLower}`;
          const bTaskText = `${b.taskLower} ${b.taskSpacedLower}`;
          const aTaskMatch = tokens.every((t) => aTaskText.includes(t)) ? 0 : 1;
          const bTaskMatch = tokens.every((t) => bTaskText.includes(t)) ? 0 : 1;
          if (aTaskMatch !== bTaskMatch) return aTaskMatch - bTaskMatch;

          const aExact = a.taskLower === query ? 0 : 1;
          const bExact = b.taskLower === query ? 0 : 1;
          if (aExact !== bExact) return aExact - bExact;

          const t0 = tokens[0];
          const aPref = a.taskLower.startsWith(t0) || a.taskSpacedLower.startsWith(query) ? 0 : 1;
          const bPref = b.taskLower.startsWith(t0) || b.taskSpacedLower.startsWith(query) ? 0 : 1;
          if (aPref !== bPref) return aPref - bPref;

          const aPos = a.taskLower.indexOf(t0);
          const bPos = b.taskLower.indexOf(t0);
          if (aPos !== bPos) return aPos - bPos;

          return a.task.length - b.task.length;
        })
        .slice(0, 12);

      for (const [idx, it] of matches.entries()) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "rc-task-suggest-item";
        btn.id = `rc-task-suggest-${idx}`;
        btn.setAttribute("role", "option");
        btn.setAttribute("aria-selected", "false");
        btn.dataset.task = it.task;
        btn.innerHTML = `<span class="rc-task-suggest-name">${it.task}</span><span class="rc-task-suggest-meta">${it.activityTitle}</span>`;
        btn.addEventListener("click", () => {
          input.value = it.task;
          suggest.hidden = true;
          setActiveIndex(-1);
          scrollToTask(it);
        });
        btn.addEventListener("mouseenter", () => {
          setActiveIndex(idx);
        });
        suggest.appendChild(btn);
      }

      suggest.hidden = matches.length === 0;
      // Reset any prior keyboard selection since we rebuilt the list.
      setActiveIndex(-1);
      return matches;
    }

    let lastMatches = [];
    let activeIndex = -1;

    function setActiveIndex(nextIndex) {
      const btns = Array.from(suggest.querySelectorAll(".rc-task-suggest-item"));
      const clamped = Number.isFinite(nextIndex) ? nextIndex : -1;
      if (btns.length === 0) {
        activeIndex = -1;
      } else if (clamped === -1) {
        activeIndex = -1;
      } else {
        activeIndex = Math.max(0, Math.min(clamped, btns.length - 1));
      }

      for (const [i, b] of btns.entries()) {
        const isActive = i === activeIndex;
        b.classList.toggle("rc-active", isActive);
        b.setAttribute("aria-selected", isActive ? "true" : "false");
      }

      if (activeIndex >= 0) {
        input.setAttribute("aria-activedescendant", `rc-task-suggest-${activeIndex}`);
        // If the dropdown ever becomes scrollable, keep the active item visible.
        btns[activeIndex]?.scrollIntoView?.({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
    }

    input.addEventListener("input", () => {
      lastMatches = renderSuggestions(input.value);
      setActiveIndex(-1);
    });

    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        if (suggest.hidden) lastMatches = renderSuggestions(input.value);
        if (lastMatches.length > 0) {
          suggest.hidden = false;
          setActiveIndex(activeIndex < 0 ? 0 : activeIndex + 1);
        }
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        if (suggest.hidden) lastMatches = renderSuggestions(input.value);
        if (lastMatches.length > 0) {
          suggest.hidden = false;
          setActiveIndex(activeIndex < 0 ? lastMatches.length - 1 : activeIndex - 1);
        }
      } else if (e.key === "Enter") {
        e.preventDefault();
        const f = getActiveFilters();
        const exact = taskIndex.find(
          (it) => passesFilters(it, f) && it.taskLower === input.value.trim().toLowerCase()
        );
        const chosen = (activeIndex >= 0 ? lastMatches[activeIndex] : null) || exact || lastMatches[0];
        if (chosen) {
          suggest.hidden = true;
          setActiveIndex(-1);
          scrollToTask(chosen);
        }
      } else if (e.key === "Escape") {
        suggest.hidden = true;
        setActiveIndex(-1);
      } else if (e.key === "Tab") {
        // keep tab behavior, but close the menu
        suggest.hidden = true;
        setActiveIndex(-1);
      }
    });

    // Click-away closes suggestions
    document.addEventListener("click", (e) => {
      if (!searchWrap.contains(e.target)) {
        suggest.hidden = true;
        setActiveIndex(-1);
      }
    });

    searchWrap.appendChild(searchLabel);
    const inputWrap = document.createElement("div");
    inputWrap.className = "rc-task-input-wrap";
    inputWrap.appendChild(input);
    inputWrap.appendChild(suggest);
    searchWrap.appendChild(inputWrap);

    selectorWrap.appendChild(searchWrap);

    // Show-all toggle goes below Activity + Task controls
    selectorWrap.appendChild(showAllWrap);

    // Filters go below Activity + Task controls
    selectorWrap.appendChild(filtersWrap);
    // Sort goes below filters
    selectorWrap.appendChild(sortWrap);

    // Wire filters after search exists

    minInput.addEventListener("input", () => {
      const sane = sanitizeSubtasksInput(minInput.value);
      if (sane !== minInput.value) minInput.value = sane;
      updateSubtasksValidityAndNote();
      window.setTimeout(applyFilters, 0);
    });
    maxInput.addEventListener("input", () => {
      const sane = sanitizeSubtasksInput(maxInput.value);
      if (sane !== maxInput.value) maxInput.value = sane;
      updateSubtasksValidityAndNote();
      window.setTimeout(applyFilters, 0);
    });
    minInput.addEventListener("change", () => {
      syncMinMaxIfCrossed("min");
      updateSubtasksValidityAndNote();
      applyFilters();
    });
    maxInput.addEventListener("change", () => {
      syncMinMaxIfCrossed("max");
      updateSubtasksValidityAndNote();
      applyFilters();
    });

    for (const cb of categoryChecks.values()) {
      cb.addEventListener("change", () => {
        updateCategoriesButton();
        applyFilters();
      });
    }

    sortSelect.addEventListener("change", () => {
      sortActivities(rootSection, sortSelect.value, select);
      applyFilters();
    });

    // Initial filter pass (defaults show all)
    updateSubtasksValidityAndNote();
    // Ensure initial sort is Activity
    sortActivities(rootSection, "activity", select);
    // Keep the Activity dropdown on the placeholder by default
    // (avoid browser form-state restoration selecting an activity on refresh).
    select.value = "";
    defaultOpt.selected = true;
    applyFilters();
    applyShowAllState();

    // Put selector under the page title, but BELOW the intro blurb if present
    const introSpan = content.querySelector("#rc-composite-activity-count");
    const introP = introSpan ? introSpan.closest("p") : null;
    if (introP && introP.parentNode) {
      introP.insertAdjacentElement("afterend", selectorWrap);
    } else {
      const titleH1 = content.querySelector("h1");
      if (titleH1 && titleH1.parentNode) {
        titleH1.insertAdjacentElement("afterend", selectorWrap);
      } else {
        content.insertAdjacentElement("afterbegin", selectorWrap);
      }
    }

    ensureBackToTopButton();

  });

  onReady(() => {
    if (!isObjectsPage()) return;
    initObjectsTableSorting();
    ensureBackToTopButton();
  });

  onReady(() => {
    if (!isFoundationModelLearningPage()) return;
    document.body.classList.add("rc-foundation-model-learning");
  });

  onReady(() => {
    if (!isLifelongLearningPage()) return;
    document.body.classList.add("rc-lifelong-learning");
  });

  onReady(() => {
    if (!isLifelongLearningPage()) return;
    document.body.classList.add("rc-lifelong-learning");
  });

  // Datasets overview page: make target task tables match the Composite Tasks look.
  onReady(() => {
    if (!isDatasetsOverviewPage()) return;
    document.body.classList.add("rc-composite-tasks", "rc-datasets-overview");

    const ATOMIC_SEEN_TASKS = [
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
    ];

    const COMPOSITE_SEEN_TASKS = [
      "DeliverStraw",
      "GetToastedBread",
      "KettleBoiling",
      "LoadDishwasher",
      "PackIdenticalLunches",
      "PreSoakPan",
      "PrepareCoffee",
      "RinseSinkBasin",
      "ScrubCuttingBoard",
      "SearingMeat",
      "SetUpCuttingStation",
      "StackBowlsCabinet",
      "SteamInMicrowave",
      "StirVegetables",
      "StoreLeftoversInBowl",
      "WashLettuce",
    ];

    const COMPOSITE_UNSEEN_TASKS = [
      "ArrangeBreadBasket",
      "ArrangeTea",
      "BreadSelection",
      "CategorizeCondiments",
      "CuttingToolSelection",
      "GarnishPancake",
      "GatherTableware",
      "HeatKebabSandwich",
      "MakeIceLemonade",
      "PanTransfer",
      "PortionHotDogs",
      "RecycleBottlesByType",
      "SeparateFreezerRack",
      "WaffleReheat",
      "WashFruitColander",
      "WeighIngredients",
    ];

    function escapeHtml(s) {
      return String(s || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function formatAtomicDescription(desc) {
      // Match Atomic Tasks page formatting for variables.
      const raw = String(desc || "");
      const esc = escapeHtml(raw);
      let out = esc.replace(/\{([^}]+)\}/g, (_m, inner) => `<span class="rc-atomic-var"><em>${inner}</em></span>`);
      out = out.replace(/\[([^\]]+)\]/g, (_m, inner) => `[<span class="rc-atomic-var"><em>${inner}</em></span>]`);
      return out;
    }

    function getAtomicClassHref(taskName) {
      const idx = window.ROBOCASA_ATOMIC_TASK_INDEX;
      const fixtures = idx && Array.isArray(idx.fixtures) ? idx.fixtures : [];
      for (const fx of fixtures) {
        const tasks = Array.isArray(fx.tasks) ? fx.tasks : [];
        for (const t of tasks) {
          if (!t || t.name !== taskName) continue;
          const base = t.github || fx.github || "";
          const lineno = t.lineno;
          if (!base) return "";
          if (lineno && Number.isFinite(lineno)) return `${base}#L${lineno}`;
          return base;
        }
      }
      return "";
    }

    function buildCompositeInfoMap(taskAttributesJson) {
      // Source of truth: same JSON used by the Composite Tasks page.
      const map = new Map();
      const tasks = taskAttributesJson && Array.isArray(taskAttributesJson.tasks) ? taskAttributesJson.tasks : [];
      for (const t of tasks) {
        if (!t || !t.name) continue;
        if (t.activity === "Atomic") continue;
        map.set(t.name, { activity: t.activity || "", description: t.description || "" });
      }
      return map;
    }

    function populateAtomicSeen(table, atomicEpisodeLengthMap) {
      const tbody = table?.querySelector("tbody");
      if (!tbody) return;
      tbody.innerHTML = "";

      const atomicAttrs = window.ROBOCASA_ATOMIC_TASK_ATTRIBUTES || {};

      for (const name of ATOMIC_SEEN_TASKS) {
        const tr = document.createElement("tr");

        const tdTask = document.createElement("td");
        const href = getAtomicClassHref(name);
        const code = document.createElement("code");
        code.textContent = name;
        if (href) {
          const a = document.createElement("a");
          a.href = href;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.appendChild(code);
          tdTask.appendChild(a);
        } else {
          tdTask.appendChild(code);
        }

        const tdDesc = document.createElement("td");
        const rawDesc = atomicAttrs?.[name]?.description || "";
        tdDesc.innerHTML = formatAtomicDescription(rawDesc);

        const tdH = document.createElement("td");
        const horizon = atomicEpisodeLengthMap?.get(name);
        tdH.textContent = Number.isFinite(horizon) ? `${Math.round(horizon)}s` : "—";

        const tdV = document.createElement("td");
        const v = getVideoForTask(name);
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "rc-atomic-video-btn";
        btn.textContent = "Demo";
        if (v) {
          btn.addEventListener("click", () => {
            const desc = (tdDesc.innerHTML || tdDesc.textContent || "").trim();
            openVideo(v.sources, name, desc);
          });
        } else {
          btn.disabled = true;
        }
        tdV.appendChild(btn);

        tr.appendChild(tdTask);
        tr.appendChild(tdDesc);
        tr.appendChild(tdH);
        tr.appendChild(tdV);
        tbody.appendChild(tr);
      }
    }

    function populateCompositeTable(table, taskNames, infoMap) {
      const tbody = table?.querySelector("tbody");
      if (!tbody) return;
      tbody.innerHTML = "";

      for (const name of taskNames) {
        const tr = document.createElement("tr");

        const tdTask = document.createElement("td");
        const info = infoMap.get(name) || {};
        const activity = info.activity || "";
        const href = activity ? sourceUrlForTask(activity, name) : "";
        const code = document.createElement("code");
        code.textContent = name;
        if (href) {
          const a = document.createElement("a");
          a.href = href;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.appendChild(code);
          tdTask.appendChild(a);
        } else {
          tdTask.appendChild(code);
        }

        const tdDesc = document.createElement("td");
        tdDesc.innerHTML = formatDescriptionToHtml(info.description || "");

        tr.appendChild(tdTask);
        tr.appendChild(tdDesc);
        tbody.appendChild(tr);
      }
    }

    (async () => {
      const attrsMap = getTaskAttributesMap() || new Map();
      const episodeLengthMap = getEpisodeLengthMap();
      const atomicEpisodeLengthMap = getAtomicEpisodeLengthMap();
      const taskAttributesJson = await loadTaskAttributesJson();
      const compInfo = buildCompositeInfoMap(taskAttributesJson);

      const atomicTable = document.getElementById("rc-datasets-atomic-seen");
      if (atomicTable) populateAtomicSeen(atomicTable, atomicEpisodeLengthMap);

      const compSeenTable = document.getElementById("rc-datasets-composite-seen");
      if (compSeenTable) {
        populateCompositeTable(compSeenTable, COMPOSITE_SEEN_TASKS, compInfo);
        addAttributesToTable(compSeenTable, attrsMap, episodeLengthMap);
      }

      const compUnseenTable = document.getElementById("rc-datasets-composite-unseen");
      if (compUnseenTable) {
        populateCompositeTable(compUnseenTable, COMPOSITE_UNSEEN_TASKS, compInfo);
        addAttributesToTable(compUnseenTable, attrsMap, episodeLengthMap);
      }
    })();
  });
})();

