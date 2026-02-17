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

  function isCompositeTasksPage() {
    const path = window.location.pathname.replace(/\\/g, "/");
    return path.endsWith("/tasks/composite_tasks.html") || path.endsWith("/tasks/composite_tasks/");
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

  function getHeadingTitle(headingEl) {
    // Sphinx adds a permalink anchor like: <a class="headerlink"...>#</a>
    // We want the visible title without that "#".
    const clone = headingEl.cloneNode(true);
    const headerlink = clone.querySelector("a.headerlink");
    if (headerlink) headerlink.remove();
    return (clone.textContent || "").trim();
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
    ["baking cookies/cakes", "cooking"],
    ["baking cookies and cakes", "cooking"],
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
      [n, n.replace("baking cookies cakes", "baking cookies/cakes")],
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
      const ths = Array.from(theadRow.querySelectorAll("th")).map((th) => (th.textContent || "").trim());
      const mobileHeader = "Navigation";
      const horizonHeader = "Horizon";
      const hasMobile = ths.includes(mobileHeader);
      const hasSubtasks = ths.includes("Subtasks");
      const legacyLengthIdx = ths.indexOf("Length");
      const horizonIdx = ths.indexOf(horizonHeader);
      const hasEpisodeLength = legacyLengthIdx >= 0 || horizonIdx >= 0;
      const hasVideo = ths.includes("Video");

      // Backwards compat: if a pre-existing column is labeled "Length", rename it to "Horizon"
      if (legacyLengthIdx >= 0 && horizonIdx < 0) {
        const thEls = Array.from(theadRow.querySelectorAll("th"));
        const legacyTh = thEls[legacyLengthIdx];
        if (legacyTh) legacyTh.textContent = horizonHeader;
      }

      const descIdx = ths.indexOf("Description");
      const insertAfter = descIdx >= 0 ? theadRow.children[descIdx] : null;
      let refNode = insertAfter ? insertAfter.nextSibling : null;

      // Insert in desired order right after Description
      if (!hasMobile) {
        const th = document.createElement("th");
        th.textContent = mobileHeader;
        th.style.textAlign = "left";
        theadRow.insertBefore(th, refNode);
        refNode = th.nextSibling;
      }
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
    if (taskIdx < 0 || classIdx < 0) return;

    const rows = Array.from(table.querySelectorAll("tbody tr"));
    for (const tr of rows) {
      const tds = Array.from(tr.children);
      const taskTd = tds[taskIdx];
      const classTd = tds[classIdx];
      if (!taskTd || !classTd) continue;

      const srcLink = classTd.querySelector("a[href]");
      if (srcLink && !taskTd.querySelector("a[href]")) {
        const a = document.createElement("a");
        a.href = srcLink.href;
        a.target = "_blank";
        a.rel = "noopener noreferrer";

        const code = taskTd.querySelector("code");
        if (code) {
          a.appendChild(code);
          taskTd.textContent = "";
          taskTd.appendChild(a);
        } else {
          a.textContent = (taskTd.textContent || "").trim();
          taskTd.textContent = "";
          taskTd.appendChild(a);
        }
      }
    }

    // Remove header + cells (delete column)
    theadRow.children[classIdx]?.remove();
    for (const tr of rows) {
      tr.children[classIdx]?.remove();
    }
  }

  function taskNameFromRow(tr) {
    // First cell contains code element with the task name
    const firstCell = tr.querySelector("td");
    if (!firstCell) return null;
    const code = firstCell.querySelector("code");
    const name = (code ? code.textContent : firstCell.textContent) || "";
    return name.trim() || null;
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
        const mobileTd = tr.querySelector("td.rc-mobile");
        const subtasksTd = tr.querySelector("td.rc-subtasks");
        const episodeLengthTd = tr.querySelector("td.rc-episode-length");
        const mobile = (mobileTd ? mobileTd.textContent : "").trim();
        const subtasks = Number.parseInt((subtasksTd ? subtasksTd.textContent : "").trim(), 10);
        const lengthText = (episodeLengthTd ? episodeLengthTd.textContent : "").trim();
        // Extract numeric value from "32s" or "—"
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
          subtasks: Number.isFinite(subtasks) ? subtasks : null,
          length: length,
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
    if (current && optById.has(current)) selectEl.value = current;
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

    // Optional user-configured base URL (takes precedence over defaults)
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
    title.textContent = "";

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
    instruction.appendChild(document.createTextNode(" "));
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

    let loadTimeout = null;
    let hasLoadedMetadata = false;

    function doClose() {
      overlay.hidden = true;
      document.body.classList.remove("rc-modal-open");
      try {
        video.pause();
      } catch {}
      if (overlay._rcLastPlayPromise) {
        overlay._rcLastPlayPromise.catch(function () {});
        overlay._rcLastPlayPromise = null;
      }
      if (overlay._rcLoadTimeout != null) {
        clearTimeout(overlay._rcLoadTimeout);
        overlay._rcLoadTimeout = null;
      }
      hasLoadedMetadata = false;
      instruction.hidden = true;
      instructionLabel.textContent = "";
      instructionText.textContent = "";
      error.hidden = true;
      error.textContent = "";
      title.textContent = "";
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
      title.textContent = taskName ? `${taskName}` : "";

      const srcsRaw = Array.isArray(sources) ? sources : [sources];
      const srcs = srcsRaw.map((s) => String(s || "").trim()).filter(Boolean);
      let attempt = 0;
      let loadTimeout = null;
      let hasLoadedMetadata = false;

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
        instructionLabel.textContent = tName ? `${tName}:` : "";
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
      if (tr.querySelector("td.rc-mobile")) continue;

      const name = taskNameFromRow(tr);
      const attrs = name ? attrsMap.get(name) : null;

      const mobileTd = document.createElement("td");
      mobileTd.className = "rc-mobile";
      mobileTd.style.textAlign = "left";
      const mobileVal = attrs?.mobile;
      mobileTd.textContent = mobileVal === "Yes" || mobileVal === "No" ? mobileVal : "—";

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
          const desc = descTd ? (descTd.innerHTML || descTd.textContent || "").trim() : "";
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

      // Insert after Description column (2nd cell)
      const tds = tr.querySelectorAll("td");
      const descTd = tds.length >= 2 ? tds[1] : null;
      if (descTd && descTd.parentNode) {
        tr.insertBefore(mobileTd, descTd.nextSibling);
        tr.insertBefore(subtasksTd, mobileTd.nextSibling);
        tr.insertBefore(episodeLengthTd, subtasksTd.nextSibling);
      } else {
        tr.appendChild(mobileTd);
        tr.appendChild(subtasksTd);
        tr.appendChild(episodeLengthTd);
      }

      // Always append Video at the far right
      tr.appendChild(videoTd);
    }
  }

  onReady(() => {
    if (!isCompositeTasksPage()) return;

    // Page-scoped styling hooks
    document.body.classList.add("rc-composite-tasks");

    // Sphinx Book Theme main content wrapper
    const content =
      document.querySelector("main .bd-article") ||
      document.querySelector("main") ||
      document.body;
    if (!content) return;

    // Activities are sections nested under the main "Composite Tasks" section.
    const rootSection = content.querySelector("section#composite-tasks");
    if (!rootSection) return;

    const activitySections = Array.from(rootSection.querySelectorAll(":scope > section[id]"));
    if (activitySections.length === 0) return;

    const activities = [];

    const attrsMap = getTaskAttributesMap();
    const episodeLengthMap = getEpisodeLengthMap();

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

    // Convert "Class File" column into a link on the Task name, then remove the column
    for (const d of Array.from(content.querySelectorAll("details.rc-activity"))) {
      const table = d.querySelector("table");
      if (table) linkifyTaskAndRemoveClassFile(table);
    }

    // Replace templated variable braces in descriptions:
    // "{left/right}" -> "[<em>left/right</em>]"
    function formatCompositeDescriptionInPlace(rootEl) {
      if (!rootEl) return;
      const textNodes = [];
      const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) textNodes.push(n);

      const re = /\{([^}]+)\}/g;
      for (const node of textNodes) {
        const s = node.nodeValue || "";
        if (!s.includes("{")) continue;
        re.lastIndex = 0;

        const frag = document.createDocumentFragment();
        let last = 0;
        let m;
        while ((m = re.exec(s))) {
          const start = m.index;
          const end = start + m[0].length;
          if (start > last) frag.appendChild(document.createTextNode(s.slice(last, start)));
          frag.appendChild(document.createTextNode("["));
          const em = document.createElement("em");
          em.textContent = m[1];
          frag.appendChild(em);
          frag.appendChild(document.createTextNode("]"));
          last = end;
        }
        if (last === 0) continue; // no matches
        if (last < s.length) frag.appendChild(document.createTextNode(s.slice(last)));
        node.parentNode.replaceChild(frag, node);
      }
    }

    for (const d of Array.from(content.querySelectorAll("details.rc-activity"))) {
      const table = d.querySelector("table");
      if (!table) continue;
      const ths = Array.from(table.querySelectorAll("thead tr th")).map((th) => (th.textContent || "").trim());
      const descIdx = ths.indexOf("Description");
      const idx = descIdx >= 0 ? descIdx : 1;
      for (const tr of Array.from(table.querySelectorAll("tbody tr"))) {
        const td = tr.children?.[idx];
        if (td) formatCompositeDescriptionInPlace(td);
      }
    }

    // After accordions exist, enrich tables with attributes if available
    if (attrsMap || episodeLengthMap) {
      for (const d of Array.from(content.querySelectorAll("details.rc-activity"))) {
        const table = d.querySelector("table");
        if (table) addAttributesToTable(table, attrsMap, episodeLengthMap);
      }
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

    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
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

    const navWrap = document.createElement("div");
    navWrap.className = "rc-filter";
    const navLabel = document.createElement("label");
    navLabel.textContent = "Navigation:";
    navLabel.setAttribute("for", "rc-filter-nav");
    const navSelect = document.createElement("select");
    navSelect.id = "rc-filter-nav";
    for (const [val, text] of [
      ["", "Any"],
      ["Yes", "Yes"],
      ["No", "No"],
    ]) {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = text;
      navSelect.appendChild(opt);
    }
    navWrap.appendChild(navLabel);
    navWrap.appendChild(navSelect);

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

    filtersWrap.appendChild(navWrap);
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
    const lengthMeta = new Map(); // key -> { min, max, inclusiveMax, label, countEl }

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
      countSpan.textContent = "(0)";

      const rightWrap = document.createElement("span");
      rightWrap.className = "rc-length-item-right";
      rightWrap.appendChild(labelSpan);

      row.appendChild(cb);
      row.appendChild(rightWrap);
      row.appendChild(countSpan);
      lengthMenu.appendChild(row);

      lengthChecks.set(it.key, cb);
      lengthMeta.set(it.key, { min: it.min, max: it.max, inclusiveMax: it.inclusiveMax, label: it.label, countEl: countSpan });

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
      item.appendChild(cb);
      item.appendChild(txt);
      catMenu.appendChild(item);
      categoryChecks.set(key, cb);
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

    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "rc-reset-filters-btn";
    resetBtn.textContent = "Reset";
    resetBtn.setAttribute("aria-label", "Reset filters");
    resetBtn.addEventListener("click", () => {
      // Reset filter inputs to defaults
      navSelect.value = "";
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
      applyFilters();
    });
    filtersWrap.appendChild(resetBtn);

    // Build task search + suggestions
    const taskIndex = buildTaskIndex(content);

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

    function updateLengthDropdownCounts() {
      try {
        // Get current filters excluding length
        const nav = (navSelect.value || "").trim();
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
            if (nav && item.mobile !== nav) continue;
            if (item.subtasks == null) continue;
            if (item.subtasks < effectiveSubtasks.minV || item.subtasks > effectiveSubtasks.maxV) continue;

            if (item.length == null) continue;
            if (!lengthMatchesInterval(item.length, it)) continue;
            count += 1;
          }

          const meta = lengthMeta.get(it.key);
          if (meta?.countEl) meta.countEl.textContent = `(${count})`;
        }
      } catch (e) {
        console.error("Error updating length dropdown counts:", e);
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
      const nav = (navSelect.value || "").trim();
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

      const cats = new Set();
      for (const [cat, cb] of categoryChecks.entries()) {
        if (cb.checked) cats.add(cat);
      }
      return {
        nav,
        minV: effective.minV,
        maxV: effective.maxV,
        hasError: !inRange,
        lengthIntervals: lengthIntervalsActive,
        lengthHasError: false, // Dropdown always has valid values
        cats,
      };
    }

    function passesFilters(item, f) {
      if (f.cats && f.cats.size > 0) {
        if (!item.metaCategory || !f.cats.has(item.metaCategory)) return false;
      } else if (f.cats && f.cats.size === 0) {
        return false;
      }
      if (f.nav && item.mobile !== f.nav) return false;
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
    navSelect.addEventListener("change", applyFilters);

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
    applyFilters();
    applyShowAllState();

    // Put selector under the page title (h1) if present
    const titleH1 = content.querySelector("h1");
    if (titleH1 && titleH1.parentNode) {
      titleH1.insertAdjacentElement("afterend", selectorWrap);
    } else {
      content.insertAdjacentElement("afterbegin", selectorWrap);
    }
  });
})();

