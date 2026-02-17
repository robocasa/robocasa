# Overview of Datasets

RoboCasa offers over 2,200 hours of demonstration data, comprising human teleoperation data and synthetic data.
Broadly, the data is split into **pretraining datasets** and **target datasets**.
The pretraining datasets feature 300 diverse tasks across 2,500 pretraining kitchens, while the target datasets feature 50 target tasks across a distinct set of 10 heldout target kitchens.

<table class="docutils rc-datasets-summary">
  <caption>Dataset statistics across pretraining and target settings.</caption>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Num Tasks</th>
      <th>Num Scenes</th>
      <th>Demos per Task</th>
      <th>Dataset Size (hrs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pretraining (Human)</td>
      <td>300</td>
      <td>2500</td>
      <td>100</td>
      <td>482</td>
    </tr>
    <tr>
      <td>Pretraining (MimicGen)</td>
      <td>60</td>
      <td>2500</td>
      <td>10,000</td>
      <td>1615</td>
    </tr>
    <tr>
      <td>Target (Human)</td>
      <td>50</td>
      <td>10</td>
      <td>500</td>
      <td>193</td>
    </tr>
  </tbody>
</table>

We provide a detailed overview of the pretraining and target datasets below.

-------
## Pretraining Datasets
RoboCasa offers ~2,000 hours of pretraining demonstration data.
The pretraining datasets feature 300 diverse tasks across 2500 pretraining kitchens. We feature both human and sythentic datasets:

### Human Datasets
482 hours of data collected via teleoperation. The data spans 300 tasks (65 atomic tasks and 235 composite tasks), with 100 demonstrations per task.
Go to the [Atomic Tasks](../tasks/atomic_tasks.html) and [Composite Tasks](../tasks/atomic_tasks.html) pages to see the list of supported tasks.

### Synthetic Datasets
1615 hours of data generated via [MimicGen](https://mimicgen.github.io/). The data spans 60 atomic tasks, with ~10k demonstrations per task.
Go to the [Atomic Tasks](../tasks/atomic_tasks.html) page to see the list of supported tasks.

-------
## Target Datasets

In addition to pretraining data, RoboCasa offers over 193 hours of high-quality demonstration data for target tasks collected via teleoperation.
The target datasets feature 50 diverse tasks across 10 distinct target kitchen scenes.
Note that these target scenes are distinct from the pretraining scenes represented in the pretraining datasets.
For each task, we provide **500 human demonstrations** collected via teleoperation.

We split these datasets into three groups:
* **Atomic-Seen** (18 tasks): 18 atomic tasks, with all tasks also represented in pretraining datasets.
* **Composite-Seen** (16 tasks): 16 composite tasks, with all tasks also represented in pretraining datasets.
* **Composite-Unseen** (16 tasks): 16 composite tasks, only seen in target datasets and not in pretraining datasets.


### Atomic-Seen Tasks

<!-- FILTER CONTROLS -->
<!-- <div style="font-family: sans-serif; margin-bottom:15px; display:flex; gap:20px; align-items:center;">
  <div>
    <label style="font-weight:600;">Category:</label><br>
    <select id="filterCategory" onchange="applyFiltersFallback(this.value, document.getElementById('filterDifficulty')?.value)" style="padding:6px; border-radius:6px; border:1px solid #ccc;">
      <option value="">All</option>
      <option value="Display">Display</option>
      <option value="Audio">Audio</option>
      <option value="Network">Network</option>
    </select>
  </div>

  <div>
    <label style="font-weight:600;">Difficulty:</label><br>
    <select id="filterDifficulty" onchange="applyFiltersFallback(document.getElementById('filterCategory')?.value, this.value)" style="padding:6px; border-radius:6px; border:1px solid #ccc;">
      <option value="">All</option>
      <option value="Easy">Easy</option>
      <option value="Medium">Medium</option>
      <option value="Hard">Hard</option>
    </select>
  </div>
</div> -->

<div class="rc-activity-body">
<table id="rc-datasets-atomic-seen" class="docutils rc-datasets-table">
  <thead>
    <tr>
      <th>Task</th>
      <th>Description</th>
      <th>Horizon</th>
      <th>Video</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
</div>

<!-- SCRIPT: debug-friendly, attaches event listeners after DOM ready -->
<!-- <script>
(function () {
  function log(...args){ try { console.log('[filter-table]', ...args); } catch(e){} }

  function applyFilters() {
    const category = (document.getElementById("filterCategory")?.value || "").toLowerCase();
    const difficulty = (document.getElementById("filterDifficulty")?.value || "").toLowerCase();
    const rows = document.querySelectorAll("#settingsTable tbody tr");
    rows.forEach(row => {
      const rowCategory = (row.cells[1]?.innerText || "").toLowerCase();
      const rowDifficulty = (row.cells[2]?.innerText || "").toLowerCase();
      const matchCategory = !category || rowCategory === category;
      const matchDifficulty = !difficulty || rowDifficulty === difficulty;
      row.style.display = (matchCategory && matchDifficulty) ? "" : "none";
    });
  }

  // Fallback so inline onchange can call this if event listeners aren't allowed
  window.applyFiltersFallback = function(catVal, diffVal){
    try {
      if (typeof catVal === 'undefined') catVal = document.getElementById("filterCategory")?.value || "";
      if (typeof diffVal === 'undefined') diffVal = document.getElementById("filterDifficulty")?.value || "";
      if (document.getElementById("filterCategory")) document.getElementById("filterCategory").value = catVal;
      if (document.getElementById("filterDifficulty")) document.getElementById("filterDifficulty").value = diffVal;
      applyFilters();
    } catch(e){ log('fallback error', e); }
  };

  document.addEventListener('DOMContentLoaded', function(){
    log('DOM ready — attaching listeners');
    try {
      const c = document.getElementById("filterCategory");
      const d = document.getElementById("filterDifficulty");
      if (!c || !d) { log('filter elements not found'); return; }

      // attach event listeners (if allowed)
      c.addEventListener('change', applyFilters);
      d.addEventListener('change', applyFilters);

      // initial run
      applyFilters();
      log('listeners attached and initial filter run completed');
    } catch (err) {
      log('error attaching listeners', err);
    }
  });

  // extra safety: run again a little later in case DOMContentLoaded already fired but platform modified DOM
  setTimeout(function(){ if (document.readyState === 'complete' || document.readyState === 'interactive'){ try{ applyFilters(); }catch(e){} } }, 800);
})();
</script> -->

### Composite-Seen Tasks
<div class="rc-activity-body">
<table id="rc-datasets-composite-seen" class="docutils rc-datasets-table">
  <thead>
    <tr>
      <th>Task</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
</div>



### Composite-Unseen Tasks
<div class="rc-activity-body">
<table id="rc-datasets-composite-unseen" class="docutils rc-datasets-table">
  <thead>
    <tr>
      <th>Task</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
</div>