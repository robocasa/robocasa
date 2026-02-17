# Fixtures

<p class="fixtures-intro">Each kitchen scene contains a wide variety of interactable fixtures. RoboCasa includes a total of 456 fixtures spanning 16 categories.</p>

<div class="fixture-filter-container">
  <label for="style-filter-input" class="fixture-filter-label">Filter by Style:</label>
  <input
    type="text"
    id="style-filter-input"
    class="fixture-filter-input"
    inputmode="numeric"
    pattern="\\d{1,2}"
    maxlength="2"
    placeholder="Enter style (1-60)"
    aria-label="Filter fixtures by style number"
  >
  <button type="button" id="style-filter-clear" class="fixture-filter-clear" onclick="clearStyleFilter()" style="display: none;">Clear</button>
</div>

<script>
  // Fixture images: load from R2 CDN (same structure as docs/fixtures: <category>/<id>.webp).
  const FIXTURE_IMAGE_BASE = "https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev";
  // Rewrite any static path (relative or absolute) to R2 base (covers Sphinx-built absolute paths).
  function toR2(path) {
    if (!path) return path;
    path = path.replace(/^\/_static\/fixtures\//, "").replace(/^\/_static\//, "");
    path = path.replace(/^\.\.\/_static\/fixtures\//, "").replace(/^\.\.\/_static\//, "");
    return path ? FIXTURE_IMAGE_BASE + "/" + path : FIXTURE_IMAGE_BASE;
  }
  (function () {
    function fixFixtureStaticPaths() {
      const viewers = Array.from(document.querySelectorAll(".fixture-viewer"));
      for (const v of viewers) {
        let base = v.getAttribute("data-base") || "";
        if (base.includes("_static")) {
          base = toR2(base);
          v.setAttribute("data-base", base);
        }
        const img = v.querySelector("img.fixture-preview-image");
        if (img) {
          let src = img.getAttribute("src") || "";
          if (src.includes("_static")) img.setAttribute("src", toR2(src));
        }
      }
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fixFixtureStaticPaths);
    else fixFixtureStaticPaths();
  })();
</script>

<!--
<div class="fixtures-table-wrap">
  <table class="rc-benchmark-table rc-fixtures-table">
  <thead>
    <tr>
      <th>Category</th>
      <th>Unique models</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Blender</td><td>22</td></tr>
    <tr><td>Coffee machine</td><td>48</td></tr>
    <tr><td>Dishwasher</td><td>25</td></tr>
    <tr><td>Electric kettle</td><td>25</td></tr>
    <tr><td>Fridge</td><td>50</td></tr>
    <tr><td>Microwave</td><td>50</td></tr>
    <tr><td>Oven</td><td>21</td></tr>
    <tr><td>Sink</td><td>49</td></tr>
    <tr><td>Stand mixer</td><td>25</td></tr>
    <tr><td>Stove</td><td>50</td></tr>
    <tr><td>Toaster</td><td>44</td></tr>
    <tr><td>Toaster oven</td><td>47</td></tr>
    <tr><td><strong>Total</strong></td><td><strong>456</strong></td></tr>
  </tbody>
  </table>
</div>
-->

<style>
/* Page width: use more of the available horizontal space (fixtures page only) */
.bd-main .bd-content .bd-article-container {
  max-width: none;
}

/* Intro */
.fixtures-intro {
  margin: 0.2rem 0 0.9rem 0;
}

/* Match Atomic/Composite table styling (numeric alignment) */
table.rc-fixtures-table th:nth-child(2),
table.rc-fixtures-table td:nth-child(2) {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

/* Table below intro, left-aligned */
.fixtures-table-wrap {
  width: 100%;
  max-width: 38rem;
  margin: 0 0 1.2rem 0; /* left-aligned by default */
}

.fixtures-table-wrap table.rc-fixtures-table {
  font-size: 0.92em;
}

.fixtures-table-wrap table.rc-fixtures-table th,
.fixtures-table-wrap table.rc-fixtures-table td {
  padding: 0.38rem 0.48rem;
}

.fixture-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.fixture-item {
  text-align: center;
  cursor: pointer;
}
.fixture-item img {
  width: 100%;
  border-radius: 6px;
  border: 1px solid rgba(128,128,128,0.3);
  transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.fixture-item img:hover {
  transform: scale(1.03);
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.fixture-item .label {
  font-size: 12px;
  margin-top: 4px;
  color: inherit;
  opacity: 0.8;
}

/* Inline (non-modal) "Style ..." label under fixture preview */
.fixture-item .fixture-preview-label {
  font-size: 14px;
  margin-top: 14px;
}

/* Highlight matching style number when filtering */
.style-highlight {
  background-color: #ffeb3b;
  color: #000;
  padding: 2px 1px;
  border-radius: 2px;
  font-weight: 600;
}

/* Modal/Lightbox styles */
.fixture-modal {
  display: none;
  position: fixed;
  z-index: 9999;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0,0,0,0.9);
  overflow: hidden;
}
.fixture-modal.active {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.modal-content {
  position: relative;
  max-width: min(90%, 990px);
  max-height: 82vh;
  text-align: center;
  padding: 0;
  margin: 0;
  background: transparent;
  border: none;
  outline: none;
  line-height: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.modal-content img {
  max-width: 100%;
  max-height: 82vh;
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 0;
  box-shadow: none;
  margin: 0;
  padding: 0;
  display: block;
  border: none;
  outline: none;
  background: transparent;
  vertical-align: bottom;
  image-rendering: auto;
  flex-shrink: 0;
}
.modal-label {
  color: white;
  font-size: 18px;
  margin-top: 20px; /* more gap from image */
  font-weight: 500;
}
.modal-close {
  position: absolute;
  top: 15px;
  right: 25px;
  color: white;
  font-size: 40px;
  font-weight: bold;
  cursor: pointer;
  z-index: 10000;
  transition: color 0.2s;
}
.modal-close:hover {
  color: #3498db;
}
.modal-nav {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  color: white;
  font-size: 50px;
  font-weight: bold;
  cursor: pointer;
  padding: 20px;
  user-select: none;
  transition: color 0.2s;
  z-index: 10000;
}
.modal-nav:hover {
  color: #3498db;
}
.modal-prev {
  left: 20px;
}
.modal-next {
  right: 20px;
}
.modal-slider-container {
  width: 80%;
  max-width: 600px;
  margin-top: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
}

.modal-arrow {
  background: transparent;
  border: none;
  color: white;
  font-size: 28px;
  line-height: 1;
  cursor: pointer;
  padding: 6px 10px;
  user-select: none;
}
.modal-arrow:hover {
  color: #3498db;
}
.modal-arrow:focus-visible {
  outline: 2px solid #3498db;
  outline-offset: 2px;
}
.modal-arrow[disabled] {
  opacity: 0.35;
  cursor: default;
}
.modal-slider {
  flex: 1;
  -webkit-appearance: none;
  appearance: none;
  height: 8px;
  background: rgba(255,255,255,0.3);
  border-radius: 4px;
  outline: none;
}
.modal-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  background: #3498db;
  border-radius: 50%;
  cursor: pointer;
  transition: background 0.2s;
}
.modal-slider::-webkit-slider-thumb:hover {
  background: #2980b9;
}
.modal-slider::-moz-range-thumb {
  width: 20px;
  height: 20px;
  background: #3498db;
  border-radius: 50%;
  cursor: pointer;
  border: none;
}
.modal-counter {
  color: white;
  font-size: 14px;
  min-width: 60px;
  text-align: center;
}

/* Collapsible section styles */
.fixture-section {
  margin-bottom: 24px;
}
.fixture-section-header {
  display: flex;
  align-items: center;
  cursor: pointer;
  user-select: none;
  padding: 8px 0;
  border-bottom: 1px solid rgba(128,128,128,0.3);
  margin-bottom: 8px;
}
.fixture-section-header:hover {
  opacity: 0.8;
}
.fixture-section-header h2 {
  margin: 0;
  font-size: 1.5em;
}
.fixture-toggle {
  display: inline-block;
  width: 20px;
  height: 20px;
  margin-right: 10px;
  text-align: center;
  font-size: 14px;
  font-weight: bold;
  color: #3498db;
  transition: transform 0.2s;
}
.fixture-section.collapsed .fixture-toggle {
  transform: rotate(-90deg);
}
.fixture-section-content {
  overflow: hidden;
  transition: max-height 0.3s ease-out, opacity 0.3s ease-out;
  max-height: 5000px;
  opacity: 1;
}
.fixture-section.collapsed .fixture-section-content {
  max-height: 0;
  opacity: 0;
}

.fixture-viewer {
  margin-top: 12px;
  max-width: 520px;
}

.fixture-inline-slider-container {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 10px;
}

/* Style filter */
.fixture-filter-container {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 16px 0 20px 0;
  padding: 12px;
  background: var(--pst-color-surface, rgba(127, 127, 127, 0.04));
  border: 1px solid var(--pst-color-border, rgba(128, 128, 128, 0.3));
  border-radius: 8px;
}

.fixture-filter-label {
  font-weight: 600;
  color: inherit;
  white-space: nowrap;
}

.fixture-filter-input {
  flex: 1;
  max-width: 200px;
  padding: 8px 12px;
  border: 1px solid var(--pst-color-border, rgba(128, 128, 128, 0.3));
  border-radius: 6px;
  font-size: 14px;
  background: var(--pst-color-background, white);
  color: inherit;
}

.fixture-filter-input:focus {
  outline: 2px solid #3498db;
  outline-offset: 2px;
  border-color: #3498db;
}

.fixture-filter-input.rc-invalid {
  border-color: #b42318 !important;
  box-shadow: 0 0 0 3px rgba(180, 35, 24, 0.15);
}

.fixture-filter-clear {
  padding: 8px 16px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: background 0.2s;
}

.fixture-filter-clear:hover {
  background: #2980b9;
}

.fixture-card.hidden-by-filter {
  display: none;
}

.fixture-inline-counter {
  font-size: 14px;
  min-width: 70px;
  text-align: center;
  opacity: 0.9;
}

.fixture-inline-slider {
  flex: 1;
  -webkit-appearance: none;
  appearance: none;
  height: 8px;
  background: rgba(128, 128, 128, 0.22);
  border-radius: 4px;
  outline: none;
}

.fixture-inline-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  background: #3498db;
  border-radius: 50%;
  cursor: pointer;
}

.fixture-inline-slider::-moz-range-thumb {
  width: 18px;
  height: 18px;
  background: #3498db;
  border-radius: 50%;
  cursor: pointer;
  border: none;
}

/* Fixture cards layout: show ~2 per row on desktop */
.fixtures-viewers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 24px;
  align-items: start;
  margin-top: 16px;
}

@media (max-width: 1000px) {
  .fixtures-viewers-grid {
    grid-template-columns: 1fr;
  }
}

.fixture-card {
  border: 1px solid var(--pst-color-border, rgba(128, 128, 128, 0.3));
  border-radius: 12px;
  /* Match docs styling (as in deployed fixtures page) */
  padding: 14px 14px 16px;
  background: var(--pst-color-surface, rgba(127, 127, 127, 0.04));
}

.fixture-card-title {
  margin: 0 0 10px 0;
  font-size: 1.6rem;
  font-weight: 800;
  line-height: 1.15;
  text-align: center;
}

.fixture-viewer {
  margin-top: 0;
  max-width: none;
}

.fixture-preview-image {
  display: block;
  width: 100%;
  height: auto;
  max-height: none; /* avoid letterboxing whitespace */
  object-fit: initial;
  background: transparent;
  vertical-align: bottom;
  line-height: 0;
}

/* Normalize image appearance in dark mode */
@media (prefers-color-scheme: dark) {
  .modal-content img,
  .fixture-preview-image {
    /* Avoid black-crush: don't alter the image pixels with CSS filters.
       Use a subtle background matte to reduce perceived "black crush"
       from the surrounding dark UI without visible borders. */
    filter: none;
    background-color: rgba(255, 255, 255, 0.06);
    border: none;
    box-shadow: none;
    border-radius: 0;
  }
  .modal-content {
    background: linear-gradient(to bottom, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.01));
  }
  .fixture-card {
    background: linear-gradient(to bottom, rgba(255, 255, 255, 0.02), rgba(127, 127, 127, 0.04));
  }
}

/* PyData theme dark-mode uses html[data-theme="dark"] (not just prefers-color-scheme).
   Force-disable any theme image filters that can cause perceived "black crush". */
html[data-theme="dark"] .modal-content img,
html[data-theme="dark"] .fixture-preview-image {
  filter: none !important;
  opacity: 1 !important;
  background-color: rgba(255, 255, 255, 0.06);
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}
</style>

<!-- Modal HTML -->
<div id="fixtureModal" class="fixture-modal">
  <span class="modal-close" onclick="closeFixtureModal()">&times;</span>
  <div class="modal-content">
    <img id="fixtureModalImage" src="" alt="Fixture">
    <div id="fixtureModalLabel" class="modal-label"></div>
  </div>
  <div class="modal-slider-container">
    <button type="button" class="modal-arrow" id="fixtureModalPrev" aria-label="Previous style" onclick="fixtureModalStep(-1)">&#8249;</button>
    <span class="modal-counter" id="fixtureModalCounter">1 / 1</span>
    <input type="range" class="modal-slider" id="fixtureModalSlider" min="1" value="1">
    <button type="button" class="modal-arrow" id="fixtureModalNext" aria-label="Next style" onclick="fixtureModalStep(1)">&#8250;</button>
  </div>
</div>

<div class="fixtures-viewers-grid">
  <div class="fixture-card">
    <h2 class="fixture-card-title">Blender</h2>
    <div class="fixture-viewer" data-name="Blender" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/blender" data-ids="1,2,3,4,5,6,7,8,11,12,13,14,15,56,18,19,21,22,23,25,30,44" data-style-groups="1;2;3;4|9;5;6;7;8|10;11|28|48|51;12|35|39;13|29|31;14|16|36;15|20|32|49;17|27|53|56;18|24|52;19|26|40|54|58;21|37|60;22|41|42|59;23|38|43|46|47;25|45|50;30|33|34;44|55|57">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/blender/1.webp" alt="Blender">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 22</span>
        <input type="range" class="fixture-inline-slider" min="1" max="22" value="1" aria-label="Blender index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Coffee Machine</h2>
    <div class="fixture-viewer" data-name="Coffee Machine" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/coffee_machine" data-ids="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,32,20,21,22,23,24,25,27,28,29,30,31,34,35,36,37,38,42,43,44,45,47,48,50,51,53,56,58,59,60" data-style-groups="1;2;3;4;5;6;7;8;9;10;11;12;13;14;15|33;16|41;17;18|57;19|32;20;21|49;22|39;23|54;24|46|52;25|26;27;28;29;30|55;31;34|40;35;36;37;38;42;43;44;45;47;48;50;51;53;56;58;59;60">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/coffee_machine/1.webp" alt="Coffee Machine">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 48</span>
        <input type="range" class="fixture-inline-slider" min="1" max="48" value="1" aria-label="Coffee Machine index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Electric Kettle</h2>
    <div class="fixture-viewer" data-name="Electric Kettle" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/electric_kettle" data-ids="1,2,3,4,9,11,12,13,14,15,16,20,21,22,23,24,25,26,31,34,38,39,41,53,55" data-style-groups="1|6;2|7;3|8;4|5;9|10;11|17|43;12|18|47|59;13|29|35|50;14|36|40|44|60;15|19;16|45;20;21|37|49;22|28|56;23|51;24|48;25|27|54|57;26|30|33|42;31|32;34;38|52;39|46;41;53;55|58">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/electric_kettle/1.webp" alt="Electric Kettle">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 25</span>
        <input type="range" class="fixture-inline-slider" min="1" max="25" value="1" aria-label="Electric Kettle index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Dishwasher</h2>
    <div class="fixture-viewer" data-name="Dishwasher" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/dishwasher" data-ids="36,40,32,15,4,30,26,47,38,18,49,20,31,17,45,12,52,8,44,6,23,5,24,13,14" data-style-groups="1|6;2|8|9;3;4;5;7|10;11|28|33;12|27|29;13|39;14|16|24|36|53|59;15|17|41|44;18|35;19|55;20|40;21|37|43|45;22|46;23|30;25|34|47|48|52|56;26|54|60;31|42;32;38|57;49|51;50;58">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/dishwasher/36.webp" alt="Dishwasher">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 25</span>
        <input type="range" class="fixture-inline-slider" min="1" max="25" value="1" aria-label="Dishwasher index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Fridge (Bottom Freezer)</h2>
    <div class="fixture-viewer" data-name="Fridge (Bottom Freezer)" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/fridge_bottom_freezer" data-ids="36,9,32,37,30,26,47,38,57,29,16,20,31,17,12,8,22" data-style-groups="1|5;2|8;3|4;6;7|9|10;11|45;12|36|59;13|24;14|16|23|49|53;15|34|51|60;17|27|32|40|46|57;18|21|43|52;19|29|35|48|58;20|30|37|41|44|55|56;22|33|39|50|54;25|26|31;28|38|42|47">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/fridge_bottom_freezer/36.webp" alt="Fridge (Bottom Freezer)">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 17</span>
        <input type="range" class="fixture-inline-slider" min="1" max="17" value="1" aria-label="Fridge (Bottom Freezer) index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Fridge (French Door)</h2>
    <div class="fixture-viewer" data-name="Fridge (French Door)" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/fridge_french_door" data-ids="1,8,10,4,6,11,12,13,15,17,19,20,22,25,30,51,44" data-style-groups="1;2|8;3|7|10;4;5|6|9;11|14|18|35;12|21|34|52|56;13|16|24|29|49|53;15|41|42;17|36|37|59;19|23|27|32|47|57;20|26|45|46|55;22|31|38;25|28|39|48|54|58;30|40|43;33|50|51|60;44">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/fridge_french_door/1.webp" alt="Fridge (French Door)">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 17</span>
        <input type="range" class="fixture-inline-slider" min="1" max="17" value="1" aria-label="Fridge (French Door) index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Fridge (Side by Side)</h2>
    <div class="fixture-viewer" data-name="Fridge (Side by Side)" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/fridge_side_by_side" data-ids="1,8,4,6,7,11,12,14,17,23,24,26,28,30,33,36" data-style-groups="1|5;2|8|10;3|4;6|9;7;11|15|18|22|34|45|49|52|53|54;12|13|16|31|41|42|46|58;14|19|21|43|55|60;17|20|29|50|59;23|27|38;24|25|47|56;26|32|48;28|37|40|44;30|39;33|35|51|57;36">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/fridge_side_by_side/1.webp" alt="Fridge (Side by Side)">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 16</span>
        <input type="range" class="fixture-inline-slider" min="1" max="16" value="1" aria-label="Fridge (Side by Side) index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Microwave</h2>
    <div class="fixture-viewer" data-name="Microwave" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/microwave" data-ids="36,40,15,4,37,30,9,42,26,47,38,18,49,57,16,20,31,17,45,12,33,8,44,55,22,11,1,6,23,46,60,34,28,41,48,50,56,27,21,54,59,13,29,19,7,2,58,39,14,3" data-style-groups="1;2|3;4;5;6;7|9;8;10;11;12|35|60;13|23;14;15;16;17;18;19;20;21|46;22;24;25;26;27|38|49;28;29;30;31;32;33;34;36;37|52;39;40;41|48;42;43;44;45;47;50;51;53;54;55;56;57;58;59">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/microwave/36.webp" alt="Microwave">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 50</span>
        <input type="range" class="fixture-inline-slider" min="1" max="50" value="1" aria-label="Microwave index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Oven</h2>
    <div class="fixture-viewer" data-name="Oven" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/oven" data-ids="46,25,40,41,19,14,48,31,53,22,43,59,38,47,32,8,55,50,60,6,4" data-style-groups="1;2|7|8;3|9;4;5;6;10;11|15|18|44|50;12|30|33|41;13|31|57;14|27|35|39|55;16|25|46;17|60;19|26|29|37|45;20|23|32|52;21|22|40|58;24|28|34|38|59;36|54|56;42|47|51;43|49;48|53">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/oven/46.webp" alt="Oven">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 21</span>
        <input type="range" class="fixture-inline-slider" min="1" max="21" value="1" aria-label="Oven index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Sink</h2>
    <div class="fixture-viewer" data-name="Sink" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/sink" data-ids="1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,37,38,40,42,43,44,45,46,47,48,50,51,53,54,57,58" data-style-groups="1;2;3|4;5;6;7;8;9;10;11|56;12;13;14;15;16;17;18;19;20;21;22;23;24|39|55;25;26|36;27;28;29;30;31|33;32|41;34;35;37;38|52;40;42|49;43;44;45;46;47;48;50;51;53;54;57|59;58|60">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/sink/1.webp" alt="Sink">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 49</span>
        <input type="range" class="fixture-inline-slider" min="1" max="49" value="1" aria-label="Sink index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Stand Mixer</h2>
    <div class="fixture-viewer" data-name="Stand Mixer" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/stand_mixer" data-ids="1,2,4,5,7,10,11,12,13,16,17,18,19,20,21,23,24,26,29,35,38,39,42,48,51" data-style-groups="1|8|9;2|3;4;5|6;7;10;11|14|15|22|59;12|37|50;13|49|52|56;16|27|45|57;17|47;18|32;19|25;20;21|30;23|28|33|46;24|40|43|53;26|31|34|55;29|41|44;35|36|60;38;39|54;42;48;51|58">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/stand_mixer/1.webp" alt="Stand Mixer">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 25</span>
        <input type="range" class="fixture-inline-slider" min="1" max="25" value="1" aria-label="Stand Mixer index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Stove (classic)</h2>
    <div class="fixture-viewer" data-name="Stove (classic)" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/stove_classic" data-ids="36,40,32,30,9,26,38,29,16,20,31,53,8,44,6,2,58" data-style-groups="1|4|6;2|5;3|10;7|9;8;11|12|32|35|39|57;13|15|37|44|59;14|16|22|27|30|51;17|23|58;18|20|21|38|41|46|52;19|34|45;24|40|60;25|42|48|54;26|33|43|47;28|31|36|50;29|49|55;53|56">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/stove_classic/36.webp" alt="Stove (classic)">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 17</span>
        <input type="range" class="fixture-inline-slider" min="1" max="17" value="1" aria-label="Stove (classic) index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Stove (wide)</h2>
    <div class="fixture-viewer" data-name="Stove (wide)" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/stove_wide" data-ids="1,60,12,13,55,16,19,18,28" data-style-groups="1|2|3|4|5|6|7|8|9|10;11|41|50|60;12|20|21|22|24|25|36|39|48|52;13|14|30|38|44|45|46|51|58;15|23|35|43|55|57;16|26|29|32|33|34|47|54|56|59;17|19|42;18|27|49;28|31|37|40|53">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/stove_wide/1.webp" alt="Stove (wide)">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 9</span>
        <input type="range" class="fixture-inline-slider" min="1" max="9" value="1" aria-label="Stove (wide) index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Stovetop</h2>
    <div class="fixture-viewer" data-name="Stovetop" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/stovetop" data-ids="1,2,5,6,7,8,9,10,35,12,13,14,56,16,17,18,49,20,21,25,55,36,48,51" data-style-groups="1|3;2|4;5;6;7;8;9;10;11|35|38|39|45|53;12|31|42|46;13|24|27|41|54;14|37|44|58;15|56;16|23|60;17;18|28|30|33;19|49;20|43;21|22;25|34|52;26|29|32|40|50|55|59;36|47|57;48;51">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/stovetop/1.webp" alt="Stovetop">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 24</span>
        <input type="range" class="fixture-inline-slider" min="1" max="24" value="1" aria-label="Stovetop index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Toaster</h2>
    <div class="fixture-viewer" data-name="Toaster" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/toaster" data-ids="1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,27,28,29,30,32,33,34,35,37,39,41,42,43,44,47,48,50,53,55,58,60" data-style-groups="1;2;3;4|5;6;7;8;9;10;11|26|45;12|51;13;14|40;15;16|23|57|59;17|49;18;19;20;21|36;22|31;24;25;27;28|46;29;30|38;32;33;34;35;37;39|56;41;42|52;43|54;44;47;48;50;53;55;58;60">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/toaster/1.webp" alt="Toaster">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 44</span>
        <input type="range" class="fixture-inline-slider" min="1" max="44" value="1" aria-label="Toaster index">
      </div>
    </div>
  </div>

  <div class="fixture-card">
    <h2 class="fixture-card-title">Toaster Oven</h2>
    <div class="fixture-viewer" data-name="Toaster Oven" data-base="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/toaster_oven" data-ids="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,32,34,35,37,38,40,41,43,44,45,46,50,52,53,54,55,57,59" data-style-groups="1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16|22;17|31|33|47;18|49;19|48;20;21|39;23;24;25;26|60;27;28|36;29|42;30;32;34|58;35;37;38;40;41;43;44;45;46|51;50;52;53;54|56;55;57;59">
      <div class="fixture-item fixture-preview" role="button" tabindex="0">
        <img class="fixture-preview-image" src="https://pub-74eb5884b9764cdf86b3352f8597995d.r2.dev/toaster_oven/1.webp" alt="Toaster Oven">
        <div class="label fixture-preview-label">Style</div>
      </div>
      <div class="fixture-inline-slider-container">
        <span class="fixture-inline-counter" aria-label="Current fixture" data-role="counter">1 / 47</span>
        <input type="range" class="fixture-inline-slider" min="1" max="47" value="1" aria-label="Toaster Oven index">
      </div>
    </div>
  </div>
</div>

<script>

// Fixture modal state (single-preview + slider)
let currentViewer = null;

// Current style filter value (for highlighting)
let currentStyleFilter = null;

// Preload range: Image() for immediate neighbors (reliable cache), link prefetch for wider range
const PRELOAD_IMAGE_RANGE = 4;   // ±4 with Image() when card not actively used
const PRELOAD_LINK_RANGE = 8;   // ±8 with <link rel="prefetch"> when card not actively used
const PRELOAD_IMAGE_RANGE_ACTIVE = 15;  // ±15 when user is using this card (slider/modal)
const PRELOAD_LINK_RANGE_ACTIVE = 25;  // ±25 prefetch when active
const HOVER_PREFETCH_COUNT = 30;       // prefetch this many images when hovering a card
const BACKFILL_CHUNK_SIZE = 10;         // backfill rest of card in chunks of this size

// Which viewer is "active" (inline slider in use); modal viewer uses currentViewer
let activeViewer = null;
let activeViewerTimeout = null;

const PREFETCH_THROTTLE_MS = 90;  // during drag, run prefetch at most this often to avoid lag on large fixtures

function prefetchAdjacentImages(viewer, currentPos, isActive) {
  // When slider is active (many styles), throttle to avoid blocking: we were re-running this on every
  // input event, removing/re-adding 50+ link elements and creating 30+ Image() each time.
  if (isActive) {
    const now = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
    const last = viewer._rcLastPrefetchTime;
    if (last != null && (now - last) < PREFETCH_THROTTLE_MS) return;
    viewer._rcLastPrefetchTime = now;
  }

  const base = viewer.dataset.base;
  const ids = getViewerIds(viewer);
  const count = ids.length;
  const imageRange = isActive ? PRELOAD_IMAGE_RANGE_ACTIVE : PRELOAD_IMAGE_RANGE;
  const linkRange = isActive ? PRELOAD_LINK_RANGE_ACTIVE : PRELOAD_LINK_RANGE;

  // Remove old prefetch links
  const oldLinks = document.querySelectorAll('link[data-fixture-prefetch]');
  oldLinks.forEach(link => link.remove());

  for (let offset = -linkRange; offset <= linkRange; offset++) {
    const targetPos = currentPos + offset;
    if (targetPos < 1 || targetPos > count || targetPos === currentPos) continue;
    const imageId = ids[targetPos - 1];
    const src = `${base}/${imageId}.webp`;

    // Immediate neighbors: force load with Image() for reliable cache (works well cross-origin)
    if (Math.abs(offset) <= imageRange) {
      const img = new Image();
      img.src = src;
    }

    // Wider range: hint via prefetch link
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.as = 'image';
    link.href = src;
    link.setAttribute('data-fixture-prefetch', 'true');
    document.head.appendChild(link);
  }
}

// Prefetch a range of images for a viewer (e.g. on hover or backfill)
function prefetchViewerRange(viewer, fromPos, toPos) {
  const base = viewer.dataset.base;
  const ids = getViewerIds(viewer);
  const count = ids.length;
  const from = Math.max(1, Math.min(fromPos, count));
  const to = Math.max(from, Math.min(toPos, count));
  for (let i = from; i <= to; i++) {
    const imageId = ids[i - 1];
    const img = new Image();
    img.src = `${base}/${imageId}.webp`;
  }
}

// Ensure the image at this position is loaded into cache (so going back to it is instant)
function ensureImageCached(viewer, pos) {
  const count = getViewerIds(viewer).length;
  if (pos < 1 || pos > count) return;
  if (!viewer._rcCachedPositions) viewer._rcCachedPositions = new Set();
  if (viewer._rcCachedPositions.has(pos)) return;
  viewer._rcCachedPositions.add(pos);
  const base = viewer.dataset.base;
  const ids = getViewerIds(viewer);
  const imageId = ids[pos - 1];
  const img = new Image();
  img.src = `${base}/${imageId}.webp`;
}

function scheduleBackfill(viewer) {
  if (viewer._rcBackfillScheduled) return;
  viewer._rcBackfillScheduled = true;
  const run = function() {
    const ids = getViewerIds(viewer);
    const count = ids.length;
    let pos = 1;
    function doChunk() {
      const end = Math.min(pos + BACKFILL_CHUNK_SIZE - 1, count);
      for (let i = pos; i <= end; i++) {
        const imageId = ids[i - 1];
        const img = new Image();
        img.src = `${viewer.dataset.base}/${imageId}.webp`;
      }
      pos = end + 1;
      if (pos <= count) {
        if (typeof requestIdleCallback !== 'undefined') {
          requestIdleCallback(doChunk, { timeout: 500 });
        } else {
          setTimeout(doChunk, 50);
        }
      }
    }
    if (typeof requestIdleCallback !== 'undefined') {
      requestIdleCallback(doChunk, { timeout: 2000 });
    } else {
      setTimeout(doChunk, 100);
    }
  };
  if (typeof requestIdleCallback !== 'undefined') {
    requestIdleCallback(run, { timeout: 2000 });
  } else {
    setTimeout(run, 500);
  }
}

function closeFixtureModal() {
  document.getElementById('fixtureModal').classList.remove('active');
  document.body.style.overflow = '';
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function getViewerIds(viewer) {
  const idsAttr = viewer.dataset.ids;
  if (idsAttr) {
    return idsAttr.split(',').map(s => parseInt(s.trim(), 10)).filter(n => Number.isFinite(n));
  }
  const count = parseInt(viewer.dataset.count || "1", 10);
  return Array.from({ length: count }, (_, i) => i + 1);
}

function getViewerStyleGroups(viewer) {
  // Format: "1|2;3;4|5|6" (groups separated by ';', style ids inside group separated by '|')
  // Cached on the element to avoid re-parsing on every slider update.
  if (viewer._rcStyleGroups) return viewer._rcStyleGroups;
  const raw = viewer.dataset.styleGroups;
  if (!raw) return null;
  const groups = raw.split(';').map(g => g.trim()).filter(Boolean).map(g => {
    return g.split('|').map(s => parseInt(s.trim(), 10)).filter(n => Number.isFinite(n));
  });
  viewer._rcStyleGroups = groups;
  return groups;
}

function sortViewerByStyleOrder(viewer) {
  // Sort the IDs and style groups by the minimum style number in each group
  const styleGroups = getViewerStyleGroups(viewer);
  
  if (!styleGroups || styleGroups.length === 0) return;
  
  // Get IDs (either from data-ids or generate from data-count)
  let ids = getViewerIds(viewer);
  
  if (ids.length !== styleGroups.length) return; // Safety check
  
  // Create array of {id, styleGroup, minStyle} for sorting
  const items = ids.map((id, index) => {
    const group = styleGroups[index] || [id];
    const minStyle = Math.min(...group);
    return { id, styleGroup: group, minStyle, originalIndex: index };
  });
  
  // Sort by minimum style number
  items.sort((a, b) => a.minStyle - b.minStyle);
  
  // Update the data attributes with sorted order
  const sortedIds = items.map(item => item.id);
  const sortedStyleGroups = items.map(item => item.styleGroup);
  
  // Update data-ids (create it if it doesn't exist, or update if it does)
  viewer.dataset.ids = sortedIds.join(',');
  
  // Update data-style-groups
  const sortedStyleGroupsStr = sortedStyleGroups.map(g => g.join('|')).join(';');
  viewer.dataset.styleGroups = sortedStyleGroupsStr;
  
  // Remove data-count if it exists (since we now have data-ids)
  if (viewer.hasAttribute('data-count')) {
    viewer.removeAttribute('data-count');
  }
  
  // Clear cache so it gets re-parsed with new order
  viewer._rcStyleGroups = null;
}

function getStyleLabelForPos(viewer, pos, fallbackId) {
  const groups = getViewerStyleGroups(viewer);
  const group = (groups && groups[pos - 1] && groups[pos - 1].length) ? groups[pos - 1] : [fallbackId];
  const styleNumbers = group.join('/');
  
  // If filtering, highlight the matching style number
  if (currentStyleFilter !== null) {
    const filterNum = currentStyleFilter;
    const highlighted = styleNumbers.split('/').map(num => {
      const numStr = String(num).trim();
      if (parseInt(numStr, 10) === filterNum) {
        return `<span class="style-highlight">${numStr}</span>`;
      }
      return numStr;
    }).join('/');
    return `Style ${highlighted}`;
  }
  
  return `Style ${styleNumbers}`;
}

function setViewerIndex(viewer, index, skipPrefetch) {
  const base = viewer.dataset.base;
  const ids = getViewerIds(viewer);
  const count = ids.length;
  const pos = clamp(parseInt(index, 10) || 1, 1, count);
  const imageId = ids[pos - 1];

  const img = viewer.querySelector('.fixture-preview-image');
  const label = viewer.querySelector('.fixture-preview-label');
  const counter = viewer.querySelector('.fixture-inline-counter');
  const slider = viewer.querySelector('.fixture-inline-slider');

  const src = `${base}/${imageId}.webp`;
  const styleLabel = getStyleLabelForPos(viewer, pos, imageId);
  
  if (!skipPrefetch) {
    const isActive = (viewer === currentViewer || viewer === activeViewer);
    prefetchAdjacentImages(viewer, pos, isActive);
  }
  
  if (img) {
    if (!viewer._rcCachedPositions) viewer._rcCachedPositions = new Set();
    const alreadySeen = viewer._rcCachedPositions.has(pos);
    // Only show loading dim when switching to an image we haven't displayed yet
    if (img.src !== src && !alreadySeen) {
      img.style.opacity = '0.6';
      img.style.transition = 'opacity 0.2s';
    }
    img.src = src;
    img.alt = styleLabel;
    
    if (alreadySeen) {
      img.style.opacity = '1';
    } else if (!img.complete) {
      img.onload = function() {
        this.style.opacity = '1';
        this.onload = null;
      };
    } else {
      img.style.opacity = '1';
    }
    viewer._rcCachedPositions.add(pos);
  }
  if (label) label.innerHTML = styleLabel;
  if (counter) counter.textContent = `${pos} / ${count}`;
  if (slider) slider.value = String(pos);

  viewer.dataset.current = String(pos);

  if (currentViewer === viewer) setModalIndex(pos);
}

// Update only counter and label (no image load) for responsive feedback while dragging
function updateViewerCounterOnly(viewer, index) {
  const ids = getViewerIds(viewer);
  const count = ids.length;
  const pos = clamp(parseInt(index, 10) || 1, 1, count);
  const imageId = ids[pos - 1];
  const styleLabel = getStyleLabelForPos(viewer, pos, imageId);
  const label = viewer.querySelector('.fixture-preview-label');
  const counter = viewer.querySelector('.fixture-inline-counter');
  const slider = viewer.querySelector('.fixture-inline-slider');
  if (label) label.innerHTML = styleLabel;
  if (counter) counter.textContent = `${pos} / ${count}`;
  if (slider) slider.value = String(pos);
  viewer.dataset.current = String(pos);
}

function updateModalCounterOnly(viewer, index) {
  const ids = getViewerIds(viewer);
  const count = ids.length;
  const pos = clamp(parseInt(index, 10) || 1, 1, count);
  const imageId = ids[pos - 1];
  const styleLabel = getStyleLabelForPos(viewer, pos, imageId);
  const modalLabel = document.getElementById('fixtureModalLabel');
  const modalCounter = document.getElementById('fixtureModalCounter');
  const modalSlider = document.getElementById('fixtureModalSlider');
  if (modalLabel) modalLabel.innerHTML = styleLabel;
  if (modalCounter) modalCounter.textContent = `${pos} / ${count}`;
  if (modalSlider) modalSlider.value = String(pos);
  viewer.dataset.current = String(pos);
  updateViewerCounterOnly(viewer, index);
}

function setModalIndex(index) {
  if (!currentViewer) return;
  const base = currentViewer.dataset.base;
  const ids = getViewerIds(currentViewer);
  const count = ids.length;
  const pos = clamp(parseInt(index, 10) || 1, 1, count);
  const imageId = ids[pos - 1];

  const modalImg = document.getElementById('fixtureModalImage');
  const modalLabel = document.getElementById('fixtureModalLabel');
  const modalCounter = document.getElementById('fixtureModalCounter');
  const modalSlider = document.getElementById('fixtureModalSlider');
  const modalPrev = document.getElementById('fixtureModalPrev');
  const modalNext = document.getElementById('fixtureModalNext');

  const src = `${base}/${imageId}.webp`;
  const styleLabel = getStyleLabelForPos(currentViewer, pos, imageId);
  
  // Prefetch adjacent images; modal viewer is always active (wider range)
  prefetchAdjacentImages(currentViewer, pos, true);
  
  if (modalImg) {
    if (!currentViewer._rcCachedPositions) currentViewer._rcCachedPositions = new Set();
    const alreadySeen = currentViewer._rcCachedPositions.has(pos);
    if (modalImg.src !== src && !alreadySeen) {
      modalImg.style.opacity = '0.6';
      modalImg.style.transition = 'opacity 0.2s';
    }
    modalImg.src = src;
    
    if (alreadySeen) {
      modalImg.style.opacity = '1';
    } else if (!modalImg.complete) {
      modalImg.onload = function() {
        this.style.opacity = '1';
        this.onload = null;
      };
    } else {
      modalImg.style.opacity = '1';
    }
    currentViewer._rcCachedPositions.add(pos);
  }
  if (modalLabel) modalLabel.innerHTML = styleLabel;
  if (modalCounter) modalCounter.textContent = `${pos} / ${count}`;
  if (modalSlider) modalSlider.value = String(pos);
  if (modalPrev) modalPrev.disabled = pos <= 1;
  if (modalNext) modalNext.disabled = pos >= count;
}

function openFixtureModalForViewer(viewer) {
  currentViewer = viewer;
  const ids = getViewerIds(viewer);
  const count = ids.length;
  const pos = parseInt(viewer.dataset.current || "1", 10) || 1;

  const modalSlider = document.getElementById('fixtureModalSlider');
  modalSlider.min = "1";
  modalSlider.max = String(count);

  setModalIndex(pos);
  document.getElementById('fixtureModal').classList.add('active');
  document.body.style.overflow = 'hidden';
  scheduleBackfill(viewer);
}

function fixtureModalSliderChange(value) {
  if (!currentViewer) return;
  setModalIndex(value);
  setViewerIndex(currentViewer, value);
}

function initModalSlider() {
  const modalSlider = document.getElementById('fixtureModalSlider');
  if (!modalSlider) return;
  modalSlider.addEventListener('input', function() {
    const value = this.value;
    const valueNum = parseInt(value, 10);
    if (currentViewer) {
      updateModalCounterOnly(currentViewer, value);
      ensureImageCached(currentViewer, valueNum);
      fixtureModalSliderChange(value);
    }
  });
  modalSlider.addEventListener('change', function() {
    if (currentViewer) currentViewer._rcLastPrefetchTime = undefined;
    fixtureModalSliderChange(this.value);
  });
}

function fixtureModalStep(delta) {
  if (!currentViewer) return;
  const ids = getViewerIds(currentViewer);
  const count = ids.length;
  const cur = parseInt(currentViewer.dataset.current || "1", 10) || 1;
  const next = clamp(cur + delta, 1, count);
  if (next === cur) return;
  setModalIndex(next);
  setViewerIndex(currentViewer, next);
}

// Style filter functionality
function filterFixturesByStyle(styleNumber) {
  const styleNum = parseInt(styleNumber, 10);

  // If invalid / empty: clear filter + reset all viewers to their first entry
  if (!styleNumber || isNaN(styleNum) || styleNum < 1 || styleNum > 60) {
    currentStyleFilter = null;
    document.querySelectorAll('.fixture-card').forEach(card => {
      card.classList.remove('hidden-by-filter');
      const viewer = card.querySelector('.fixture-viewer');
      if (viewer) setViewerIndex(viewer, 1, true);
    });
    const clearBtn = document.getElementById('style-filter-clear');
    const input = document.getElementById('style-filter-input');
    const hasValue = input && String(input.value || '').trim().length > 0;
    if (clearBtn) clearBtn.style.display = hasValue ? 'inline-block' : 'none';
    return;
  }
  
  // Store the filter value for highlighting
  currentStyleFilter = styleNum;

  // Jump every fixture viewer to the entry that contains this style number.
  // If a fixture truly doesn't contain the style, hide it.
  document.querySelectorAll('.fixture-card').forEach(card => {
    const viewer = card.querySelector('.fixture-viewer');
    if (!viewer) {
      card.classList.add('hidden-by-filter');
      return;
    }

    const styleGroups = getViewerStyleGroups(viewer);
    if (!styleGroups || styleGroups.length === 0) {
      card.classList.add('hidden-by-filter');
      return;
    }

    const matchIdx = styleGroups.findIndex(group => group.includes(styleNum));
    if (matchIdx === -1) {
      card.classList.add('hidden-by-filter');
      return;
    }

    card.classList.remove('hidden-by-filter');
    setViewerIndex(viewer, matchIdx + 1, true);
  });
  
  if (currentViewer) {
    const modalPos = parseInt(currentViewer.dataset.current || "1", 10) || 1;
    setModalIndex(modalPos);
  }

  const clearBtn = document.getElementById('style-filter-clear');
  if (clearBtn) clearBtn.style.display = 'inline-block';
}

function clearStyleFilter() {
  const input = document.getElementById('style-filter-input');
  if (input) {
    input.value = '';
    input.classList.remove('rc-invalid');
    filterFixturesByStyle('');
  }
}

// Close modal on escape key
document.addEventListener('keydown', function(e) {
  const modal = document.getElementById('fixtureModal');
  if (modal && modal.classList.contains('active')) {
    if (e.key === 'Escape') {
      closeFixtureModal();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      fixtureModalStep(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      fixtureModalStep(1);
    }
  }
});

// Close modal when clicking outside the image
document.getElementById('fixtureModal').addEventListener('click', function(e) {
  if (e.target === this) closeFixtureModal();
});

function initFixtureViewers() {
  const viewers = document.querySelectorAll('.fixture-viewer');
  viewers.forEach(function(viewer) {
    // Sort styles by order (minimum style number in each group)
    sortViewerByStyleOrder(viewer);
    
    // Initialize to the first image
    setViewerIndex(viewer, 1);

    const slider = viewer.querySelector('.fixture-inline-slider');
    const preview = viewer.querySelector('.fixture-preview');
    if (slider) {
      const ids = getViewerIds(viewer);
      slider.max = String(ids.length);
      function setActiveAndBackfill() {
        activeViewer = viewer;
        if (activeViewerTimeout) clearTimeout(activeViewerTimeout);
        activeViewerTimeout = setTimeout(function() { activeViewer = null; }, 3000);
        scheduleBackfill(viewer);
      }
      slider.addEventListener('focus', setActiveAndBackfill);
      slider.addEventListener('input', function() {
        setActiveAndBackfill();
        const value = this.value;
        const valueNum = parseInt(value, 10);
        updateViewerCounterOnly(viewer, value);
        ensureImageCached(viewer, valueNum);
        // Update image every time; prefetch is throttled inside setViewerIndex so fast drags stay smooth
        setViewerIndex(viewer, value);
      });
      slider.addEventListener('change', function() {
        viewer._rcLastPrefetchTime = undefined;  // force prefetch for final position when drag ends
        setViewerIndex(viewer, this.value);
      });
      slider.addEventListener('blur', function() {
        activeViewer = null;
        if (activeViewerTimeout) clearTimeout(activeViewerTimeout);
        activeViewerTimeout = null;
      });
    }

    // Click (or Enter/Space) opens modal. Slider stays usable both inline & in modal.
    if (preview) {
      preview.addEventListener('click', function() {
        openFixtureModalForViewer(viewer);
      });
      preview.addEventListener('keydown', function(ev) {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          openFixtureModalForViewer(viewer);
        }
      });
    }

    // Prefetch first N images when user hovers over the card (once per card)
    const card = viewer.closest('.fixture-card');
    if (card && !card._rcHoverPrefetchAttached) {
      card._rcHoverPrefetchAttached = true;
      card.addEventListener('mouseenter', function() {
        if (viewer._rcHoverPrefetched) return;
        viewer._rcHoverPrefetched = true;
        const count = getViewerIds(viewer).length;
        const to = Math.min(HOVER_PREFETCH_COUNT, count);
        if (to < 1) return;
        function run() {
          prefetchViewerRange(viewer, 1, to);
        }
        if (typeof requestIdleCallback !== 'undefined') {
          requestIdleCallback(run, { timeout: 1000 });
        } else {
          setTimeout(run, 100);
        }
      });
    }
  });
  
  // Prefetch initial images for all viewers after a short delay
  // This ensures the page is interactive first, then prefetches in background
  setTimeout(function() {
    const viewers = document.querySelectorAll('.fixture-viewer');
    viewers.forEach(function(viewer) {
      const ids = getViewerIds(viewer);
      if (ids.length > 0) {
        // Prefetch first few images for each viewer
        prefetchAdjacentImages(viewer, 1);
      }
    });
  }, 500);
}

// Initialize style filter
function initStyleFilter() {
  const filterInput = document.getElementById('style-filter-input');
  if (filterInput) {
    filterInput.addEventListener('input', function(e) {
      // Keep only digits and limit to 2 characters (1-2 digits)
      let v = String(e.target.value || '').replace(/\\D+/g, '').slice(0, 2);
      if (v !== e.target.value) e.target.value = v;

      const n = parseInt(v, 10);
      const invalid = v.length > 0 && (!Number.isFinite(n) || n < 1 || n > 60);
      e.target.classList.toggle('rc-invalid', invalid);

      filterFixturesByStyle(v);
    });
    filterInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        // Re-validate on Enter
        const v = String(e.target.value || '').replace(/\\D+/g, '').slice(0, 2);
        e.target.value = v;
        const n = parseInt(v, 10);
        const invalid = v.length > 0 && (!Number.isFinite(n) || n < 1 || n > 60);
        e.target.classList.toggle('rc-invalid', invalid);
        filterFixturesByStyle(v);
      }
    });
  }
}

// Initialize on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    initFixtureViewers();
    initModalSlider();
    initStyleFilter();
  });
} else {
  initFixtureViewers();
  initModalSlider();
  initStyleFilter();
}
</script>

