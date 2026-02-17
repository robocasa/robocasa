# Scenes

RoboCasa offers a large array of kitchen scenes with fully interactive cabinets, drawers, and appliances.
We model 2,500 pretraining kitchen scenes, and 10 distinct target kitchen scenes.
Each scene is defined by a **(layout, style)** pair.
Layouts define the basic floor plan of the kitchen, setting the arrangment of fixtures (cabinets, microwaves, counters, etc.).
Styles define relevant textures and fixture attributes (cabinet handle types, door types, coffee machine types, etc).

The list of layout definitions can he found [here](https://github.com/robocasa/robocasa/tree/main/robocasa/models/assets/scenes/kitchen_layouts) and style definitions can be found [here](https://github.com/robocasa/robocasa/tree/main/robocasa/models/assets/scenes/kitchen_styles).

Environments can be initialized with pretraining or target kitchens using the `split` argument. For example:
```
import robocasa
import gymnasium as gym

# creates an environment with pretraining kitchen scenes
env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain",
)

# creates an environment with target kitchen scenes
env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="target",
)
```

<style>
.sort-btn {
  background: #3498db;
  color: white;
  border: none;
  padding: 8px 16px;
  margin: 4px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}
.sort-btn:hover { background: #2980b9; }
.sort-btn.active { background: #1a5276; }
.scene-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.scene-item {
  text-align: center;
  cursor: pointer;
}
.scene-item img {
  width: 100%;
  border-radius: 6px;
  border: 1px solid rgba(128,128,128,0.3);
  transition: transform 0.2s, box-shadow 0.2s;
}
.scene-item img:hover {
  transform: scale(1.03);
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.scene-item .label {
  font-size: 12px;
  margin-top: 4px;
  color: inherit;
  opacity: 0.8;
}
.scene-item .layout-style {
  font-size: 11px;
  margin-top: 6px;
  color: inherit;
  opacity: 0.6;
  font-weight: 500;
}

/* Modal/Lightbox styles */
.scene-modal {
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
.scene-modal.active {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.modal-content {
  position: relative;
  max-width: min(90%, 900px);
  max-height: 75vh;
  text-align: center;
}
.modal-content img {
  max-width: 100%;
  max-height: 70vh;
  width: auto;
  height: auto;
  aspect-ratio: 1080 / 720;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.modal-label {
  color: white;
  font-size: 18px;
  margin-top: 12px;
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

/* Normalize image appearance in dark mode */
@media (prefers-color-scheme: dark) {
  .modal-content img,
  .scene-item img {
    /* Avoid black-crush: don't alter the image pixels with CSS filters.
       Instead, give the image a subtle neutral backdrop + border in dark mode. */
    filter: none;
    /* A slightly brighter matte + halo reduces perceived "black crush"
       from the surrounding dark UI without altering the image pixels. */
    background-color: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow:
      0 0 0 1px rgba(0, 0, 0, 0.35) inset,
      0 0 0 10px rgba(255, 255, 255, 0.025);
    border-radius: 8px;
  }
  .modal-content {
    background: linear-gradient(to bottom, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.01));
  }
}

/* PyData theme dark-mode uses html[data-theme="dark"] (not just prefers-color-scheme).
   Override any theme-level image filters to prevent black crush. */
html[data-theme="dark"] .modal-content img,
html[data-theme="dark"] .scene-item img {
  filter: none !important;
  opacity: 1 !important;
  background-color: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.14);
  box-shadow:
    0 0 0 1px rgba(0, 0, 0, 0.35) inset,
    0 0 0 10px rgba(255, 255, 255, 0.025);
  border-radius: 8px;
}
</style>

<script>
  // Sphinx copies assets from `html_static_path` into `_static/`.
  // Ensure scene thumbnails point to the emitted `_static/` locations.
  (function () {
    function fixSceneStaticPaths() {
      const rules = [
        ["../pretrain_scenes/", "../_static/pretrain_scenes/"],
        ["../target_scenes/", "../_static/target_scenes/"],
      ];
      const imgs = Array.from(document.querySelectorAll(".scene-item img"));
      for (const img of imgs) {
        const src = img.getAttribute("src") || "";
        for (const [from, to] of rules) {
          if (src.startsWith(from)) {
            img.setAttribute("src", to + src.slice(from.length));
            break;
          }
        }
      }
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fixSceneStaticPaths);
    else fixSceneStaticPaths();
  })();
</script>

<!-- Modal HTML -->
<div id="sceneModal" class="scene-modal">
  <span class="modal-close" onclick="closeModal()">&times;</span>
  <div class="modal-content">
    <img id="modalImage" src="" alt="Scene">
    <div id="modalLabel" class="modal-label"></div>
  </div>
  <div class="modal-slider-container">
    <button type="button" class="modal-arrow" id="sceneModalPrev" aria-label="Previous scene" onclick="navigateModal(-1)">&#8249;</button>
    <span class="modal-counter" id="modalCounter">1 / 1</span>
    <input type="range" class="modal-slider" id="modalSlider" min="0" value="0" onchange="sliderChange(this.value)" oninput="sliderChange(this.value)">
    <button type="button" class="modal-arrow" id="sceneModalNext" aria-label="Next scene" onclick="navigateModal(1)">&#8250;</button>
  </div>
</div>

## Target kitchens

<div class="scene-grid" id="sceneGridTarget">
  <div class="scene-item" data-layout="1" data-style="1"><img src="../target_scenes/layout1_style1.png"><div class="label">One Wall Layout / Industrial Style<br><span class="layout-style">layout 1 / style 1</span></div></div>
  <div class="scene-item" data-layout="2" data-style="2"><img src="../target_scenes/layout2_style2.png"><div class="label">One Wall Island Layout / Scandinavian Style<br><span class="layout-style">layout 2 / style 2</span></div></div>
  <div class="scene-item" data-layout="3" data-style="3"><img src="../target_scenes/layout3_style3.png"><div class="label">L-shaped Layout / Modern 1 Style<br><span class="layout-style">layout 3 / style 3</span></div></div>
  <div class="scene-item" data-layout="4" data-style="4"><img src="../target_scenes/layout4_style4.png"><div class="label">L-shaped Island Layout / Farmhouse Style<br><span class="layout-style">layout 4 / style 4</span></div></div>
  <div class="scene-item" data-layout="5" data-style="5"><img src="../target_scenes/layout5_style5.png"><div class="label">Galley Layout / Modern 2 Style<br><span class="layout-style">layout 5 / style 5</span></div></div>
  <div class="scene-item" data-layout="6" data-style="6"><img src="../target_scenes/layout6_style6.png"><div class="label">U-shaped Layout / Traditional Style<br><span class="layout-style">layout 6 / style 6</span></div></div>
  <div class="scene-item" data-layout="7" data-style="7"><img src="../target_scenes/layout7_style7.png"><div class="label">U-shaped Island Layout / Mediterranean Style<br><span class="layout-style">layout 7 / style 7</span></div></div>
  <div class="scene-item" data-layout="8" data-style="8"><img src="../target_scenes/layout8_style8.png"><div class="label">G-shaped Layout / Rustic Style<br><span class="layout-style">layout 8 / style 8</span></div></div>
  <div class="scene-item" data-layout="9" data-style="9"><img src="../target_scenes/layout9_style9.png"><div class="label">G-shaped Large Layout / Coastal Style<br><span class="layout-style">layout 9 / style 9</span></div></div>
  <div class="scene-item" data-layout="10" data-style="10"><img src="../target_scenes/layout10_style10.png"><div class="label">Wraparound Layout / Transitional Style<br><span class="layout-style">layout 10 / style 10</span></div></div>
</div>


## Pretraining kitchens 

For pretraining kitchens, we provide 50 layouts and 50 styles.
The layouts are sourced from real world kitchens based on Zillow listings found across diverse locations in the United States (California Bay Area, Denver, Austin, Atlanta, Boston).
Each layout can be paired with any style, resulting in **2,500 total pretraining kitchen scenes**.

<div>
  <button class="sort-btn active" onclick="sortScenes('layout')">Sort by Layout</button>
  <button class="sort-btn" onclick="sortScenes('style')">Sort by Style</button>
</div>

<div class="scene-grid" id="sceneGrid">
  <div class="scene-item" data-layout="11" data-style="27"><img src="../pretrain_scenes/layout11_style27.png"><div class="label">Layout 11 / Style 27</div></div>
  <div class="scene-item" data-layout="12" data-style="11"><img src="../pretrain_scenes/layout12_style11.png"><div class="label">Layout 12 / Style 11</div></div>
  <div class="scene-item" data-layout="13" data-style="20"><img src="../pretrain_scenes/layout13_style20.png"><div class="label">Layout 13 / Style 20</div></div>
  <div class="scene-item" data-layout="14" data-style="23"><img src="../pretrain_scenes/layout14_style23.png"><div class="label">Layout 14 / Style 23</div></div>
  <div class="scene-item" data-layout="15" data-style="15"><img src="../pretrain_scenes/layout15_style15.png"><div class="label">Layout 15 / Style 15</div></div>
  <div class="scene-item" data-layout="16" data-style="16"><img src="../pretrain_scenes/layout16_style16.png"><div class="label">Layout 16 / Style 16</div></div>
  <div class="scene-item" data-layout="17" data-style="32"><img src="../pretrain_scenes/layout17_style32.png"><div class="label">Layout 17 / Style 32</div></div>
  <div class="scene-item" data-layout="18" data-style="26"><img src="../pretrain_scenes/layout18_style26.png"><div class="label">Layout 18 / Style 26</div></div>
  <div class="scene-item" data-layout="19" data-style="13"><img src="../pretrain_scenes/layout19_style13.png"><div class="label">Layout 19 / Style 13</div></div>
  <div class="scene-item" data-layout="20" data-style="55"><img src="../pretrain_scenes/layout20_style55.png"><div class="label">Layout 20 / Style 55</div></div>
  <div class="scene-item" data-layout="21" data-style="58"><img src="../pretrain_scenes/layout21_style58.png"><div class="label">Layout 21 / Style 58</div></div>
  <div class="scene-item" data-layout="22" data-style="30"><img src="../pretrain_scenes/layout22_style30.png"><div class="label">Layout 22 / Style 30</div></div>
  <div class="scene-item" data-layout="23" data-style="24"><img src="../pretrain_scenes/layout23_style24.png"><div class="label">Layout 23 / Style 24</div></div>
  <div class="scene-item" data-layout="24" data-style="50"><img src="../pretrain_scenes/layout24_style50.png"><div class="label">Layout 24 / Style 50</div></div>
  <div class="scene-item" data-layout="25" data-style="39"><img src="../pretrain_scenes/layout25_style39.png"><div class="label">Layout 25 / Style 39</div></div>
  <div class="scene-item" data-layout="26" data-style="22"><img src="../pretrain_scenes/layout26_style22.png"><div class="label">Layout 26 / Style 22</div></div>
  <div class="scene-item" data-layout="27" data-style="53"><img src="../pretrain_scenes/layout27_style53.png"><div class="label">Layout 27 / Style 53</div></div>
  <div class="scene-item" data-layout="28" data-style="29"><img src="../pretrain_scenes/layout28_style29.png"><div class="label">Layout 28 / Style 29</div></div>
  <div class="scene-item" data-layout="29" data-style="57"><img src="../pretrain_scenes/layout29_style57.png"><div class="label">Layout 29 / Style 57</div></div>
  <div class="scene-item" data-layout="30" data-style="25"><img src="../pretrain_scenes/layout30_style25.png"><div class="label">Layout 30 / Style 25</div></div>
  <div class="scene-item" data-layout="31" data-style="36"><img src="../pretrain_scenes/layout31_style36.png"><div class="label">Layout 31 / Style 36</div></div>
  <div class="scene-item" data-layout="32" data-style="54"><img src="../pretrain_scenes/layout32_style54.png"><div class="label">Layout 32 / Style 54</div></div>
  <div class="scene-item" data-layout="33" data-style="60"><img src="../pretrain_scenes/layout33_style60.png"><div class="label">Layout 33 / Style 60</div></div>
  <div class="scene-item" data-layout="34" data-style="37"><img src="../pretrain_scenes/layout34_style37.png"><div class="label">Layout 34 / Style 37</div></div>
  <div class="scene-item" data-layout="35" data-style="14"><img src="../pretrain_scenes/layout35_style14.png"><div class="label">Layout 35 / Style 14</div></div>
  <div class="scene-item" data-layout="36" data-style="12"><img src="../pretrain_scenes/layout36_style12.png"><div class="label">Layout 36 / Style 12</div></div>
  <div class="scene-item" data-layout="37" data-style="17"><img src="../pretrain_scenes/layout37_style17.png"><div class="label">Layout 37 / Style 17</div></div>
  <div class="scene-item" data-layout="38" data-style="42"><img src="../pretrain_scenes/layout38_style42.png"><div class="label">Layout 38 / Style 42</div></div>
  <div class="scene-item" data-layout="39" data-style="28"><img src="../pretrain_scenes/layout39_style28.png"><div class="label">Layout 39 / Style 28</div></div>
  <div class="scene-item" data-layout="40" data-style="33"><img src="../pretrain_scenes/layout40_style33.png"><div class="label">Layout 40 / Style 33</div></div>
  <div class="scene-item" data-layout="41" data-style="48"><img src="../pretrain_scenes/layout41_style48.png"><div class="label">Layout 41 / Style 48</div></div>
  <div class="scene-item" data-layout="42" data-style="51"><img src="../pretrain_scenes/layout42_style51.png"><div class="label">Layout 42 / Style 51</div></div>
  <div class="scene-item" data-layout="43" data-style="31"><img src="../pretrain_scenes/layout43_style31.png"><div class="label">Layout 43 / Style 31</div></div>
  <div class="scene-item" data-layout="44" data-style="43"><img src="../pretrain_scenes/layout44_style43.png"><div class="label">Layout 44 / Style 43</div></div>
  <div class="scene-item" data-layout="45" data-style="52"><img src="../pretrain_scenes/layout45_style52.png"><div class="label">Layout 45 / Style 52</div></div>
  <div class="scene-item" data-layout="46" data-style="38"><img src="../pretrain_scenes/layout46_style38.png"><div class="label">Layout 46 / Style 38</div></div>
  <div class="scene-item" data-layout="47" data-style="40"><img src="../pretrain_scenes/layout47_style40.png"><div class="label">Layout 47 / Style 40</div></div>
  <div class="scene-item" data-layout="48" data-style="41"><img src="../pretrain_scenes/layout48_style41.png"><div class="label">Layout 48 / Style 41</div></div>
  <div class="scene-item" data-layout="49" data-style="44"><img src="../pretrain_scenes/layout49_style44.png"><div class="label">Layout 49 / Style 44</div></div>
  <div class="scene-item" data-layout="50" data-style="19"><img src="../pretrain_scenes/layout50_style19.png"><div class="label">Layout 50 / Style 19</div></div>
  <div class="scene-item" data-layout="51" data-style="34"><img src="../pretrain_scenes/layout51_style34.png"><div class="label">Layout 51 / Style 34</div></div>
  <div class="scene-item" data-layout="52" data-style="18"><img src="../pretrain_scenes/layout52_style18.png"><div class="label">Layout 52 / Style 18</div></div>
  <div class="scene-item" data-layout="53" data-style="45"><img src="../pretrain_scenes/layout53_style45.png"><div class="label">Layout 53 / Style 45</div></div>
  <div class="scene-item" data-layout="54" data-style="56"><img src="../pretrain_scenes/layout54_style56.png"><div class="label">Layout 54 / Style 56</div></div>
  <div class="scene-item" data-layout="55" data-style="49"><img src="../pretrain_scenes/layout55_style49.png"><div class="label">Layout 55 / Style 49</div></div>
  <div class="scene-item" data-layout="56" data-style="47"><img src="../pretrain_scenes/layout56_style47.png"><div class="label">Layout 56 / Style 47</div></div>
  <div class="scene-item" data-layout="57" data-style="46"><img src="../pretrain_scenes/layout57_style46.png"><div class="label">Layout 57 / Style 46</div></div>
  <div class="scene-item" data-layout="58" data-style="59"><img src="../pretrain_scenes/layout58_style59.png"><div class="label">Layout 58 / Style 59</div></div>
  <div class="scene-item" data-layout="59" data-style="35"><img src="../pretrain_scenes/layout59_style35.png"><div class="label">Layout 59 / Style 35</div></div>
  <div class="scene-item" data-layout="60" data-style="21"><img src="../pretrain_scenes/layout60_style21.png"><div class="label">Layout 60 / Style 21</div></div>
</div>

<script>
// Scene modal variables
let currentScenes = [];
let currentIndex = 0;
let currentGridId = '';

function openModal(gridId, index) {
  currentGridId = gridId;
  const grid = document.getElementById(gridId);
  currentScenes = Array.from(grid.querySelectorAll('.scene-item'));
  currentIndex = index;
  
  const slider = document.getElementById('modalSlider');
  slider.max = currentScenes.length - 1;
  slider.value = currentIndex;
  
  updateModalContent();
  document.getElementById('sceneModal').classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  document.getElementById('sceneModal').classList.remove('active');
  document.body.style.overflow = '';
}

function updateModalContent() {
  const item = currentScenes[currentIndex];
  const img = item.querySelector('img');
  const label = item.querySelector('.label');
  
  document.getElementById('modalImage').src = img.src;
  document.getElementById('modalLabel').innerHTML = label.innerHTML;
  document.getElementById('modalSlider').value = currentIndex;
  document.getElementById('modalCounter').textContent = (currentIndex + 1) + ' / ' + currentScenes.length;
  
  // Update arrow disabled states
  const modalPrev = document.getElementById('sceneModalPrev');
  const modalNext = document.getElementById('sceneModalNext');
  if (modalPrev) modalPrev.disabled = currentIndex <= 0;
  if (modalNext) modalNext.disabled = currentIndex >= currentScenes.length - 1;
}

function navigateModal(direction) {
  const newIndex = currentIndex + direction;
  if (newIndex < 0 || newIndex >= currentScenes.length) return;
  currentIndex = newIndex;
  updateModalContent();
}

function sliderChange(value) {
  currentIndex = parseInt(value);
  updateModalContent();
}

// Close modal on escape key
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeModal();
  if (e.key === 'ArrowLeft') navigateModal(-1);
  if (e.key === 'ArrowRight') navigateModal(1);
});

// Close modal when clicking outside the image
document.getElementById('sceneModal').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});

// Add click handlers to scene items
function initSceneClicks() {
  ['sceneGridTarget', 'sceneGrid'].forEach(function(gridId) {
    const grid = document.getElementById(gridId);
    if (grid) {
      const items = grid.querySelectorAll('.scene-item');
      items.forEach(function(item, index) {
        item.onclick = function() { openModal(gridId, index); };
      });
    }
  });
}

// Initialize on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initSceneClicks);
} else {
  initSceneClicks();
}

function sortScenes(by) {
  const grid = document.getElementById('sceneGrid');
  const items = Array.from(grid.children);
  items.sort((a, b) => parseInt(a.dataset[by]) - parseInt(b.dataset[by]));
  items.forEach(item => grid.appendChild(item));
  document.querySelectorAll('.sort-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
  // Re-initialize click handlers after sorting
  initSceneClicks();
}
</script>
