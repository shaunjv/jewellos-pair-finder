/*
  SIMPLE APP FLOW

  1. The page opens and boot() runs.
  2. boot() asks the backend for all products.
  3. The user chooses a necklace.
  4. loadRecommendations() sends that necklace ID to the backend.
  5. The backend finds matching earrings.
  6. This file receives the results and shows them as cards.
*/
const $ = selector => document.querySelector(selector);
let products;
let currentRecommendation;

// These two functions are called by the buttons in index.html.
// They hide one page section and show the other one.
// The URL also changes, so reloading keeps the user on the same page.
function showView(view) {
  $('#landing').classList.toggle('hidden', view === 'matcher');
  $('#matcher').classList.toggle('hidden', view !== 'matcher');
}

function openMatcher() {
  window.history.pushState({}, '', '/?view=matcher');
  showView('matcher');
}

function openLanding() {
  window.history.pushState({}, '', '/');
  showView('landing');
}

async function api(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Something went wrong.');
  return data;
}

function img(item) {
  // The backend calls this field "image" and returns a local URL such as
  // /images/Nck_1.jpg. Use that URL directly in the browser.
  return `<img src="${item.image}" alt="${item.id}" loading="lazy">`;
}

function card(item, position) {
  // These are the three numbers created by the image models.
  // Showing them lets the user see what helped this earring rank well.
  const details = item.details;
  const signals = details ? `<div class="signals"><p class="signal-title">Model match signals</p><p>Visual detail <b>${Math.round(details.fine_detail * 100)}%</b></p><p>Overall style <b>${Math.round(details.style * 100)}%</b></p><p>Colour and metal <b>${Math.round(details.palette * 100)}%</b></p></div>` : '';
  return `<article class="card"><span class="rank">#${position}</span>${img(item)}<h3>${item.id}</h3><p class="score">${item.score}% match</p>${signals}<p>${item.explanation}</p></article>`;
}

// This is the first function that runs.
// It gets the necklaces and earrings from the backend.
// Then it puts necklaces in the two dropdown boxes.
// It also connects buttons to their functions.
// Finally, it shows matches for the first necklace.
async function boot() {
  products = await api('/api/products');
  for (const select of [$('#necklace-select'), $('#label-necklace')]) {
    select.innerHTML = products.necklaces
      .map(item => `<option value="${item.id}">${item.id} · ${item.image.split('/').pop().replace('.jpg', '')}</option>`)
      .join('');
  }

  $('#necklace-select').onchange = loadRecommendations;
  $('#label-necklace').onchange = renderLabels;
  document.querySelectorAll('.tab').forEach(button => {
    button.onclick = () => switchTab(button.dataset.tab);
  });
  $('#more-button').onclick = showMore;
  $('#save-labels').onclick = saveLabels;
  $('#train').onclick = train;
  await loadRecommendations();
}

// This function changes between the two tabs.
// "recommend" means show matching earrings.
// "learn" means show the rating/training page.
// We load the rating images only when the user opens Learn.
// This helps the first page open faster.
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(item => item.classList.toggle('active', item.dataset.tab === tab));
  $('#recommend').classList.toggle('hidden', tab !== 'recommend');
  $('#learn').classList.toggle('hidden', tab !== 'learn');
  if (tab === 'learn') renderLabels();
}

// This runs whenever the user selects a necklace.
// Example: if the user chooses N01, id is "N01".
// We send N01 to server.py using an API request.
// server.py sends back ranked earrings.
// This function puts the best three earrings on the screen.
async function loadRecommendations() {
  const id = $('#necklace-select').value;
  $('#loading').classList.remove('hidden');
  $('#top-results').innerHTML = '';
  $('#more-results').classList.add('hidden');
  $('#more-button').textContent = 'Show additional earrings';

  try {
    currentRecommendation = await api(`/api/recommendations/${id}`);
    const necklace = products.necklaces.find(item => item.id === id);
    $('#selected').innerHTML = `<div>${img(necklace)}</div><div><h2>Matching methodology</h2><p>The app compares catalogue images locally using fine visual detail, overall style, and colour or metal harmony.</p></div>`;
    $('#top-results').innerHTML = currentRecommendation.top.map((item, index) => card(item, index + 1)).join('');
  } catch (error) {
    $('#top-results').innerHTML = `<p>${error.message}</p>`;
  } finally {
    $('#loading').classList.add('hidden');
  }
}

// The backend also sends lower-ranked earrings.
// We do not show their image cards immediately.
// We show them only after the user clicks this button.
// This makes the first result page feel faster.
function showMore() {
  const area = $('#more-results');
  const isHidden = area.classList.contains('hidden');
  area.classList.toggle('hidden');
  $('#more-button').textContent = isHidden ? 'Hide additional options' : 'Show additional earrings';
  if (isHidden) area.innerHTML = currentRecommendation.remaining.map((item, index) => card(item, index + 4)).join('');
}

// This runs when the user opens the Label and train tab.
// First, it gets old ratings that were already saved.
// Then it shows every earring with a dropdown from 0 to 3.
async function renderLabels() {
  const saved = await api('/api/labels');
  const selected = $('#label-necklace').value;
  // Show the necklace here too. The user needs a reference image while
  // choosing whether each earring is a poor, weak, good, or excellent match.
  const necklace = products.necklaces.find(item => item.id === selected);
  $('#label-reference').innerHTML = `<div>${img(necklace)}</div><div><p class="eyebrow">YOUR REFERENCE ITEM</p><h3>${necklace.id}</h3><p>Use this necklace image while rating every earring below.</p></div>`;
  const values = Object.fromEntries(saved.ratings.filter(item => item.necklace_id === selected).map(item => [item.earring_id, item.quality]));
  $('#label-status').textContent = `${saved.count} labelled pairs saved locally.`;
  $('#label-grid').innerHTML = products.earrings.map(item => `<article class="card">${img(item)}<h3>${item.id}</h3><select class="rating" data-id="${item.id}"><option value="">Not rated</option>${[0, 1, 2, 3].map(value => `<option value="${value}" ${values[item.id] === value ? 'selected' : ''}>${value} — ${['poor', 'weak', 'good', 'excellent'][value]}</option>`).join('')}</select></article>`).join('');
}

// This reads the ratings selected by the user.
// It sends them to server.py.
// server.py saves them in labels/pair_labels.csv.
// Important: this only saves ratings. It does not train yet.
async function saveLabels() {
  const ratings = {};
  document.querySelectorAll('.rating').forEach(item => {
    if (item.value !== '') ratings[item.dataset.id] = Number(item.value);
  });
  const response = await fetch('/api/labels', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ necklace_id: $('#label-necklace').value, ratings }),
  });
  const data = await response.json();
  $('#label-status').textContent = response.ok ? `${data.count} labelled pairs saved locally.` : data.detail;
}

// This sends a "please train" request to server.py.
// The backend reads the saved ratings and trains a small model.
// DINOv2 and SigLIP2 are NOT trained again here.
// They are already trained models used to understand images.
async function train() {
  const response = await fetch('/api/train', { method: 'POST' });
  const data = await response.json();
  $('#label-status').textContent = response.ok ? `Training complete on ${data.label_count} labels.` : data.detail;
}

// The landing-page animation uses this function.
// When a section enters the screen, it becomes visible.
// When it leaves the screen, it can animate again next time.
function setupScrollAnimation() {
  const items = document.querySelectorAll('.reveal');
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => entry.target.classList.toggle('visible', entry.isIntersecting));
  }, { threshold: 0.16 });
  items.forEach(item => observer.observe(item));
}

// Check the URL when the page opens. This lets /?view=matcher open the matcher.
showView(new URLSearchParams(window.location.search).get('view') === 'matcher' ? 'matcher' : 'landing');
setupScrollAnimation();
boot();
