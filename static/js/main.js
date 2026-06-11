const input = document.getElementById('newsInput');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const resultCard = document.getElementById('resultCard');
const errorCard = document.getElementById('errorCard');

input.addEventListener('input', () => {
  const len = input.value.length;
  charCount.textContent = `${len} character${len !== 1 ? 's' : ''}`;
});

function clearInput() {
  input.value = '';
  charCount.textContent = '0 characters';
  resultCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  resultCard.classList.remove('fake-result', 'real-result');
}

async function analyze() {
  const text = input.value.trim();
  if (!text) {
    showError('Please enter some text to analyze.');
    return;
  }
  if (text.length < 20) {
    showError('Text is too short. Please enter at least 20 characters.');
    return;
  }

  setLoading(true);
  resultCard.classList.add('hidden');
  errorCard.classList.add('hidden');

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Something went wrong.');
      return;
    }

    showResult(data);
  } catch (err) {
    showError('Could not connect to the server. Make sure Flask is running.');
  } finally {
    setLoading(false);
  }
}

function showResult(data) {
  const isFake = data.label === 'FAKE';

  resultCard.classList.remove('hidden', 'fake-result', 'real-result');
  resultCard.classList.add(isFake ? 'fake-result' : 'real-result');

  const labelEl = document.getElementById('resultLabel');
  labelEl.textContent = isFake ? '🚨 FAKE NEWS' : '✅ REAL NEWS';
  labelEl.className = 'result-label ' + (isFake ? 'fake' : 'real');

  const confEl = document.getElementById('resultConfidence');
  confEl.innerHTML = `Confidence<strong>${data.confidence}%</strong>`;

  setTimeout(() => {
    document.getElementById('realBar').style.width = data.real_prob + '%';
    document.getElementById('fakeBar').style.width = data.fake_prob + '%';
    document.getElementById('realPct').textContent = data.real_prob + '%';
    document.getElementById('fakePct').textContent = data.fake_prob + '%';
  }, 50);
}

function showError(msg) {
  errorCard.classList.remove('hidden');
  document.getElementById('errorMsg').textContent = msg;
}

function setLoading(state) {
  analyzeBtn.disabled = state;
  btnText.classList.toggle('hidden', state);
  btnLoader.classList.toggle('hidden', !state);
}

input.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') analyze();
});
