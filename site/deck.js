/* Slide engine: one slide at a time, steps inside a slide, hash routing. */
(() => {
  const W = 1280, H = 720;
  const PARTS = ['Start', 'The idea', 'Chunking', 'Embeddings', 'Vector index', 'Hybrid search', 'Generation', 'Your docs', 'RAG vs long context', 'Evaluation'];
  const stage = document.getElementById('stage');
  const slides = [...document.querySelectorAll('.slide')];
  const partsEl = document.getElementById('parts');
  const countEl = document.getElementById('count');
  const overview = document.getElementById('overview');
  const inited = new Set();
  let cur = 0, step = 0;

  window.SPEED = 1;
  window.DEMOS = window.DEMOS || {};
  window.sleep = ms => new Promise(r => setTimeout(r, ms / window.SPEED));

  /* ---------- code blocks: line numbers, light syntax colouring ---------- */
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const TOKEN = /(#.*$)|(f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*')|\b(def|return|for|in|if|else|import|from|as|while|not|and|or|lambda|True|False|None)\b|\b(\d+(?:\.\d+)?)\b|(____)/g;
  const colour = line => {
    let out = '', last = 0, m;
    TOKEN.lastIndex = 0;
    while ((m = TOKEN.exec(line))) {
      out += esc(line.slice(last, m.index));
      const cls = m[1] ? 'c' : m[2] ? 's' : m[3] ? 'k' : m[4] ? 'n' : 'blank';
      out += `<span class="${cls}">${esc(m[0])}</span>`;
      last = m.index + m[0].length;
    }
    return out + esc(line.slice(last));
  };
  document.querySelectorAll('pre.code').forEach(pre => {
    const lines = pre.textContent.replace(/^\n+|\s+$/g, '').split('\n');
    pre.innerHTML = lines.map((l, i) => `<span class="ln" data-n="${i + 1}">${colour(l) || ' '}</span>`).join('');
    const map = {};
    (pre.dataset.hl || '').split(';').filter(Boolean).forEach(part => {
      const [s, spec] = part.split(':');
      map[+s] = new Set();
      spec.split(',').forEach(r => {
        const [a, b] = r.split('-').map(Number);
        for (let n = a; n <= (b || a); n++) map[+s].add(n);
      });
    });
    pre._hl = map;
  });

  /* ---------- steps ---------- */
  const maxStep = s => {
    let m = +(s.dataset.steps || 0);
    s.querySelectorAll('[data-step]').forEach(e => m = Math.max(m, +e.dataset.step));
    s.querySelectorAll('[data-only]').forEach(e => m = Math.max(m, +e.dataset.only));
    s.querySelectorAll('[data-at]').forEach(e => m = Math.max(m, +e.dataset.at));
    s.querySelectorAll('pre.code').forEach(p => Object.keys(p._hl || {}).forEach(k => m = Math.max(m, +k)));
    return m;
  };

  function apply() {
    const s = slides[cur];
    slides.forEach((x, i) => x.classList.toggle('current', i === cur));
    const demo = window.DEMOS[s.id];
    if (demo && !inited.has(s.id)) { inited.add(s.id); demo.init && demo.init(s); }
    s.querySelectorAll('[data-step]').forEach(e => e.classList.toggle('on', step >= +e.dataset.step));
    s.querySelectorAll('[data-only]').forEach(e => e.classList.toggle('on', step === +e.dataset.only));
    s.querySelectorAll('[data-at]').forEach(e => {
      e.classList.toggle('active', step === +e.dataset.at);
      e.classList.toggle('done', step > +e.dataset.at);
    });
    s.querySelectorAll('pre.code').forEach(p => {
      const set = p._hl && p._hl[step];
      p.classList.toggle('has-hl', !!set);
      p.querySelectorAll('.ln').forEach(l => l.classList.toggle('hl', !!set && set.has(+l.dataset.n)));
    });
    if (demo && demo.step) demo.step(step, s);
    if (location.hash !== '#' + s.id) history.replaceState(null, '', '#' + s.id);
    countEl.textContent = `${cur + 1} / ${slides.length}`;
    const part = +s.dataset.part;
    [...partsEl.children].forEach((b, p) => {
      const mine = slides.filter(x => +x.dataset.part === p);
      const done = mine.filter(x => slides.indexOf(x) <= cur).length;
      b.classList.toggle('cur', p === part);
      b.firstChild.style.width = (p < part ? 100 : p > part ? 0 : 100 * done / mine.length) + '%';
    });
  }

  function go(i, st = 0) {
    cur = Math.max(0, Math.min(slides.length - 1, i));
    step = st === 'end' ? maxStep(slides[cur]) : st;
    apply();
  }
  const next = () => step < maxStep(slides[cur]) ? (step++, apply()) : cur < slides.length - 1 && go(cur + 1);
  const prev = () => step > 0 ? (step--, apply()) : cur > 0 && go(cur - 1, 'end');
  window.deck = { go, next, prev, get step() { return step; }, get slide() { return slides[cur].id; } };

  /* ---------- chrome ---------- */
  PARTS.forEach((name, p) => {
    const b = document.createElement('button');
    b.innerHTML = `<i></i><span>${name}</span>`;
    b.onclick = () => go(slides.findIndex(x => +x.dataset.part === p));
    partsEl.appendChild(b);
  });
  document.getElementById('prev').onclick = prev;
  document.getElementById('next').onclick = next;

  function fit() {
    const vp = document.getElementById('viewport').getBoundingClientRect();
    const k = Math.min(vp.width / W, vp.height / H) * 0.985;
    stage.style.transform = `scale(${k}) translate(-50%, -50%)`;
  }
  addEventListener('resize', fit);
  if (window.ResizeObserver) new ResizeObserver(fit).observe(document.getElementById('viewport'));

  function toggleOverview(force) {
    const open = force ?? !overview.classList.contains('open');
    overview.classList.toggle('open', open);
    if (!open) return;
    overview.innerHTML = '<p class="small">Click a slide to jump to it. <kbd>O</kbd> or <kbd>Esc</kbd> closes.</p>';
    PARTS.forEach((name, p) => {
      const mine = slides.filter(x => +x.dataset.part === p);
      if (!mine.length) return;
      const h = document.createElement('h3'); h.textContent = name; overview.appendChild(h);
      const g = document.createElement('div'); g.className = 'grid'; overview.appendChild(g);
      mine.forEach(x => {
        const b = document.createElement('button');
        const i = slides.indexOf(x);
        b.textContent = `${i + 1}. ${(x.querySelector('h1,h2') || {}).textContent || x.id}`;
        b.classList.toggle('cur', i === cur);
        b.onclick = () => { toggleOverview(false); go(i); };
        g.appendChild(b);
      });
    });
  }

  const setTheme = t => { document.documentElement.dataset.theme = t; try { localStorage.setItem('rag-theme', t); } catch (e) {} };
  try { setTheme(localStorage.getItem('rag-theme') || 'light'); } catch (e) { setTheme('light'); }
  document.getElementById('theme').onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  document.getElementById('ovbtn').onclick = () => toggleOverview();
  const speedBtn = document.getElementById('speed');
  speedBtn.onclick = () => {
    window.SPEED = window.SPEED === 1 ? 2 : window.SPEED === 2 ? 0.5 : 1;
    speedBtn.textContent = `speed ${window.SPEED}×`;
  };

  addEventListener('keydown', e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    if (e.key === 'Escape') { toggleOverview(false); e.target.blur && e.target.blur(); return; }
    if (typing) return;
    const k = e.key;
    if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'Enter') { if (e.target.tagName === 'BUTTON' && (k === ' ' || k === 'Enter')) return; e.preventDefault(); next(); }
    else if (k === 'ArrowLeft' || k === 'PageUp' || k === 'Backspace') { e.preventDefault(); prev(); }
    else if (k === 'Home') go(0);
    else if (k === 'End') go(slides.length - 1);
    else if (k === 'o' || k === 'O') toggleOverview();
    else if (k === 't' || k === 'T') document.getElementById('theme').click();
    else if (k === 'f' || k === 'F') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  });

  let tx = null;
  addEventListener('touchstart', e => { tx = e.target.closest('input,button,select,.panel') ? null : e.touches[0].clientX; }, { passive: true });
  addEventListener('touchend', e => {
    if (tx === null) return;
    const dx = e.changedTouches[0].clientX - tx;
    if (Math.abs(dx) > 50) dx < 0 ? next() : prev();
    tx = null;
  });

  /* ---------- quizzes ---------- */
  document.querySelectorAll('.quiz').forEach(q => {
    const right = +q.dataset.answer;
    q.querySelectorAll('.opts button').forEach((b, i) => b.onclick = () => {
      q.querySelectorAll('.opts button').forEach(x => x.classList.remove('wrong'));
      b.classList.add(i === right ? 'right' : 'wrong');
      if (i === right) q.classList.add('answered');
    });
  });

  const fromHash = () => {
    const i = slides.findIndex(x => '#' + x.id === location.hash);
    if (i >= 0 && i !== cur) go(i);
  };
  addEventListener('hashchange', fromHash);
  fit();
  const start = slides.findIndex(x => '#' + x.id === location.hash);
  go(start >= 0 ? start : 0);
})();
