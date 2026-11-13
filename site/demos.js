/* Interactive demos. All numbers come from data.js (built by tools/build_data.py). */
(() => {
  const D = window.RAG_DATA;
  const N = D.chunks.length;
  const DEMOS = window.DEMOS = window.DEMOS || {};
  const $ = (s, r = document) => r.querySelector(s);
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const title = c => c.text.split('. ')[0];
  const idx = id => D.chunks.findIndex(c => c.id === id);
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const order = scores => scores.map((s, i) => [s, i]).sort((x, y) => y[0] - x[0] || x[1] - y[1]).map(x => x[1]);
  const SVG = (tag, attrs = {}) => {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };

  /* ---------- search maths (mirrors the notebook) ---------- */
  const STOP = new Set('a an the i my me you your is are am do does did can could what when where which who how if in on at to for of and or it be with from that this until much many there have has'.split(' '));
  const tokenize = t => (t.toLowerCase().match(/[a-z]+-\d+|[a-z0-9]+/g) || []).filter(w => !STOP.has(w));
  const docs = D.chunks.map(c => tokenize(c.text));
  const avgdl = docs.reduce((s, d) => s + d.length, 0) / N;
  const tf = docs.map(d => { const m = {}; d.forEach(w => m[w] = (m[w] || 0) + 1); return m; });
  const idf = {};
  (() => {
    const df = {};
    tf.forEach(m => Object.keys(m).forEach(w => df[w] = (df[w] || 0) + 1));
    let sum = 0; const neg = [];
    for (const w in df) {
      idf[w] = Math.log(N - df[w] + 0.5) - Math.log(df[w] + 0.5);
      sum += idf[w];
      if (idf[w] < 0) neg.push(w);
    }
    const eps = 0.25 * sum / Object.keys(df).length;
    neg.forEach(w => idf[w] = eps);
  })();
  const bm25 = q => {
    const scores = new Array(N).fill(0), k1 = 1.5, b = 0.75;
    tokenize(q).forEach(w => {
      if (!(w in idf)) return;
      for (let i = 0; i < N; i++) {
        const f = tf[i][w] || 0;
        scores[i] += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * docs[i].length / avgdl));
      }
    });
    return scores;
  };
  const semantic = qe => D.emb.map(e => dot(e, qe));
  const rrf = (rankings, weights) => {
    const pts = new Map();
    rankings.forEach((r, j) => r.forEach((i, pos) => pts.set(i, (pts.get(i) || 0) + weights[j] / (60 + pos + 1))));
    return [...pts.entries()].sort((a, b) => b[1] - a[1]);
  };
  window.RAG = { bm25, semantic, rrf, tokenize, order };

  /* ---------- optional live embedding model ---------- */
  let extractor = null;
  async function loadLive(btn) {
    btn.disabled = true; btn.textContent = 'loading model…';
    try {
      const m = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2');
      extractor = await m.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
      document.querySelectorAll('.livebtn').forEach(b => { b.textContent = 'live model ready'; b.disabled = true; });
      document.querySelectorAll('.textin').forEach(t => { t.disabled = false; t.placeholder = 'type any question, press Enter'; });
    } catch (e) {
      btn.textContent = 'could not load (offline?)'; btn.disabled = false;
    }
  }
  const embedLive = async text => Array.from((await extractor(text, { pooling: 'mean', normalize: true })).data);

  /* query bar: prepared question chips + optional typed question */
  const SHORT = { retake: 'retake an exam', cs240: 'CS-240 semester', late: '3 days late', books: 'borrow books', quiet: 'quiet time',
    stress: 'feeling stressed', scholarship: 'keep scholarship', guests: 'friend sleeps over', password: 'forgot password',
    drop: 'drop a course', ai: 'use ChatGPT', parking: 'parking permit' };
  function queryBar(root, ids, onQuery, typed = true, short = false) {
    root.innerHTML = `<div class="chips"></div>` + (short ? '<p class="qtext"></p>' : '') + (typed ? `<div class="row" style="gap:8px;margin-top:8px">
      <input class="textin" disabled placeholder="load the live model to type your own">
      <button class="btn livebtn">type your own</button></div>` : '');
    const chips = $('.chips', root);
    const pick = (q, chip) => {
      chips.querySelectorAll('.chip').forEach(c => c.classList.toggle('sel', c === chip));
      if (short) $('.qtext', root).textContent = '“' + q.text + '”';
      onQuery(q);
    };
    ids.forEach(id => {
      const q = D.queries.find(x => x.id === id);
      const c = document.createElement('button');
      c.className = 'chip'; c.textContent = short ? SHORT[id] : q.text; c.onclick = () => pick(q, c);
      chips.appendChild(c);
    });
    if (typed) {
      const btn = $('.livebtn', root), inp = $('.textin', root);
      if (extractor) { btn.textContent = 'live model ready'; btn.disabled = true; inp.disabled = false; }
      btn.onclick = () => loadLive(btn);
      inp.onkeydown = async e => {
        if (e.key !== 'Enter' || !inp.value.trim() || !extractor) return;
        pick({ id: 'typed', text: inp.value.trim(), emb: await embedLive(inp.value.trim()), gold: null }, null);
      };
    }
    return { first: () => chips.firstChild.click() };
  }

  const rankRow = (i, score, max, gold, digits = 3) => `<div class="rankrow${gold === D.chunks[i].id ? ' gold' : ''}">
    <i class="bar" style="width:${Math.max(0, 100 * score / max)}%"></i><span class="id" style="position:relative">${D.chunks[i].id}</span>
    <span class="ttl">${esc(title(D.chunks[i]))}</span><span class="sc">${score.toFixed(digits)}</span></div>`;

  /* ---------- 2D map of chunks ---------- */
  const MW = 620, MH = 470, PAD = 34;
  const px = i => [PAD + D.xy[i][0] * (MW - 2 * PAD), PAD + D.xy[i][1] * (MH - 2 * PAD)];
  function makeMap(svg, onHover) {
    svg.setAttribute('viewBox', `0 0 ${MW} ${MH}`);
    const gLinks = svg.appendChild(SVG('g')), gDots = svg.appendChild(SVG('g')), gQ = svg.appendChild(SVG('g'));
    const dots = D.chunks.map((c, i) => {
      const [x, y] = px(i);
      const d = SVG('circle', { class: 'dot', cx: x, cy: y, r: 7 });
      d.addEventListener('mouseenter', () => onHover && onHover(i));
      d.appendChild(SVG('title')).textContent = `${c.id} · ${title(c)}`;
      return gDots.appendChild(d);
    });
    const place = sims => {
      const top = order(sims).slice(0, 3);
      let wx = 0, wy = 0, ws = 0;
      top.forEach(i => { const w = Math.exp((sims[i] - sims[top[0]]) / 0.05), [x, y] = px(i); wx += w * x; wy += w * y; ws += w; });
      return [wx / ws + 16, wy / ws - 16];
    };
    return {
      svg, dots, gQ, gLinks,
      clear() { gLinks.innerHTML = ''; gQ.innerHTML = ''; dots.forEach(d => d.setAttribute('class', 'dot')); },
      showQuery(sims, k = 3) {
        this.clear();
        const [qx, qy] = place(sims), top = order(sims).slice(0, k);
        top.forEach((i, r) => {
          const [x, y] = px(i);
          gLinks.appendChild(SVG('line', { class: 'link', x1: qx, y1: qy, x2: x, y2: y }));
          dots[i].setAttribute('class', 'dot ' + (r === 0 ? 'best' : 'top'));
        });
        gQ.appendChild(SVG('path', { class: 'qstar', d: star(qx, qy, 12) }));
        const t = gQ.appendChild(SVG('text', { class: 'qlabel', x: qx + 15, y: qy - 10 })); t.textContent = 'your question';
        return [qx, qy];
      },
    };
  }
  const star = (cx, cy, r) => {
    let d = '';
    for (let i = 0; i < 10; i++) {
      const a = Math.PI / 5 * i - Math.PI / 2, rr = i % 2 ? r * 0.45 : r;
      d += (i ? 'L' : 'M') + (cx + rr * Math.cos(a)).toFixed(1) + ' ' + (cy + rr * Math.sin(a)).toFixed(1);
    }
    return d + 'Z';
  };

  /* ================= Part 1 ================= */
  const clip = (s0, n) => {
    const s = s0.replace(/^\s*\*\s+/gm, '• ').replace(/\*/g, '').replace(/\n{2,}/g, '\n');
    return s.length > n ? s.slice(0, n).replace(/\s+\S*$/, '') + ' …' : s;
  };
  DEMOS['ask'] = {
    init(s) { $('#ask-answer', s).textContent = clip(D.gen['retake|none'] || '(run tools/build_data.py --gen to record the model answer)', 520); },
  };
  DEMOS['recap'] = {
    init(s) {
      const c = $('#pipe-svg').cloneNode(true);
      c.removeAttribute('id'); c.classList.add('lit');
      c.querySelectorAll('[data-at]').forEach(e => { e.removeAttribute('data-at'); e.classList.remove('active', 'done'); });
      $('#recap-pipe', s).appendChild(c);
    },
  };

  /* ================= Part 2: chunking ================= */
  DEMOS['chunk-demo'] = {
    init(s) {
      const paras = ['Grade scale', 'Late work', 'Extensions'].map(t => D.chunks.find(c => c.text.startsWith(t)).text);
      const text = paras.join('\n\n');
      const A = text.indexOf('loses 10 percent'), B = text.indexOf('receives a mark of zero') + 23;
      const root = $('#chunk-root', s);
      root.innerHTML = `
        <div class="controls">
          <label>split by <select id="ck-mode"><option value="fixed">fixed size (characters)</option><option value="sent">sentences</option><option value="para">paragraphs</option><option value="rec">recursive (LangChain)</option></select></label>
          <label>chunk size <input type="range" id="ck-size" min="60" max="700" step="10" value="150"><output id="ck-size-o"></output></label>
          <label>overlap <input type="range" id="ck-ov" min="0" max="200" step="10" value="0"><output id="ck-ov-o"></output></label>
        </div>
        <div class="panel grow"><div class="chunktext" id="ck-text" style="white-space:pre-wrap"></div></div>
        <div class="stat"><span><b id="ck-n"></b> chunks</span><span>average <b id="ck-avg"></b> characters</span>
          <span style="margin-left:auto">The late-work rule (10% per day, zero after five days):</span><span class="verdict" id="ck-verdict"></span></div>`;
      root.style.cssText = 'display:flex;flex-direction:column;gap:12px;min-height:0';
      const mode = $('#ck-mode', s), size = $('#ck-size', s), ov = $('#ck-ov', s);
      const recLookup = (sz, o) => {            // real output of LangChain's splitter, computed ahead of time
        const R = D.chunkRec, snap = (v, [a, b, st]) => Math.max(a, Math.min(b, a + Math.round((v - a) / st) * st));
        const s2 = snap(sz, R.sizes), o2 = Math.floor(Math.min(snap(o, R.ovs), s2 / 2) / R.ovs[2]) * R.ovs[2];
        return R.grid[s2 + '|' + o2] || [];
      };
      const ranges = () => {
        const out = [], sz = +size.value;
        if (mode.value === 'rec') return recLookup(sz, +ov.value);
        if (mode.value === 'fixed') {
          const st = Math.max(10, sz - Math.min(+ov.value, sz - 10));
          for (let a = 0; a < text.length; a += st) out.push([a, Math.min(text.length, a + sz)]);
        } else if (mode.value === 'para') {
          let a = 0; paras.forEach(p => { out.push([a, a + p.length]); a += p.length + 2; });
        } else {
          const re = /[^.!?]+[.!?]+\s*/g; let m, a = null, end = 0;
          while ((m = re.exec(text))) {
            if (a !== null && m.index + m[0].length - a > sz) { out.push([a, end]); a = null; }
            if (a === null) a = m.index;
            end = m.index + m[0].length;
          }
          if (a !== null) out.push([a, text.length]);
        }
        return out;
      };
      const draw = () => {
        const fixed = mode.value === 'fixed';
        const rec = mode.value === 'rec';
        size.disabled = mode.value === 'para'; ov.disabled = !(fixed || rec);
        $('#ck-size-o', s).textContent = size.value;
        $('#ck-ov-o', s).textContent = fixed ? ov.value : rec ? Math.floor(Math.min(+ov.value, +size.value / 2) / 20) * 20 : '–';
        const r = ranges();
        const cuts = [...new Set(r.flat())].sort((a, b) => a - b);
        let html = '';
        for (let j = 0; j < cuts.length - 1; j++) {
          const a = cuts[j], b = cuts[j + 1];
          const cover = r.map((x, n) => x[0] <= a && x[1] >= b ? n : -1).filter(n => n >= 0);
          const cls = cover.length > 1 ? 'ckov' : cover.length ? 'ck' + cover[0] % 4 : '';
          html += `<span class="${cls}">${esc(text.slice(a, b))}</span>`;
        }
        $('#ck-text', s).innerHTML = html;
        $('#ck-n', s).textContent = r.length;
        $('#ck-avg', s).textContent = Math.round(r.reduce((t, x) => t + x[1] - x[0], 0) / r.length);
        const ok = r.some(x => x[0] <= A && x[1] >= B);
        const v = $('#ck-verdict', s);
        v.className = 'verdict ' + (ok ? 'ok' : 'no');
        v.textContent = ok ? '✓ stays in one chunk' : '✗ cut across chunks';
      };
      [mode, size, ov].forEach(e => e.oninput = draw);
      draw();
    },
  };

  /* ================= Part 2: recursive splitting, animated ================= */
  DEMOS['chunk-rec'] = {
    init(s) {
      /* The same algorithm as LangChain's RecursiveCharacterTextSplitter (no overlap), written out so we can
         show its rounds. window.RECDEMO.check() compares it with the library's real output. */
      const T = D.recDemo.text;
      const SEPS = ['\n\n', '\n', '. ', ' ', ''], NAMES = ['blank line', 'line break', 'full stop', 'space', 'anywhere'];
      const size = $('#rc-size', s);
      $('#rc-ladder', s).innerHTML = NAMES.map((n, i) => `${i ? '<em>→</em>' : ''}<span class="recstep" data-i="${i}"><b>${i + 1}</b><i>${n}</i></span>`).join('');

      const pick = (a, b, seps) => {                 // first separator in the list that occurs in this piece
        for (let i = 0; i < seps.length; i++) if (seps[i] === '' || T.slice(a, b).includes(seps[i])) return [seps[i], seps.slice(i + 1)];
        return ['', []];
      };
      const cutAt = (a, b, sep) => {                 // the separator stays at the end of the piece before it
        const out = [];
        if (sep === '') { for (let i = a; i < b; i++) out.push([i, i + 1]); return out; }
        let from = a, at;
        while ((at = T.indexOf(sep, from)) !== -1 && at < b) { out.push([from, at + sep.length]); from = at + sep.length; }
        if (from < b) out.push([from, b]);
        return out.filter(x => x[1] > x[0]);
      };
      const trim = ([a, b]) => { while (a < b && /\s/.test(T[a])) a++; while (b > a && /\s/.test(T[b - 1])) b--; return [a, b]; };
      const merge = (good, max) => {                 // glue neighbours while they stay within the maximum
        const out = []; let cur = [], total = 0;
        const flush = () => { if (cur.length) { const r = trim([cur[0][0], cur[cur.length - 1][1]]); if (r[1] > r[0]) out.push(r); } cur = []; total = 0; };
        good.forEach(d => { const len = d[1] - d[0]; if (total + len > max && cur.length) flush(); cur.push(d); total += len; });
        flush();
        return out;
      };
      const split = (a, b, seps, max) => {
        const [sep, rest] = pick(a, b, seps); let good = [], out = [];
        cutAt(a, b, sep).forEach(p => {
          if (p[1] - p[0] < max) good.push(p);
          else { if (good.length) { out = out.concat(merge(good, max)); good = []; } out = out.concat(rest.length ? split(p[0], p[1], rest, max) : [p]); }
        });
        if (good.length) out = out.concat(merge(good, max));
        return out;
      };
      /* the rounds shown on screen: every piece that is still too long is cut with its next separator */
      const rounds = max => {
        let pieces = [{ r: [0, T.length], seps: SEPS, long: T.length >= max }];
        const out = [{ kind: 'start', pieces, used: [] }];
        while (pieces.some(p => p.long)) {
          const used = new Set(), next = [], fresh = new Set();
          pieces.forEach(p => {
            if (!p.long) { next.push(p); return; }
            const [sep, rest] = pick(p.r[0], p.r[1], p.seps);
            const parts = cutAt(p.r[0], p.r[1], sep).filter(r => T.slice(r[0], r[1]).trim());   // ignore pieces that are only white space
            if (parts.length > 1) used.add(SEPS.indexOf(sep));
            parts.forEach((r, k) => { if (k) fresh.add(r[0]); next.push({ r, seps: rest, long: r[1] - r[0] >= max && rest.length > 0 }); });
          });
          pieces = next;
          if (used.size) out.push({ kind: 'round', pieces, used: [...used].sort(), fresh });   // show a round only if it made a visible cut
        }
        out.push({ kind: 'final', chunks: split(0, T.length, SEPS, max), used: [] });
        return out;
      };
      window.RECDEMO = { check: () => Object.entries(D.recDemo.real).filter(([sz, real]) => JSON.stringify(split(0, T.length, SEPS, +sz)) !== JSON.stringify(real)).map(x => x[0]) };

      let R = [], at = 0, token = 0;
      const listRow = (r, cls, n) => `<div class="${cls}"><i style="font-style:normal;color:var(--muted)">${n}</i><span>${esc(T.slice(r[0], r[1]).trim())}</span><b>${r[1] - r[0]}</b></div>`;
      const draw = () => {
        const max = +size.value, st = R[at], seen = new Set(R.slice(0, at + 1).flatMap(x => x.used));
        $('#rc-size-o', s).textContent = max + ' characters';
        s.querySelectorAll('#rc-ladder .recstep').forEach(e => { const i = +e.dataset.i; e.classList.toggle('now', st.used.includes(i)); e.classList.toggle('used', seen.has(i)); });
        const cap = $('#rc-cap', s);
        if (st.kind === 'final') {
          $('#rc-text', s).innerHTML = st.chunks.map((r, n) => `<span class="rp ck${n % 4}">${esc(T.slice(r[0], r[1]))}</span>`).join('<i class="cut"></i>');
          $('#rc-title', s).textContent = `Result: ${st.chunks.length} chunks`;
          $('#rc-list', s).innerHTML = st.chunks.slice(0, 9).map((r, n) => listRow(r, 'ok', n + 1)).join('') + (st.chunks.length > 9 ? `<p>… and ${st.chunks.length - 9} more</p>` : '');
          cap.className = 'card hit';
          cap.innerHTML = `<b>Glue.</b> Nothing is too long any more. Small neighbours are joined back together as long as they stay within ${max} characters. Each colour is one finished chunk.`;
          return;
        }
        const P = st.pieces, bad = P.filter(p => p.long).length;
        $('#rc-text', s).innerHTML = P.map((p, n) => (n ? `<i class="cut${st.fresh && st.fresh.has(p.r[0]) ? ' new' : ''}"></i>` : '') + `<span class="rp ${p.long ? 'long' : 'ok'}">${esc(T.slice(p.r[0], p.r[1]))}</span>`).join('');
        $('#rc-title', s).textContent = `${P.length} piece${P.length > 1 ? 's' : ''} · ${bad} too long`;
        const show = P.length > 9 ? P.filter(p => p.long).slice(0, 4).concat(P.filter(p => !p.long).slice(0, 5)) : P;
        $('#rc-list', s).innerHTML = show.map(p => listRow(p.r, p.long ? 'long' : 'ok', P.indexOf(p) + 1)).join('') + (P.length > 9 ? `<p>… ${P.length} pieces in total</p>` : '');
        if (st.kind === 'start') {
          cap.className = 'card ' + (bad ? 'bad' : 'hit');
          cap.innerHTML = bad ? `<b>Start.</b> The whole text is one piece of ${T.length} characters. The maximum is ${max}, so it is too long and must be split.`
            : `<b>Start.</b> The whole text is ${T.length} characters and the maximum is ${max}, so it already fits.`;
        } else {
          cap.className = 'card ' + (bad ? 'work' : 'hit');
          const names = st.used.map(i => NAMES[i]).join(' or ');
          cap.innerHTML = `<b>Split at every ${names}</b>, but only inside the pieces that were too long (red). ` +
            (bad ? `${bad} piece${bad > 1 ? 's are' : ' is'} still longer than ${max}, so the splitter moves down its list.` : `Every piece now fits within ${max}.`);
        }
      };
      const reset = () => { token++; R = rounds(+size.value); at = 0; draw(); };
      const step = () => { if (at < R.length - 1) { at++; draw(); return true; } return false; };
      $('#rc-step', s).onclick = () => { token++; if (!step()) reset(); };
      $('#rc-reset', s).onclick = reset;
      $('#rc-play', s).onclick = async () => { reset(); const my = token; while (my === token && at < R.length - 1) { await sleep(1700); if (my !== token) return; step(); } };
      size.oninput = reset;
      reset();
    },
  };

  /* ================= Part 3: embeddings ================= */
  DEMOS['emb-why'] = {
    init(s) {
      const q = D.queries.find(x => x.id === 'retake'), kw = bm25(q.text), top = order(kw);
      $('#why-kw', s).innerHTML = top.slice(0, 3).map(i => rankRow(i, kw[i], kw[top[0]], q.gold, 2)).join('');
      $('#why-kw-note', s).innerHTML = `Rules that contain the word "exam". The resit rule is at position <b>${top.indexOf(idx(q.gold)) + 1}</b> of ${N}.`;
    },
  };
  DEMOS['emb-idea'] = {
    init(s) {
      D.simsets.forEach(st => st.sentences.forEach(x => { if (Math.abs(x.score) < 0.005) x.score = 0; }));
      const sets = D.simsets, chips = $('#sim-chips', s), rows = $('#sim-rows', s), fan = $('#sim-fan', s);
      const OX = 46, OY = 310, L = 205, ROWH = 42;
      let cur = 0, shown = false;
      const col = sc => sc >= 0.5 ? 'var(--hit)' : sc >= 0.25 ? 'var(--work)' : 'var(--muted)';
      const tip = (ang, len) => [OX + len * Math.cos(ang), OY - len * Math.sin(ang)];
      const draw = () => {
        const set = sets[cur], sent = set.sentences;
        chips.querySelectorAll('.chip').forEach((c, i) => c.classList.toggle('sel', i === cur));
        $('#sim-query', s).textContent = set.query;
        const first = set.vec.slice(0, 8), mx = Math.max(...first.map(Math.abs));
        $('#sim-vec', s).className = 'vcells';
        $('#sim-vec', s).innerHTML = '<span class="mono">[</span>' + first.map(x => `<b style="background:color-mix(in srgb, var(--${x < 0 ? 'bad' : 'hit'}) ${Math.round(12 + 45 * Math.abs(x) / mx)}%, transparent)">${x.toFixed(2)}</b>`).join('') + '<span class="mono">… ]</span>';
        const rank = sent.map((_, i) => i).sort((x, y) => sent[y].score - sent[x].score);
        rows.style.position = 'relative';
        rows.innerHTML = sent.map((t, i) => {
          const pos = shown ? rank.indexOf(i) : i;
          return `<div class="rankrow simrow" data-i="${i}" style="position:absolute;left:0;right:0;top:0;height:${ROWH - 7}px;grid-template-columns:1fr 64px;transform:translateY(${pos * ROWH}px);transition:transform .7s cubic-bezier(.2,.8,.2,1)">
            <i class="bar" style="width:${shown ? Math.max(0, 100 * t.score) : 0}%;transition:width .7s;background:${shown ? 'color-mix(in srgb, ' + col(t.score) + ' 22%, transparent)' : 'transparent'}"></i>
            <span class="ttl" style="font-size:16.5px">${esc(t.text)}</span><span class="sc" style="font-size:19px;font-weight:700;color:${shown ? col(t.score) : 'var(--muted)'}">${shown ? t.score.toFixed(2) : '?'}</span></div>`;
        }).join('');
        let g = `<path d="M${OX} ${OY}H${OX + L + 20}" stroke="var(--line)" stroke-width="1.5" stroke-dasharray="4 5"/>`;
        /* place each score label at its arrow tip, then push it outward until it touches no other label */
        const LR = {}, placed = [];
        sent.map((x, i) => [Math.acos(Math.max(-1, Math.min(1, x.score))), i]).sort((p, q) => p[0] - q[0]).forEach(([ang, i]) => {
          let r = L + 6, pos = tip(ang, r);
          for (let n = 0; n < 8 && placed.some(q => Math.abs(q[0] - pos[0]) < 40 && Math.abs(q[1] - pos[1]) < 22); n++) { r += 23; pos = tip(ang, r); }
          placed.push(pos); LR[i] = r;
        });
        if (shown) sent.forEach((t, i) => {
          const ang = Math.acos(Math.max(-1, Math.min(1, t.score))), [x, y] = tip(ang, L - 24), [lx, ly] = tip(ang, LR[i]);
          g += `<g class="fanarrow" data-i="${i}" style="transition:opacity .2s"><line x1="${OX}" y1="${OY}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${col(t.score)}" stroke-width="4" stroke-linecap="round"/>
            <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${col(t.score)}"/>
            <text x="${lx.toFixed(1)}" y="${(ly + 5).toFixed(1)}" font-size="15" font-weight="700" fill="${col(t.score)}" text-anchor="middle">${t.score.toFixed(2)}</text></g>`;
        });
        g += `<line x1="${OX}" y1="${OY}" x2="${OX + L}" y2="${OY}" stroke="var(--res)" stroke-width="6" stroke-linecap="round"/>
          <path d="M${OX + L + 12} ${OY}l-16 -9v18z" fill="var(--res)"/><text x="${OX + L - 4}" y="${OY + 26}" font-size="16" font-weight="700" fill="var(--res)" text-anchor="end">the query</text>`;
        if (!shown) g += `<text x="${OX + 130}" y="${OY - 120}" font-size="17" fill="var(--muted)" text-anchor="middle">six arrows appear here</text>`;
        fan.innerHTML = g;
        rows.querySelectorAll('.simrow').forEach(r => {
          r.onmouseenter = () => {
            fan.querySelectorAll('.fanarrow').forEach(a => a.style.opacity = a.dataset.i === r.dataset.i ? 1 : .15);
            const t = sent[+r.dataset.i];
            if (shown) $('#sim-note', s).textContent = `cos(${Math.round(Math.acos(Math.max(-1, Math.min(1, t.score))) * 180 / Math.PI)}°) = ${t.score.toFixed(2)}`;
          };
          r.onmouseleave = () => {
            fan.querySelectorAll('.fanarrow').forEach(a => a.style.opacity = 1);
            $('#sim-note', s).textContent = 'cosine similarity = cos(angle). Same direction → 1, unrelated → near 0.';
          };
        });
      };
      sets.forEach((set, i) => {
        const c = document.createElement('button');
        c.className = 'chip'; c.textContent = set.query; c.onclick = () => { cur = i; draw(); };
        chips.appendChild(c);
      });
      this.show = v => { if (v !== shown) { shown = v; draw(); if (v) requestAnimationFrame(() => {}); } };
      draw();
    },
    step(n) { this.show(n >= 1); },
  };
  DEMOS['emb-demo'] = {
    init(s) {
      const hover = $('#emb-hover', s);
      const map = makeMap($('#emb-map', s), i => { hover.innerHTML = `<b>${D.chunks[i].id}</b> · ${esc(D.chunks[i].text)}`; });
      const bar = queryBar($('#emb-q', s), ['retake', 'stress', 'guests', 'ai', 'password', 'books'], q => {
        const sims = semantic(q.emb), top = order(sims).slice(0, 4);
        map.showQuery(sims, 3);
        $('#emb-rank', s).innerHTML = top.map(i => rankRow(i, sims[i], sims[top[0]], q.gold)).join('');
        hover.innerHTML = `<b>${D.chunks[top[0]].id}</b> · ${esc(D.chunks[top[0]].text)}`;
      }, true, true);
      this.bar = bar;
    },
    step(n) { if (n === 1 && !this.ran) { this.ran = true; this.bar.first(); } },
  };

  /* ================= Part 4: index ================= */
  const CL = ['#e39a22', '#14917f', '#6452d9', '#cf4257', '#3b82c4', '#8a8f2a'];
  DEMOS['idx-demo'] = {
    init(s) {
      const K = D.centroids.length;
      const map = makeMap($('#idx-map', s));
      const paint = () => map.dots.forEach((d, i) => { d.setAttribute('class', 'dot'); d.style.fill = CL[D.cluster[i]]; d.style.opacity = .8; });
      const rings = D.centroids.map((_, c) => {
        const m = D.chunks.map((_, i) => i).filter(i => D.cluster[i] === c).map(px);
        const cx = m.reduce((t, p) => t + p[0], 0) / m.length, cy = m.reduce((t, p) => t + p[1], 0) / m.length;
        const g = SVG('g');
        g.appendChild(SVG('rect', { class: 'cent', x: cx - 9, y: cy - 9, width: 18, height: 18, rx: 3, stroke: CL[c], transform: `rotate(45 ${cx} ${cy})` }));
        map.svg.insertBefore(g, map.gQ);
        return g.firstChild;
      });
      paint();
      let q = null, token = 0;
      const np = $('#idx-np', s), run = $('#idx-run', s);
      const reset = () => {
        token++; paint(); map.gLinks.innerHTML = ''; map.gQ.innerHTML = '';
        rings.forEach(r => r.setAttribute('fill', 'none'));
        $('#idx-exact', s).textContent = N; $('#idx-ivf', s).textContent = '0';
        $('#idx-out', s).innerHTML = '<p class="small">Pick a question, choose how many clusters to open, press Run.</p>';
        $('#idx-np-o', s).textContent = np.value;
      };
      queryBar($('#idx-q', s), ['guests', 'stress', 'retake', 'late', 'ai'], x => { q = x; reset(); run.disabled = false; }, false, true);
      np.oninput = reset;
      run.onclick = async () => {
        reset(); const my = token;
        const sims = semantic(q.emb), best = order(sims)[0];
        const probe = order(D.centroids.map(c => dot(c, q.emb))).slice(0, +np.value);
        let count = 0; const out = $('#idx-ivf', s);
        const starAt = (() => { const [x, y] = px(best); return [x + 16, y - 16]; })();
        map.gQ.appendChild(SVG('path', { class: 'qstar', d: star(starAt[0], starAt[1], 12) }));
        map.dots.forEach(d => d.style.opacity = .15);
        for (let c = 0; c < K; c++) {              // step 1: compare with the cluster centres
          rings[c].setAttribute('fill', CL[c]); out.textContent = ++count; await sleep(170);
          if (my !== token) return;
          if (!probe.includes(c)) rings[c].setAttribute('fill', 'none');
        }
        let found = -1;
        for (const c of probe) for (let i = 0; i < N; i++) {   // step 2: scan only the opened clusters
          if (D.cluster[i] !== c) continue;
          map.dots[i].style.opacity = 1; out.textContent = ++count; await sleep(70);
          if (my !== token) return;
          if (found < 0 || sims[i] > sims[found]) found = i;
        }
        map.dots[found].setAttribute('class', 'dot best'); map.dots[found].style.fill = '';
        const hit = found === best;
        $('#idx-out', s).innerHTML = `<div class="rank">
          <div class="small">Exact search finds</div>${rankRow(best, sims[best], sims[best], null)}
          <div class="small">The index finds</div>${rankRow(found, sims[found], sims[best], null)}</div>
          <p style="margin-top:10px"><span class="verdict ${hit ? 'ok' : 'no'}">${hit ? '✓ same answer, ' + count + ' comparisons instead of ' + N : '✗ missed the best chunk: it sits in a cluster we did not open'}</span></p>`;
      };
      reset(); run.disabled = true;
    },
  };

  /* ================= Part 5: hybrid ================= */
  DEMOS['hyb-why'] = {
    init(s) {
      const q = D.queries.find(x => x.id === 'cs240'), sims = semantic(q.emb), top = order(sims).slice(0, 4);
      $('#hyb-why-q', s).textContent = q.text;
      $('#hyb-why-rank', s).innerHTML = top.map(i => rankRow(i, sims[i], sims[top[0]], q.gold)).join('');
    },
  };
  DEMOS['hyb-demo'] = {
    init(s) {
      const ROW = 43, flip = $('#hyb-fused', s), mix = $('#hyb-mix', s);
      const rows = D.chunks.map((c, i) => {
        const r = document.createElement('div');
        r.className = 'rankrow'; r.style.opacity = 0;
        r.innerHTML = `<span class="id">${c.id}</span><span class="ttl">${esc(title(c))}</span><span class="sc"></span>`;
        return flip.appendChild(r);
      });
      let q = null, kw, sem;
      const fuse = () => {
        if (!q) return;
        const a = +mix.value / 100;
        $('#hyb-mix-o', s).textContent = a === .5 ? 'equal' : a < .5 ? `keyword ×${(2 - 2 * a).toFixed(1)}` : `meaning ×${(2 * a).toFixed(1)}`;
        const fused = rrf([order(sem).slice(0, 5), order(kw).slice(0, 5).filter(i => kw[i] > 0)], [2 * a, 2 - 2 * a]);
        const pos = new Map(fused.map(([i], r) => [i, r]));
        rows.forEach((r, i) => {
          const p = pos.has(i) ? pos.get(i) : 99;
          r.style.transform = `translateY(${Math.min(p, 5) * ROW}px)`;
          r.style.opacity = p < 5 ? 1 : 0;
          r.classList.toggle('gold', q.gold === D.chunks[i].id);
          if (p < 5) r.lastChild.textContent = fused[p][1].toFixed(4);
        });
      };
      const bar = queryBar($('#hyb-q', s), ['cs240', 'retake', 'guests', 'late', 'scholarship'], x => {
        q = x; kw = bm25(q.text); sem = semantic(q.emb);
        const kt = order(kw).slice(0, 5), st = order(sem).slice(0, 5);
        $('#hyb-kw', s).innerHTML = kt.map(i => rankRow(i, kw[i], kw[kt[0]] || 1, q.gold, 2)).join('');
        $('#hyb-sem', s).innerHTML = st.map(i => rankRow(i, sem[i], sem[st[0]], q.gold)).join('');
        $('#hyb-tokens', s).textContent = tokenize(q.text).join(' · ');
        fuse();
      });
      mix.oninput = fuse;
      this.bar = bar;
    },
    step(n) { if (n === 1 && !this.ran) { this.ran = true; this.bar.first(); } },
  };

  /* ================= Part 6: generation ================= */
  const promptHtml = (q, ctx, strict, cite) => {
    let h = 'You are a helpful assistant for Moghazy University students.\n';
    if (!ctx.length) return h + `\n<span class="qq">Question: ${esc(q)}</span>\nAnswer:`;
    h += '<span class="rule">' + (strict
      ? 'Answer the question using ONLY the context below.\nIf the answer is not in the context, say "I don\'t know based on the handbook."'
      : 'Use the context below to help answer the question.') + '</span>\n';
    if (cite) h += '<span class="rule">Cite the chunk ids you used in square brackets, like [c12].</span>\n';
    h += '\nContext:\n' + ctx.map(id => `<span class="ctx">[${id}] ${esc(D.chunks[idx(id)].text)}</span>`).join('\n');
    return h + `\n\n<span class="qq">Question: ${esc(q)}</span>\nAnswer:`;
  };
  DEMOS['gen-why'] = {
    init(s) {
      $('#gen-why-loose', s).textContent = clip(D.gen['parking|none'] || '(not recorded yet)', 300);
      $('#gen-why-strict', s).textContent = clip(D.gen['parking|k3_s1_c1'] || '(not recorded yet)', 300);
      $('#gen-why-k1', s).textContent = clip(D.gen['retake|k1_s1_c1'] || '(not recorded yet)', 300);
    },
  };
  DEMOS['gen-demo'] = {
    init(s) {
      const ret = $('#gen-ret', s), k = $('#gen-k', s), strict = $('#gen-strict', s), cite = $('#gen-cite', s);
      let q = null, token = 0;
      const draw = async () => {
        if (!q) return;
        const on = ret.checked;
        [k, strict, cite].forEach(e => e.disabled = !on);
        const ctx = on ? D.genCtx[q.id].slice(0, +k.value) : [];
        $('#gen-prompt', s).innerHTML = promptHtml(q.text, ctx, strict.checked, cite.checked);
        $('#gen-words', s).textContent = $('#gen-prompt', s).textContent.split(/\s+/).length + ' words sent to the model';
        const key = `${q.id}|` + (on ? `k${k.value}_s${+strict.checked}_c${+cite.checked}` : 'none');
        const ans = D.gen[key] || '(no recorded answer for this setting; try it in Colab)';
        const out = $('#gen-answer', s), my = ++token;
        out.textContent = '';
        for (let i = 0; i < ans.length && my === token; i += 3) { out.textContent = ans.slice(0, i + 3); await sleep(12); }
      };
      queryBar($('#gen-q', s), ['retake', 'cs240', 'late', 'parking'], x => { q = x; draw(); }, false).first();
      [ret, k, strict, cite].forEach(e => e.onchange = draw);
    },
  };

  /* ================= RAG vs long context calculator ================= */
  /* OpenAI API list prices, Standard tier, USD per 1M tokens, read from the pricing page on 2026-11-12.
     in = uncached input, cached = cached input read, write = 30-minute cache write.
     Requests above LONG input tokens are billed at 2x these input and cache rates. */
  const PRICES = {
    'gpt-6-luna': { in: 0.10, cached: 0.01, write: 0.125 },
    'gpt-6.1-terra': { in: 1.00, cached: 0.05, write: 1.25 },
    'gpt-6.1-sol': { in: 2.00, cached: 0.10, write: 2.50 },
    'gpt-6-astra': { in: 10.00, cached: 1.00, write: 12.50 },
  };
  const WINDOW = 1050000, LONG = 272000;
  DEMOS['lc-calc'] = {
    init(s) {
      const g = id => $('#' + id, s);
      let model = 'gpt-6.1-sol';
      const fmtTok = t => t >= 1e6 ? (t / 1e6).toFixed(2).replace(/\.?0+$/, '') + 'M' : t >= 1e3 ? Math.round(t / 1e3) + 'K' : String(t);
      const money = v => v >= 100 ? '$' + Math.round(v).toLocaleString() : v >= 1 ? '$' + v.toFixed(2) : v >= 0.01 ? '$' + v.toFixed(3) : v >= 0.0001 ? '$' + v.toFixed(5) : '$' + v.toFixed(7);
      const rate = v => '$' + (v < 0.1 ? v.toFixed(3).replace(/0$/, '') : v.toFixed(2));
      const draw = () => {
        const P = PRICES[model];
        const size = Math.round(Math.pow(10, +g('lc-size').value)), qpd = Math.round(Math.pow(10, +g('lc-q').value)), ragTok = +g('lc-rag').value;
        g('lc-models').querySelectorAll('.chip').forEach(c => c.classList.toggle('sel', c.textContent === model));
        g('lc-rates').innerHTML = `input ${rate(P.in)} · cached input ${rate(P.cached)} · cache write ${rate(P.write)}<br>context window ${WINDOW.toLocaleString()} tokens`;
        g('lc-size-o').textContent = fmtTok(size) + ' tokens'; g('lc-q-o').textContent = qpd.toLocaleString() + ' / day';
        g('lc-rag-o').textContent = ragTok.toLocaleString() + ' tokens';
        const fits = size <= WINDOW, mult = size > LONG ? 2 : 1, tier = mult === 2 ? ' × 2 (over 272K)' : '';
        const rows = [
          ['Long context, no caching', size * P.in * mult / 1e6, 'bad', `${fmtTok(size)} × ${rate(P.in)} / 1M${tier}`, true],
          ['Long context, documents cached', size * P.cached * mult / 1e6, 'work', `${fmtTok(size)} × ${rate(P.cached)} / 1M${tier}`, true],
          ['RAG (top chunks only)', ragTok * P.in / 1e6, 'hit', `${ragTok.toLocaleString()} × ${rate(P.in)} / 1M`, false],
        ];
        const max = Math.max(...rows.map(r => r[1]));
        g('lc-bars').innerHTML = rows.map(([name, c, cls, formula, lc]) => `<div style="margin-bottom:11px;${!fits && lc ? 'opacity:.35' : ''}">
          <div class="row" style="justify-content:space-between;font-size:17px"><span>${name}</span><span class="mono"><b>${money(c)}</b> per question · ${money(c * qpd)} per day</span></div>
          <div style="height:14px;border-radius:5px;background:var(--${cls});width:${Math.max(0.6, 100 * c / max)}%"></div>
          <div class="mono" style="font-size:13.5px;color:var(--muted);margin-top:2px">= ${formula}</div></div>`).join('');
        const v = g('lc-fit');
        v.className = 'verdict ' + (fits ? 'ok' : 'no');
        v.textContent = fits ? `✓ fits in the ${fmtTok(WINDOW)} window` : `✗ does not fit in the ${fmtTok(WINDOW)} window: long context is not an option`;
        g('lc-ratio').innerHTML = fits
          ? `Each time the cache has expired, the next question pays to write it again: <b>${money(size * P.write * mult / 1e6)}</b>. With the cache warm, long context costs <b>${(size * P.cached * mult / (ragTok * P.in)).toFixed(1)}×</b> what RAG costs per question.`
          : 'You have to choose what to send. That choosing step is retrieval.';
      };
      Object.keys(PRICES).forEach(m => {
        const c = document.createElement('button');
        c.className = 'chip'; c.textContent = m; c.onclick = () => { model = m; draw(); };
        g('lc-models').appendChild(c);
      });
      s.querySelectorAll('input[type=range]').forEach(e => e.oninput = draw);
      draw();
    },
  };

  /* ================= "where we are" map slides ================= */
  const STAGES = ['docs', 'chunks', 'emb', 'index', 'question', 'qemb', 'search', 'prompt', 'llm', 'answer'];
  const EDGE_TO = ['chunks', 'emb', 'index', 'search', 'qemb', 'search', 'prompt', 'llm', 'answer'];
  const FLOW = [['question'], ['docs', 'chunks'], ['emb', 'qemb'], ['index'], ['search'], ['prompt', 'llm', 'answer']];
  document.querySelectorAll('.slide[data-focus]').forEach(sl => {
    DEMOS[sl.id] = {
      init(s) {
        const step = +s.dataset.focus, here = FLOW[step], was = FLOW.slice(0, step).flat();
        const c = $('#pipe-svg').cloneNode(true);
        c.removeAttribute('id'); c.classList.add('focus');
        c.querySelectorAll('[data-at]').forEach(e => { e.removeAttribute('data-at'); e.classList.remove('active', 'done'); });
        const mark = (e, name) => { if (here.includes(name)) e.classList.add('here'); else if (was.includes(name)) e.classList.add('was'); };
        c.querySelectorAll('.stage').forEach((e, i) => mark(e, STAGES[i]));
        c.querySelectorAll('.edge').forEach((e, i) => mark(e, EDGE_TO[i]));
        $('.mapholder', s).appendChild(c);
      },
    };
  });

  /* ================= Part 9: evaluation bars ================= */
  DEMOS['ev-bars'] = {
    init(s) {
      const M = {
        recall: { name: 'Context recall', fix: 'split the text into better chunks, retrieve more of them, or add keyword search' },
        precision: { name: 'Context precision', fix: 'add a reranker, or retrieve fewer chunks' },
        faith: { name: 'Faithfulness', fix: 'tighten the prompt (answer only from the context) and lower the temperature' },
        rel: { name: 'Response relevancy', fix: 'ask for short, direct answers in the prompt' },
        acc: { name: 'Answer accuracy', fix: 'read ten wrong answers beside their chunks, and check the reference answers' },
      };
      const GROUPS = [
        { title: 'Retrieval', q: 'did the right text reach the model?', ids: ['recall', 'precision'], pic: 'ret' },
        { title: 'Generation', q: 'did the model use that text well?', ids: ['faith', 'rel'], pic: 'gen' },
        { title: 'End to end', q: 'is the final answer right?', ids: ['acc'], pic: 'acc' },
      ];
      const IDS = Object.keys(M);
      const PRESETS = [
        ['Healthy', [.9, .9, .95, .85, .9]],
        ['Retrieval misses', [.35, .7, .85, .8, .4]],
        ['Buried in noise', [.9, .3, .7, .65, .7]],
        ['Making things up', [.9, .85, .3, .8, .45]],
        ['Faithful but wrong', [.55, .5, .95, .85, .2]],
        ['Dodging the question', [.85, .8, .6, .25, .3]],
      ];
      const band = v => v >= 0.8 ? 2 : v >= 0.5 ? 1 : 0;
      const col = v => ['var(--bad)', 'var(--work)', 'var(--hit)'][band(v)];
      const rows = $('#eb-rows', s);
      rows.innerHTML = GROUPS.map(g => `<div class="ev2 panel">
        <div class="ev2-bars"><h3>${g.title} <span>${g.q}</span></h3>
          ${g.ids.map(id => `<div class="ev2-row"><b>${M[id].name}</b><input type="range" class="ebar" id="eb-${id}" min="0" max="1" step="0.05" value="0.9" aria-label="${M[id].name}"><output id="eb-${id}-o"></output></div>`).join('')}
        </div>
        <div class="ev2-pic" id="ep-${g.pic}"></div></div>`).join('');

      const pics = {
        ret(v) {                               // four retrieved chunks, two of which the answer needs
          const found = v.recall >= 0.75 ? 2 : v.recall >= 0.25 ? 1 : 0, noise = 4 - found;
          const u = Array(found).fill('u'), n = Array(noise).fill('n');
          const order = !found || v.precision >= 0.8 ? [...u, ...n] : v.precision >= 0.5 ? ['n', ...u, ...n.slice(1)] : [...n, ...u];
          const blocks = order.map((k, i) => `<span class="ck ${k}"><i>#${i + 1}</i>${k === 'u' ? 'useful' : 'noise'}</span>`).join('') +
            Array(2 - found).fill('<span class="ck miss">needed,<br>not retrieved</span>').join('');
          const cap = (found === 2 ? 'Both chunks the answer needs were retrieved' : found === 1 ? 'One chunk the answer needs is missing' : 'Neither chunk the answer needs was retrieved') +
            (!found ? '.' : v.precision >= 0.8 ? ', and the useful ones are on top.' : v.precision >= 0.5 ? ', with some noise ranked above.' : ', but the useful ones are buried under noise.');
          return `<div class="picrow"><em>retrieved, in rank order</em>${blocks}</div><p>${cap}</p>`;
        },
        gen(v) {                               // a four-sentence answer
          const flagged = x => x >= 0.8 ? 0 : Math.max(1, Math.round((1 - x) * 4));   // a healthy score flags nothing
          const bad = flagged(v.faith), off = flagged(v.rel), w = [92, 78, 86, 64];
          const lines = w.map((x, i) => {
            const uns = i >= 4 - bad, offt = i >= 4 - off;
            return `<div class="aline${uns ? ' uns' : ''}${offt ? ' off' : ''}"><i style="width:${x * 0.5}%"></i><b>${uns ? 'not in any chunk' : 'backed by a chunk'}</b>${offt ? '<u>off the question</u>' : ''}</div>`;
          }).join('');
          const cap = (bad === 0 ? 'Every sentence is backed by a chunk' : bad === 4 ? 'No sentence is backed by a chunk' : `${bad} of 4 sentences ${bad > 1 ? 'are' : 'is'} not backed by any chunk`) +
            (off === 0 ? ', and all of it answers the question.' : off >= 3 ? ', and most of it does not answer the question.' : `, and ${off} ${off > 1 ? 'drift' : 'drifts'} away from the question.`);
          return `<div class="alines">${lines}</div><p>${cap}</p>`;
        },
        acc(v) {                               // four facts in the reference answer
          const m = Math.round(v.acc * 4);
          const facts = [0, 1, 2, 3].map(i => `<span class="fact ${i < m ? 'y' : 'n'}">${i < m ? '✓' : '✗'}</span>`).join('');
          return `<div class="picrow"><em>facts in the reference answer</em>${facts}<p style="margin:0 0 0 10px">${m} of 4 match. ${['The answer disagrees with the reference.', 'The answer is only partly right.', 'The answer agrees with the reference.'][band(v.acc)]}</p></div>`;
        },
      };

      const draw = () => {
        const v = Object.fromEntries(IDS.map(id => [id, +$('#eb-' + id, s).value]));
        IDS.forEach(id => {
          const el = $('#eb-' + id, s);
          el.style.setProperty('--fill', col(v[id])); el.style.setProperty('--pct', v[id] * 100 + '%');
          const o = $(`#eb-${id}-o`, s); o.textContent = v[id].toFixed(2); o.style.color = col(v[id]);
        });
        GROUPS.forEach(g => { $('#ep-' + g.pic, s).innerHTML = pics[g.pic](v); });
        const low = id => v[id] < 0.5, ok = id => v[id] >= 0.8;
        let head, tone = 'bad';
        if (IDS.every(ok)) { head = '<b>Healthy.</b> Keep these numbers as your baseline and run the same questions again after every change.'; tone = 'hit'; }
        else if (low('recall')) head = '<b>Start with retrieval.</b> The text the answer needs is not reaching the model, and no prompt can fix that.';
        else if (low('acc') && ok('faith')) head = '<b>Faithful but wrong.</b> The model repeats its context accurately, and the context is wrong or incomplete. Look at retrieval before the prompt.';
        else if (low('faith')) head = '<b>A generation problem.</b> Retrieval is delivering, and the model is not sticking to what it was given.';
        else if (low('precision')) head = '<b>Found, but buried.</b> The right chunks are retrieved and outranked by noise.';
        else if (low('rel')) head = '<b>Not answering the question.</b> The answers may be true and supported, but they do not address what was asked.';
        else if (low('acc')) head = '<b>Upstream looks fine, yet answers disagree with the references.</b> Read some cases by hand.';
        else { head = '<b>No single failure.</b> Nothing is broken, but there is room to improve.'; tone = 'work'; }
        const todo = IDS.filter(id => !ok(id)).sort((a, b) => v[a] - v[b]).slice(0, 2);
        const steps = todo.map((id, i) => `<span class="step"><em>${i ? 'Then' : 'First'}</em> <b style="color:${col(v[id])}">${M[id].name} ${v[id].toFixed(2)}</b>: ${M[id].fix}.</span>`).join('');
        const h = $('#eb-head', s); h.className = 'card ' + tone;
        h.innerHTML = head + (steps ? `<div class="steps">${steps}</div>` : '');
        $('#eb-presets', s).querySelectorAll('.chip').forEach((c, i) => c.classList.toggle('sel', PRESETS[i][1].every((x, k) => x === v[IDS[k]])));
      };
      PRESETS.forEach(([label, vals]) => {
        const c = document.createElement('button');
        c.className = 'chip'; c.textContent = label;
        c.onclick = () => { IDS.forEach((id, k) => $('#eb-' + id, s).value = vals[k]); draw(); };
        $('#eb-presets', s).appendChild(c);
      });
      rows.querySelectorAll('input').forEach(e => e.oninput = draw);
      draw();
    },
  };
})();
